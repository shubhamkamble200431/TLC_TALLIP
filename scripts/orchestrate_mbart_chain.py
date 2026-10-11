#!/usr/bin/env python
"""
Sequences the requested mBART pipeline order under the DGX's hard 4-job submit
cap (can't just pre-submit a 5-job dependency chain at once):

    NLLB baseline + mBART baseline (parallel)
        -> mBART BBS search (its own selected_bbs_weights.json)
            -> mBART 48-run factorial, halves A/B (run_c / run_d)

State lives in jobs/slurm/chain_state.json.  Idempotent: run via cron every
~10 min; it only submits a stage's job(s) once (checked via job-id fields in
the state file) and only advances once sacct reports every job of the current
stage in a terminal state.  Bootstrap by writing the initial
{"stage": "baselines_running", "baseline_nllb_job": "...", "baseline_mbart_job": "..."}
after submitting those two jobs by hand.

    python scripts/orchestrate_mbart_chain.py
"""
import datetime
import json
import os
import subprocess

REPO = "/raid/kahuja/TLC_TALLIP"
STATE_PATH = os.path.join(REPO, "jobs", "slurm", "chain_state.json")
LOG_PATH = os.path.join(REPO, "jobs", "slurm", "logs", "chain_orchestrator.log")
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "DEADLINE"}


def log(msg: str) -> None:
    line = f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def load_state() -> dict:
    if os.path.isfile(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {}


def save_state(s: dict) -> None:
    with open(STATE_PATH, "w") as f:
        json.dump(s, f, indent=2)


def job_state(job_id):
    if not job_id:
        return None
    r = subprocess.run(["sacct", "-n", "-X", "-j", str(job_id), "-o", "State"],
                        capture_output=True, text=True, timeout=20)
    lines = [ln.strip() for ln in r.stdout.strip().splitlines() if ln.strip()]
    return lines[0].split()[0] if lines else None  # "CANCELLED by 1013" -> "CANCELLED"


def is_terminal(state) -> bool:
    # sacct can suffix e.g. "CANCELLED+" (cancelled while still pending/queued)
    return state is not None and state.rstrip("+") in TERMINAL


def sbatch(extra_args: list) -> str | None:
    cmd = ["sbatch", "--parsable"] + extra_args
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=30)
    jid = r.stdout.strip()
    if not jid.isdigit():
        log(f"sbatch FAILED: {cmd} -> stdout={r.stdout!r} stderr={r.stderr!r}")
        return None
    log(f"submitted {cmd} -> job {jid}")
    return jid


def main():
    s = load_state()
    stage = s.get("stage")
    if stage is None:
        log("no jobs/slurm/chain_state.json / no 'stage' -> nothing to do "
            "(bootstrap by submitting the two baseline jobs and writing the state file)")
        return

    if stage == "baselines_running":
        # baseline_jobs: list of job ids (the 2 balanced multi-architecture
        # baseline jobs).  Falls back to the older 2 named fields if present
        # (pre-multi-architecture chain state).
        job_ids = s.get("baseline_jobs") or [j for j in
                  (s.get("baseline_nllb_job"), s.get("baseline_mbart_job")) if j]
        states = {jid: job_state(jid) for jid in job_ids}
        log(f"stage=baselines_running  {states}")
        if job_ids and all(is_terminal(v) for v in states.values()):
            s["baseline_final_states"] = states
            for jid, st in states.items():
                if st != "COMPLETED":
                    log(f"WARNING: baseline job {jid} ended {st}")
            if not s.get("bbs_search_job"):
                jid = sbatch(["--export=ALL", "-J", "bbs_mbart",
                              os.path.join(REPO, "jobs/slurm/bbs_search_mbart.slurm")])
                if jid:
                    s["bbs_search_job"] = jid
            if s.get("bbs_search_job"):
                s["stage"] = "bbs_search_running"
            save_state(s)

    elif stage == "bbs_search_running":
        st = job_state(s.get("bbs_search_job"))
        sel_path = os.path.join(REPO, "experiments", "v2", "bbs_search_mbart", "selected_bbs_weights.json")
        log(f"stage=bbs_search_running  bbs_search_mbart={st}  selected_file_exists={os.path.isfile(sel_path)}")
        if is_terminal(st):
            s["bbs_search_final_state"] = st
            # run_split.slurm passes --bbs-weights selected, which HARD-ERRORS
            # (SystemExit) on every one of the 48 runs if this file is missing
            # -- never submit run_c/run_d into that; stop and surface it instead.
            if st != "COMPLETED" or not os.path.isfile(sel_path):
                log(f"ERROR: bbs_search_mbart job {s.get('bbs_search_job')} ended {st} and "
                    f"{sel_path} {'exists' if os.path.isfile(sel_path) else 'is MISSING'} -- "
                    "NOT submitting run_c/run_d (they would hard-fail with no weights file). "
                    "Chain paused at stage=bbs_search_blocked; needs a human look "
                    "(re-run jobs/slurm/bbs_search_mbart.slurm, or reset stage to retry).")
                s["stage"] = "bbs_search_blocked"
                save_state(s)
                return
            # explicit ordering: bbs_mbart -> baseline_mt5 -> run_c/run_d (user
            # request) -- don't submit the 48-run factorial until mt5 is also
            # done, even though the search itself is unblocked.
            mt5_job = s.get("mt5_job")
            if mt5_job:
                mt5_st = job_state(mt5_job)
                log(f"  waiting on baseline_mt5 job {mt5_job} before run_c/run_d: {mt5_st}")
                if not is_terminal(mt5_st):
                    save_state(s)
                    return
                s["mt5_final_state"] = mt5_st
            if not s.get("mbart_factorial_a_job"):
                jid = sbatch(["--export=ALL,EXPERIMENT=mbart50,HALF=a", "-J", "run_c",
                              os.path.join(REPO, "jobs/slurm/run_split.slurm")])
                if jid:
                    s["mbart_factorial_a_job"] = jid
            if not s.get("mbart_factorial_b_job"):
                jid = sbatch(["--export=ALL,EXPERIMENT=mbart50,HALF=b", "-J", "run_d",
                              os.path.join(REPO, "jobs/slurm/run_split.slurm")])
                if jid:
                    s["mbart_factorial_b_job"] = jid
            if s.get("mbart_factorial_a_job") and s.get("mbart_factorial_b_job"):
                s["stage"] = "mbart48_running"
            save_state(s)

    elif stage == "mbart48_running":
        log(f"stage=mbart48_running  run_c={s.get('mbart_factorial_a_job')} "
            f"run_d={s.get('mbart_factorial_b_job')}  -- chain complete, nothing left to orchestrate")
        s["stage"] = "done"
        save_state(s)

    elif stage == "done":
        pass

    elif stage == "bbs_search_blocked":
        log("stage=bbs_search_blocked -- waiting for a human to fix the mBART BBS search "
            "and reset jobs/slurm/chain_state.json (nothing auto-submitted)")

    else:
        log(f"unknown stage {stage!r}")


if __name__ == "__main__":
    main()

"""
Per-run status tracking (brief sections 3, 26).

``status.json`` moves through:  started -> running -> {completed | failed | interrupted}

The tracker installs SIGINT/SIGTERM handlers and an atexit hook so a killed GPU
job is recorded as ``interrupted`` (not silently left as ``running``), and never
as ``completed``.
"""
from __future__ import annotations

import atexit
import json
import os
import signal
import traceback
from datetime import datetime, timezone
from typing import Optional

STATES = ("started", "running", "completed", "failed", "interrupted")


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


class StatusTracker:
    def __init__(self, run_dir: str, meta: Optional[dict] = None):
        self.path = os.path.join(run_dir, "status.json")
        os.makedirs(run_dir, exist_ok=True)
        self.data = {
            "status": "started",
            "created": _now(),
            "updated": _now(),
            "started_at": None,
            "ended_at": None,
            "failure_reason": None,
            "history": [],
            "meta": meta or {},
        }
        self._final = False
        self._prev_handlers = {}
        self._save()

    # ------------------------------------------------------------------
    def _save(self):
        self.data["updated"] = _now()
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp, self.path)

    def set(self, status: str, failure_reason: Optional[str] = None):
        assert status in STATES, status
        self.data["status"] = status
        self.data["history"].append({"status": status, "at": _now(), "reason": failure_reason})
        if status == "running" and self.data["started_at"] is None:
            self.data["started_at"] = _now()
        if status in ("completed", "failed", "interrupted"):
            self.data["ended_at"] = _now()
            self._final = True
        if failure_reason:
            self.data["failure_reason"] = failure_reason
        self._save()

    def update_meta(self, **kw):
        self.data["meta"].update(kw)
        self._save()

    # ------------------------------------------------------------------
    def __enter__(self):
        self.set("running")
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._prev_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._on_signal)
        atexit.register(self._on_atexit)
        return self

    def __exit__(self, exc_type, exc, tb):
        for sig, h in self._prev_handlers.items():
            try:
                signal.signal(sig, h)
            except Exception:
                pass
        if exc_type is None:
            if not self._final:
                self.set("completed")
            return False
        reason = f"{exc_type.__name__}: {exc}"
        self.set("failed", failure_reason=reason)
        with open(os.path.join(os.path.dirname(self.path), "traceback.txt"), "w") as f:
            f.write("".join(traceback.format_exception(exc_type, exc, tb)))
        return False  # re-raise

    def _on_signal(self, signum, frame):
        if not self._final:
            self.set("interrupted", failure_reason=f"signal {signum}")
        raise KeyboardInterrupt(f"signal {signum}")

    def _on_atexit(self):
        if not self._final:
            self.set("interrupted", failure_reason="process exited without finalising")


def read_status(run_dir: str) -> Optional[dict]:
    p = os.path.join(run_dir, "status.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def is_completed(run_dir: str) -> bool:
    s = read_status(run_dir)
    return bool(s and s.get("status") == "completed")

import sys, time
import train_mamba_baseline as m
m.EPOCHS = 1
orig_load = m.load_parallel_corpus
def small_load(data_dir, s, t):
    src, tgt = orig_load(data_dir, s, t)
    return src[:2000], tgt[:2000]
m.load_parallel_corpus = small_load
sys.argv = ["x", "--size", "small", "--seed", "999", "--batch-size", sys.argv[1]]
t0=time.time()
m.main()
print("TOTAL", time.time()-t0)

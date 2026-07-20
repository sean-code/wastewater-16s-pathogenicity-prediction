#!/usr/bin/env python3
"""
train_16s_model.py  --  train Model B (the 16S k-mer pathogenicity Random Forest).

Feature encoding matches the GOMICS 16S design: the canonical k-mer frequency
vector of each 16S sequence. For k=6 the 4^6 = 4096 possible 6-mers are collapsed
by pairing each k-mer with its reverse complement (DNA is double-stranded), giving
(4096 + 64)/2 = 2080 strand-invariant features. Training uses fragment
augmentation (random sub-windows) so the classifier is robust to short amplicon
reads. The trained model + feature index are written to saved_model_16s/.

Input: labeled_16s.fasta (headers: id|label|genus) from labeled_16s_build.py.

Usage:
    python train_16s_model.py --labeled saved_model_16s/labeled_16s.fasta \
        --kmer-k 6 --augment 3 --n-estimators 400
"""
import argparse, itertools, json, pickle, random
from pathlib import Path
import numpy as np

COMP = str.maketrans("ACGT", "TGCA")
def rc(s): return s.translate(COMP)[::-1]

def canonical_index(k):
    keys = sorted({min(km, rc(km)) for km in
                   map("".join, itertools.product("ACGT", repeat=k))})
    return {km: i for i, km in enumerate(keys)}

def kmer_vector(seq, k, idx):
    seq = seq.upper(); v = np.zeros(len(idx), np.float32); n = 0
    for i in range(len(seq) - k + 1):
        km = seq[i:i+k]
        if any(c not in "ACGT" for c in km):
            continue
        j = idx.get(min(km, rc(km)))
        if j is not None:
            v[j] += 1; n += 1
    return v / n if n else v

def read_fasta(path):
    name, buf = None, []
    for ln in open(path):
        if ln.startswith(">"):
            if name: yield name, "".join(buf)
            name, buf = ln[1:].strip(), []
        else:
            buf.append(ln.strip())
    if name: yield name, "".join(buf)

def fragments(seq, n, lo=250, hi=500, rng=random):
    out, L = [], len(seq)
    for _ in range(n):
        if L <= lo: break
        wl = rng.randint(lo, min(hi, L)); s = rng.randint(0, L - wl)
        out.append(seq[s:s+wl])
    return out

LABEL_MAP = {"pathogen": 1, "non_pathogen": 0, "nonpathogen": 0, "1": 1, "0": 0}

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labeled", default="saved_model_16s/labeled_16s.fasta")
    ap.add_argument("--kmer-k", type=int, default=6)
    ap.add_argument("--augment", type=int, default=3, help="fragments per sequence")
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="saved_model_16s")
    args = ap.parse_args()

    rng = random.Random(args.seed); np.random.seed(args.seed)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    idx = canonical_index(args.kmer_k)
    X, y = [], []
    for hdr, seq in read_fasta(args.labeled):
        parts = hdr.split("|")
        if len(parts) < 2:
            continue
        lab = LABEL_MAP[parts[1].strip().lower()]
        X.append(kmer_vector(seq, args.kmer_k, idx)); y.append(lab)
        for fr in fragments(seq, args.augment, rng=rng):
            X.append(kmer_vector(fr, args.kmer_k, idx)); y.append(lab)
    X, y = np.vstack(X), np.array(y)
    print(f"[train] matrix {X.shape} | k={args.kmer_k} -> {len(idx)} features | "
          f"positives {int(y.sum())}/{len(y)}")

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=args.n_estimators,
                                class_weight="balanced",
                                random_state=args.seed, n_jobs=-1).fit(X, y)

    model_path = out / "kmer16s_model.pkl"
    index_path = out / "kmer_index.json"
    pickle.dump(rf, open(model_path, "wb"))
    json.dump({str(i): km for km, i in idx.items()}, open(index_path, "w"))
    print(f"[done] Model B -> {model_path}")
    print(f"       feature index ({len(idx)} canonical {args.kmer_k}-mers) -> {index_path}")

if __name__ == "__main__":
    main()

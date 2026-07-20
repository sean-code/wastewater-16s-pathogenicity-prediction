#!/usr/bin/env python3
"""
validate_16s_model.py
=====================================================================
Ground-truth validation harness for Model B (the 16S k-mer pathogenicity
Random Forest of the GOMICS 16S track).

It computes the REAL classification accuracy the paper needs, using the
1,432 labeled 16S genes (826 pathogen + 606 non-pathogen), under two schemes:

  1. Stratified k-fold CV        -> standard performance (in-distribution).
  2. Leave-one-genus-out CV      -> generalization to UNSEEN genera. This is
     the leakage-controlled number: every test genus is fully removed from
     training, so a correct call cannot come from having memorized the genus.

Metrics: accuracy, balanced accuracy, sensitivity (recall+), specificity,
precision, F1, MCC, AUROC, AUPRC, plus a calibration/reliability curve and
Brier score (the RF is known to lean positive at 0.5).

Feature encoding matches Model B exactly: canonical k-mer frequency vector
(k=6 -> 2,080 strand-invariant features). Optional fragment augmentation makes
the estimate representative of short amplicon reads.

-------------------------------------------------------------------
INPUTS (pick ONE of the two labeled-data forms):
  (a) --labeled-fasta FILE
        A FASTA whose headers encode label and (optionally) genus, e.g.
        >GCF_000005845|pathogen|Escherichia
        >seq123|non_pathogen|Lactobacillus
        Delimiter configurable via --header-sep (default '|').
        Field order: id, label, genus  (genus optional -> parsed from organism).
  (b) --labels-csv FILE  --seqs-fasta FILE
        CSV with columns: seq_id,label[,genus]  and a matching FASTA.

Label values may be: pathogen / non_pathogen (aliases: 1/0, path/nonpath,
pos/neg, true/false) -- see LABEL_MAP.

OUTPUT (into --outdir, default ./model_b_validation):
  metrics_summary.json         all headline numbers
  cv_stratified_folds.csv      per-fold metrics (stratified k-fold)
  cv_leave_genus_out.csv       per-genus held-out metrics
  confusion_stratified.csv     pooled out-of-fold confusion matrix
  confusion_leavegenus.csv     pooled out-of-fold confusion matrix (LOGO)
  fig_roc.png                  ROC curves (both schemes)
  fig_confusion.png            confusion matrices
  fig_calibration.png          reliability curve + Brier score
-------------------------------------------------------------------
USAGE (HPC):
  python validate_16s_model.py --labeled-fasta labeled_16s.fasta \
      --kmer-k 6 --folds 5 --augment 3 --outdir model_b_validation
"""
import argparse, json, os, sys, itertools, random
from collections import Counter
import numpy as np

# ------------------------------ label handling ------------------------------
LABEL_MAP = {
    "pathogen": 1, "path": 1, "pos": 1, "positive": 1, "1": 1, "true": 1, "p": 1,
    "non_pathogen": 0, "nonpathogen": 0, "non-pathogen": 0, "nonpath": 0,
    "neg": 0, "negative": 0, "0": 0, "false": 0, "np": 0, "commensal": 0,
}

def norm_label(x):
    k = str(x).strip().lower().replace(" ", "_")
    if k not in LABEL_MAP:
        raise ValueError(f"Unrecognized label {x!r}. Extend LABEL_MAP if needed.")
    return LABEL_MAP[k]

# ------------------------------ FASTA I/O -----------------------------------
def read_fasta(path):
    name, seq = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq)
                name, seq = line[1:], []
            else:
                seq.append(line.strip())
    if name is not None:
        yield name, "".join(seq)

# ------------------------------ canonical k-mer featurizer ------------------
_COMP = str.maketrans("ACGT", "TGCA")
def _revcomp(s): return s.translate(_COMP)[::-1]

def build_canonical_kmers(k):
    """Ordered list of canonical k-mers (a k-mer paired with its rev-comp;
    the lexicographically smaller of the pair is the canonical key)."""
    keys = {}
    for tup in itertools.product("ACGT", repeat=k):
        km = "".join(tup)
        can = min(km, _revcomp(km))
        keys[can] = True
    return sorted(keys)

def kmer_vector(seq, k, index):
    seq = seq.upper()
    v = np.zeros(len(index), dtype=np.float32)
    n = 0
    for i in range(len(seq) - k + 1):
        km = seq[i:i+k]
        if any(c not in "ACGT" for c in km):
            continue
        can = min(km, _revcomp(km))
        j = index.get(can)
        if j is not None:
            v[j] += 1.0
            n += 1
    if n > 0:
        v /= n                      # frequency, not raw count -> length-normalized
    return v

def random_fragments(seq, n_frag, min_len=250, max_len=500, rng=None):
    """Yield random sub-windows to emulate short amplicon reads."""
    rng = rng or random
    L = len(seq)
    out = []
    if L <= min_len:
        return out
    for _ in range(n_frag):
        wl = rng.randint(min_len, min(max_len, L))
        start = rng.randint(0, L - wl)
        out.append(seq[start:start+wl])
    return out

# ------------------------------ data loading --------------------------------
def load_labeled(args):
    """Return records: list of dicts {id, seq, y, genus}."""
    recs = []
    if args.labeled_fasta:
        sep = args.header_sep
        for hdr, seq in read_fasta(args.labeled_fasta):
            parts = [p.strip() for p in hdr.split(sep)]
            if len(parts) < 2:
                raise ValueError(f"Header {hdr!r} lacks a label field (sep={sep!r}).")
            sid, lab = parts[0], parts[1]
            genus = parts[2] if len(parts) > 2 and parts[2] else _guess_genus(hdr)
            recs.append(dict(id=sid, seq=seq, y=norm_label(lab), genus=genus))
    else:
        import csv
        seqs = {sid.split()[0]: s for sid, s in read_fasta(args.seqs_fasta)}
        with open(args.labels_csv) as fh:
            for row in csv.DictReader(fh):
                sid = row["seq_id"].strip()
                if sid not in seqs:
                    continue
                genus = (row.get("genus") or "").strip() or _guess_genus(sid)
                recs.append(dict(id=sid, seq=seqs[sid], y=norm_label(row["label"]),
                                 genus=genus))
    if not recs:
        sys.exit("No labeled records loaded -- check inputs.")
    return recs

def _guess_genus(text):
    """Best-effort genus from an organism string; else 'unknown'."""
    for tok in text.replace("|", " ").replace("_", " ").split():
        if tok[:1].isupper() and tok.isalpha() and len(tok) > 2:
            return tok
    return "unknown"

# ------------------------------ metrics -------------------------------------
def binary_metrics(y_true, y_pred, y_prob):
    from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
        f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
        balanced_accuracy_score, accuracy_score)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    out = dict(
        n=int(len(y_true)), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        sensitivity_recall=float(recall_score(y_true, y_pred, zero_division=0)),
        specificity=float(spec),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        mcc=float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else float("nan"),
    )
    try:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
        out["auprc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        out["auroc"] = out["auprc"] = float("nan")
    return out

# ------------------------------ CV drivers ----------------------------------
def featurize(recs, k, index, augment, rng):
    """Return X (base features, 1 row/record) and metadata; augmentation is
    applied per-fold on TRAIN only, so here we just cache base vectors."""
    X = np.vstack([kmer_vector(r["seq"], k, index) for r in recs])
    y = np.array([r["y"] for r in recs])
    groups = np.array([r["genus"] for r in recs])
    return X, y, groups

def make_rf(args, seed):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=args.n_estimators, class_weight="balanced",
        random_state=seed, n_jobs=args.n_jobs, min_samples_leaf=1)

def _augmented_train(recs, tr_idx, k, index, augment, rng):
    """Build an augmented training matrix from the training records only."""
    Xs, ys = [], []
    for i in tr_idx:
        r = recs[i]
        Xs.append(kmer_vector(r["seq"], k, index)); ys.append(r["y"])
        for frag in random_fragments(r["seq"], augment, rng=rng):
            Xs.append(kmer_vector(frag, k, index)); ys.append(r["y"])
    return np.vstack(Xs), np.array(ys)

def run_cv(recs, k, index, splitter, args, scheme_name, rng):
    """Generic out-of-fold CV: returns pooled preds/probs + per-fold rows."""
    y_all = np.array([r["y"] for r in recs])
    groups = np.array([r["genus"] for r in recs])
    oof_prob = np.full(len(recs), np.nan)
    fold_rows = []
    split_iter = (splitter.split(np.zeros(len(recs)), y_all, groups)
                  if scheme_name == "leave_genus_out"
                  else splitter.split(np.zeros(len(recs)), y_all))
    for fold, (tr, te) in enumerate(split_iter):
        if len(set(y_all[tr])) < 2:
            continue  # skip degenerate training folds
        if args.augment > 0:
            Xtr, ytr = _augmented_train(recs, tr, k, index, args.augment, rng)
        else:
            Xtr = np.vstack([kmer_vector(recs[i]["seq"], k, index) for i in tr])
            ytr = y_all[tr]
        Xte = np.vstack([kmer_vector(recs[i]["seq"], k, index) for i in te])
        clf = make_rf(args, args.seed + fold)
        clf.fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:, list(clf.classes_).index(1)]
        oof_prob[te] = prob
        pred = (prob >= args.threshold).astype(int)
        row = binary_metrics(y_all[te], pred, prob)
        row["fold"] = fold
        if scheme_name == "leave_genus_out":
            row["held_out_genus"] = str(groups[te][0]) if len(te) else ""
        fold_rows.append(row)
    mask = ~np.isnan(oof_prob)
    pooled_pred = (oof_prob[mask] >= args.threshold).astype(int)
    pooled = binary_metrics(y_all[mask], pooled_pred, oof_prob[mask])
    return pooled, fold_rows, y_all[mask], oof_prob[mask]

# ------------------------------ plotting ------------------------------------
def make_figs(outdir, results):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, confusion_matrix
    from sklearn.calibration import calibration_curve
    # ROC
    plt.figure(figsize=(5, 5))
    for name, r in results.items():
        yt, yp = r["y_true"], r["y_prob"]
        fpr, tpr, _ = roc_curve(yt, yp)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUROC={r['pooled']['auroc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="#999")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("Model B ROC (out-of-fold)"); plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "fig_roc.png"), dpi=200); plt.close()
    # confusion matrices
    fig, axes = plt.subplots(1, len(results), figsize=(4*len(results), 3.6))
    if len(results) == 1: axes = [axes]
    for ax, (name, r) in zip(axes, results.items()):
        cm = confusion_matrix(r["y_true"], (np.array(r["y_prob"]) >= r["threshold"]).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.max()/2 else "black", fontsize=13)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["non-path", "pathogen"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["non-path", "pathogen"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(name, fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "fig_confusion.png"), dpi=200); plt.close()
    # calibration (use leave-genus-out if present else first)
    key = "leave_genus_out" if "leave_genus_out" in results else list(results)[0]
    r = results[key]
    frac_pos, mean_pred = calibration_curve(r["y_true"], r["y_prob"], n_bins=10, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot(mean_pred, frac_pos, "o-", label=key)
    plt.plot([0, 1], [0, 1], "--", color="#999", label="perfectly calibrated")
    plt.xlabel("Mean predicted P(pathogen)"); plt.ylabel("Observed pathogen fraction")
    plt.title(f"Calibration (Brier={r['pooled'].get('brier', float('nan')):.3f})")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_calibration.png"), dpi=200); plt.close()

# ------------------------------ main ----------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("labeled data (choose one form)")
    src.add_argument("--labeled-fasta")
    src.add_argument("--labels-csv"); src.add_argument("--seqs-fasta")
    src.add_argument("--header-sep", default="|")
    ap.add_argument("--kmer-k", type=int, default=6)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--augment", type=int, default=3, help="fragments per training seq (0=off)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-genus-size", type=int, default=1,
                    help="min records for a genus to form its own LOGO fold")
    ap.add_argument("--outdir", default="model_b_validation")
    args = ap.parse_args()

    if not args.labeled_fasta and not (args.labels_csv and args.seqs_fasta):
        ap.error("provide --labeled-fasta OR (--labels-csv AND --seqs-fasta)")
    os.makedirs(args.outdir, exist_ok=True)
    rng = random.Random(args.seed); np.random.seed(args.seed)

    recs = load_labeled(args)
    n_pos = sum(r["y"] for r in recs); n_neg = len(recs) - n_pos
    genera = Counter(r["genus"] for r in recs)
    print(f"[data] {len(recs)} labeled 16S | {n_pos} pathogen / {n_neg} non-pathogen "
          f"| {len(genera)} genera")

    index = {km: i for i, km in enumerate(build_canonical_kmers(args.kmer_k))}
    print(f"[features] k={args.kmer_k} -> {len(index)} canonical k-mers | "
          f"augment={args.augment}")

    from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
    from sklearn.metrics import brier_score_loss

    results = {}

    # ---- stratified k-fold ----
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    pooled, folds, yt, yp = run_cv(recs, args.kmer_k, index, skf, args, "stratified", rng)
    pooled["brier"] = float(brier_score_loss(yt, yp))
    results["stratified_kfold"] = dict(pooled=pooled, folds=folds, y_true=yt.tolist(),
                                       y_prob=yp.tolist(), threshold=args.threshold)

    # ---- leave-one-genus-out (generalization / leakage-controlled) ----
    # collapse tiny genera into an 'other' group only if requested
    if args.min_genus_size > 1:
        keep = {g for g, c in genera.items() if c >= args.min_genus_size}
        for r in recs:
            if r["genus"] not in keep:
                r["genus"] = "other"
    logo = LeaveOneGroupOut()
    pooled2, folds2, yt2, yp2 = run_cv(recs, args.kmer_k, index, logo, args, "leave_genus_out", rng)
    pooled2["brier"] = float(brier_score_loss(yt2, yp2)) if len(set(yt2)) > 1 else float("nan")
    results["leave_genus_out"] = dict(pooled=pooled2, folds=folds2, y_true=yt2.tolist(),
                                      y_prob=yp2.tolist(), threshold=args.threshold)

    # ---- write outputs ----
    import csv
    def write_folds(path, rows):
        if not rows: return
        cols = sorted({k for r in rows for k in r})
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
            for r in rows: w.writerow(r)
    write_folds(os.path.join(args.outdir, "cv_stratified_folds.csv"), results["stratified_kfold"]["folds"])
    write_folds(os.path.join(args.outdir, "cv_leave_genus_out.csv"), results["leave_genus_out"]["folds"])

    summary = dict(
        n_records=len(recs), n_pathogen=int(n_pos), n_non_pathogen=int(n_neg),
        n_genera=len(genera), kmer_k=args.kmer_k, n_features=len(index),
        augment=args.augment, threshold=args.threshold,
        stratified_kfold=results["stratified_kfold"]["pooled"],
        leave_genus_out=results["leave_genus_out"]["pooled"],
    )
    with open(os.path.join(args.outdir, "metrics_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    try:
        make_figs(args.outdir, results)
    except Exception as e:
        print("[warn] figure generation skipped:", e)

    # ---- console report ----
    def line(name, m):
        print(f"\n== {name} ==")
        for key in ["accuracy", "balanced_accuracy", "sensitivity_recall",
                    "specificity", "precision", "f1", "mcc", "auroc", "auprc", "brier"]:
            if key in m and m[key] == m[key]:
                print(f"   {key:20s} {m[key]:.3f}")
        print(f"   confusion (tn,fp,fn,tp) = ({m['tn']},{m['fp']},{m['fn']},{m['tp']})")
    line("Stratified k-fold (in-distribution)", results["stratified_kfold"]["pooled"])
    line("Leave-one-genus-out (generalization)", results["leave_genus_out"]["pooled"])
    print(f"\n[done] wrote metrics + figures to {args.outdir}/")
    print("      -> report leave_genus_out numbers as the headline accuracy in the paper.")

if __name__ == "__main__":
    main()

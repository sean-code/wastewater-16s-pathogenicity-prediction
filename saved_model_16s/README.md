# saved_model_16s/

Model artifacts for the 16S pipeline. Populated by the setup scripts; the small
files are committed (needed for inference), the large ones are git-ignored and
rebuilt locally.

| File | Produced by | Commit? |
|------|-------------|---------|
| `kmer16s_model.pkl` | `train_16s_model.py` | optional (size-dependent) |
| `kmer_index.json` | `train_16s_model.py` | yes (small, needed for inference) |
| `pathogen_16s_catalog.json` | curated by hand / carried from GOMICS | yes |
| `labeled_16s.fasta` | `labeled_16s_build.py` | git-ignored (regenerate) |
| `ref_16s/` | `ref_16s_download.py` / `build_16s_blast_db.py` | git-ignored (large BLAST DB) |

Rebuild everything with:

```bash
python ref_16s_download.py
python labeled_16s_build.py --pathogen-genomes <dir> --nonpathogen-genomes <dir>
python train_16s_model.py --kmer-k 6 --augment 3
```

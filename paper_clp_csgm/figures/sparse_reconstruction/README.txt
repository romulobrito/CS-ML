Sparse-measurement illustration PNGs for Appendix (app:sparse_recon_examples).

Regenerate from repo root:
  python scripts/paper_sparse_reconstruction_figures.py --mode f03 --skip-lfista
  python scripts/paper_sparse_reconstruction_figures.py --mode crosswell --skip-lfista --rho 0.1 --cross-step 16

Optional flags: --seed, --rho, --example-index, --out-dir, --f03-step, --cw-val-frac.

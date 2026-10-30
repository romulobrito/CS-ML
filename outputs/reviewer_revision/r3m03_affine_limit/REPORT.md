# Affine limit of the frozen smoother

The selected-alpha audit was not overwritten.
Ceiling selected-alpha RMSE matches the recomputed alpha=1e6 curve
with max absolute difference 9.714e-17.
The affine fit uses only the sparse noisy observations.
It has no alpha. It is the least-squares line on the window index.

Groups were fixed before this comparison: all rows, ceiling rows, and interior rows.

all_108: n=108, median profile RMSE 5.197e-05, max profile RMSE 3.670e-04, max abs coordinate gap 2.354e-03, median |RMSE gap| 4.067e-06, max |RMSE gap| 1.960e-05.
  affine lower than alpha=1e6 in 29; alpha=1e6 lower in 79; CLP lower than affine in 61; affine lower than CLP in 47.
ceiling_62_expected: n=62, median profile RMSE 7.632e-05, max profile RMSE 3.670e-04, max abs coordinate gap 2.354e-03, median |RMSE gap| 3.938e-06, max |RMSE gap| 1.868e-05.
  affine lower than alpha=1e6 in 14; alpha=1e6 lower in 48; CLP lower than affine in 38; affine lower than CLP in 24.
interior: n=46, median profile RMSE 3.630e-05, max profile RMSE 1.571e-04, max abs coordinate gap 1.027e-03, median |RMSE gap| 4.199e-06, max |RMSE gap| 1.960e-05.
  affine lower than alpha=1e6 in 15; alpha=1e6 lower in 31; CLP lower than affine in 23; affine lower than CLP in 23.

Cell-mean ordering of the additional affine comparator against CLP:
noise 0.01: CLP lower cell mean in 0/9; affine lower cell mean in 9/9.
noise 0.05: CLP lower cell mean in 5/9; affine lower cell mean in 4/9.
noise 0.10: CLP lower cell mean in 8/9; affine lower cell mean in 1/9.
noise 0.20: CLP lower cell mean in 9/9; affine lower cell mean in 0/9.

A small profile gap means alpha=1e6 is already near the affine limit.
A large profile gap means the frozen ceiling is not that limit.
Neither result changes the selected-alpha smoother table.

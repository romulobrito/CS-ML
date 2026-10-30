# R3-03 smoothness audit

Protocol was written before test scoring. The alpha grid was not changed.
CLP-CSGM was not retrained. Interpolation matched the saved sweep
with max absolute difference 9.714e-17.
Noise levels are simulated and were not calibrated to laboratory uncertainty.
Wells 861 and 1254 are not in this comparison.

Cells: 36. Seed rows: 108.
Mean RMSE favors CLP over smoothing in 23 cells, smoothing over CLP in 13, ties 0.
Seed comparisons: CLP lower in 62, smoothing lower in 46.
Selected alpha at the grid minimum in 0 rows and at the grid maximum in 62 rows.
A selection at either end means the frozen grid did not contain a lower validation score inside the searched range.
The grid was not extended.

Positive delta_smooth_minus_clp means CLP has the lower cell-mean RMSE.
This check does not establish operational preference.

By simulated noise standard deviation, cell-mean counts (CLP lower, smoothing lower):
noise 0.01: CLP lower than smoothing in 1/9 cells; smoothing lower in 8/9; CLP lower than interpolation in 0/9.
noise 0.05: CLP lower than smoothing in 5/9 cells; smoothing lower in 4/9; CLP lower than interpolation in 7/9.
noise 0.10: CLP lower than smoothing in 8/9 cells; smoothing lower in 1/9; CLP lower than interpolation in 9/9.
noise 0.20: CLP lower than smoothing in 9/9 cells; smoothing lower in 0/9; CLP lower than interpolation in 9/9.

The reference noise 0.01 still favors interpolation over CLP in all nine cells.
Smoothing also has the lower cell mean than CLP in eight of those nine cells.
At noise 0.10, smoothing has the lower cell mean only at step 8 and rho 0.20.
At noise 0.20, CLP has the lower cell mean than this smoother in all nine cells.
Those larger noises remain a sensitivity check, not a field calibration.
Selections at alpha 1e6 are kept. The grid was not extended to chase them.

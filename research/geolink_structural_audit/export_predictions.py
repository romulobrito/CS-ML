"""Save one original GeoLink test pass without changing predictions."""

from pathlib import Path

import numpy as np


METHODS = (
    "pred_clp",
    "pred_csgm",
    "pred_interp",
    "pred_prior",
    "pred_ridge",
)


def save_prediction_bundle(
    output_file,
    *,
    seed,
    mask,
    well_id,
    window_id,
    y_true,
    observed,
    predictions,
    lambda_clp,
    y_measured=None,
    extras=None,
):
    """Export one original test pass without changing predictions.

    extras may hold numeric metadata arrays already aligned to the 72 windows.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    if y_true.shape != (72, 64):
        raise ValueError("Unexpected targets: {0}".format(y_true.shape))
    if observed.shape != y_true.shape:
        raise ValueError("Observation mask shape mismatch")
    if not np.all(observed.sum(axis=1) == 13):
        raise ValueError("Each window must have 13 observations")

    well_id = np.asarray(well_id).astype(str)
    window_id = np.asarray(window_id).astype(str)
    if well_id.shape != (72,) or len(set(well_id.tolist())) != 12:
        raise ValueError("Expected 72 windows from 12 real wells")
    if window_id.shape != (72,):
        raise ValueError("Window IDs must align with test arrays")
    if len(set(window_id.tolist())) != 72:
        raise ValueError("Window IDs must be unique")

    if mask == "contiguous_gap":
        if observed[:, 16:48].any():
            raise ValueError("Central gap contains observations")
        if not np.all(observed[:, :16].sum(axis=1) == 6):
            raise ValueError("Expected 6 left observations")
        if not np.all(observed[:, 48:].sum(axis=1) == 7):
            raise ValueError("Expected 7 right observations")
    elif mask != "random":
        raise ValueError("Unknown mask: {0}".format(mask))

    payload = {
        "seed": np.array(seed, dtype=np.int64),
        "mask": np.array(mask),
        "well_id": well_id,
        "window_id": window_id,
        "y_true": y_true,
        "observed": observed,
        "lambda_clp": np.asarray(lambda_clp, dtype=np.float64),
    }
    for name in METHODS:
        if name not in predictions:
            raise KeyError("Missing prediction: {0}".format(name))
        arr = np.asarray(predictions[name], dtype=np.float64)
        if arr.shape != y_true.shape or not np.isfinite(arr).all():
            raise ValueError("Invalid {0}: {1}".format(name, arr.shape))
        payload[name] = arr

    if y_measured is not None:
        measured = np.asarray(y_measured, dtype=np.float64)
        if measured.shape != y_true.shape:
            raise ValueError("Measured values shape mismatch")
        if not np.isfinite(measured[observed]).all():
            raise ValueError("Invalid observed measurement values")
        payload["y_measured"] = measured

    if extras:
        for key, value in extras.items():
            if key in payload:
                raise ValueError("Extra key collides with payload: {0}".format(key))
            payload[key] = np.asarray(value)

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    print("Saved {0}".format(path), flush=True)

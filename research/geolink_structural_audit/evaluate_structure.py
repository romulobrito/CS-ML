"""Regional and structural scores for the exported GeoLink pilot predictions."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = {
    "clp": "pred_clp",
    "csgm": "pred_csgm",
    "interp": "pred_interp",
    "prior": "pred_prior",
    "ridge": "pred_ridge",
}

EXPECTED = {
    ("random", "clp"): 0.4957,
    ("random", "csgm"): 0.4950,
    ("random", "interp"): 0.3316,
    ("random", "prior"): 1.2131,
    ("random", "ridge"): 0.9838,
    ("contiguous_gap", "clp"): 0.6592,
    ("contiguous_gap", "csgm"): 0.6264,
    ("contiguous_gap", "interp"): 0.4242,
    ("contiguous_gap", "prior"): 1.2126,
    ("contiguous_gap", "ridge"): 0.9862,
}

CENTER = np.zeros(64, dtype=bool)
CENTER[16:48] = True
BOUNDARY_EDGE = np.zeros(63, dtype=bool)
BOUNDARY_EDGE[15] = True
BOUNDARY_EDGE[47] = True

REGIONS = {
    "all": np.ones(64, dtype=bool),
    "central": CENTER,
    "external": ~CENTER,
}

SUM_COLS = [
    "n_points",
    "sse",
    "sae",
    "sum_error",
    "n_edges",
    "delta_sse",
    "sum_dx",
    "sum_dy",
    "sum_dx2",
    "sum_dy2",
    "sum_dxdy",
]

DISPLAY = {
    "clp": "CLP",
    "csgm": "CSGM",
    "interp": "Interpolation",
    "prior": "Prior-only",
    "ridge": "Ridge aux",
}


def score(y, yp, observed, region):
    """Score only unobserved samples and valid adjacent pairs.

    An edge is used only when both endpoints are unobserved, both lie in the
    region, and the edge does not cross the central/external boundary.
    """
    valid = (~observed) & region
    error = yp[valid] - y[valid]
    valid_edges = valid[:-1] & valid[1:] & ~BOUNDARY_EDGE
    dx = np.diff(y)[valid_edges]
    dy = np.diff(yp)[valid_edges]
    return {
        "n_points": int(error.size),
        "sse": float(np.sum(error ** 2)),
        "sae": float(np.sum(np.abs(error))),
        "sum_error": float(np.sum(error)),
        "n_edges": int(dx.size),
        "delta_sse": float(np.sum((dy - dx) ** 2)),
        "sum_dx": float(np.sum(dx)),
        "sum_dy": float(np.sum(dy)),
        "sum_dx2": float(np.sum(dx ** 2)),
        "sum_dy2": float(np.sum(dy ** 2)),
        "sum_dxdy": float(np.sum(dx * dy)),
    }


def derived_metrics(df):
    """Recover pooled metrics from additive sufficient statistics."""
    table = df.copy()
    n = table["n_points"].replace(0, np.nan)
    n_edges = table["n_edges"].replace(0, np.nan)
    table["rmse"] = np.sqrt(table["sse"] / n)
    table["mae"] = table["sae"] / n
    table["bias"] = table["sum_error"] / n
    table["delta_rmse"] = np.sqrt(table["delta_sse"] / n_edges)
    vx = table["sum_dx2"] - table["sum_dx"] ** 2 / n_edges
    vy = table["sum_dy2"] - table["sum_dy"] ** 2 / n_edges
    cov = table["sum_dxdy"] - table["sum_dx"] * table["sum_dy"] / n_edges
    vx = vx.clip(lower=0)
    vy = vy.clip(lower=0)
    denominator = np.sqrt(vx * vy).replace(0, np.nan)
    table["delta_corr"] = (cov / denominator).clip(-1, 1)
    table["amplitude_ratio"] = np.sqrt(vy / vx.replace(0, np.nan))
    return table


def aggregate(df, keys):
    """Sum sufficient statistics, then convert them back to metrics."""
    table = df.groupby(keys, as_index=False)[SUM_COLS].sum()
    return derived_metrics(table)


def read_bundle(path):
    """Load one seed/mask bundle and reject a malformed export."""
    with np.load(path, allow_pickle=False) as handle:
        data = {key: handle[key] for key in handle.files}
    required = {
        "seed",
        "mask",
        "well_id",
        "window_id",
        "y_true",
        "observed",
        *METHODS.values(),
    }
    missing = required - set(data)
    if missing:
        raise ValueError("{0}: missing {1}".format(path, sorted(missing)))
    seed = int(data["seed"].item())
    mask = str(data["mask"].item())
    y = np.asarray(data["y_true"], dtype=float)
    observed = np.asarray(data["observed"], dtype=bool)
    wells = np.asarray(data["well_id"]).astype(str)
    windows = np.asarray(data["window_id"]).astype(str)
    if seed not in (7, 23, 41):
        raise ValueError("Unexpected seed {0}".format(seed))
    if mask not in ("random", "contiguous_gap"):
        raise ValueError("Unexpected mask {0}".format(mask))
    if y.shape != (72, 64) or observed.shape != (72, 64):
        raise ValueError("Wrong array shape: {0}".format(path))
    if not np.isfinite(y).all():
        raise ValueError("Non-finite target")
    if not np.all(observed.sum(axis=1) == 13):
        raise ValueError("Expected 13 observations per window")
    if len(wells) != 72 or len(windows) != 72:
        raise ValueError("Missing well/window identities")
    if len(set(wells.tolist())) != 12:
        raise ValueError("Expected 12 test wells")
    if mask == "contiguous_gap":
        if observed[:, 16:48].any():
            raise ValueError("Central region contains observations")
        if not np.all(observed[:, :16].sum(axis=1) == 6):
            raise ValueError("Wrong left sampling count")
        if not np.all(observed[:, 48:].sum(axis=1) == 7):
            raise ValueError("Wrong right sampling count")
    for _method, key in METHODS.items():
        arr = np.asarray(data[key], dtype=float)
        if arr.shape != y.shape or not np.isfinite(arr).all():
            raise ValueError("Invalid prediction {0}".format(key))
    return seed, mask, data


def check_reproduction(seed_summary, output_dir):
    """Mean of 3 per-seed pooled RMSE values, as in the pilot."""
    table = seed_summary[seed_summary["region"] == "all"].groupby(["mask", "method"])["rmse"].mean()
    records = []
    for (mask, method), reference in EXPECTED.items():
        current = float(table.loc[(mask, method)])
        error = abs(current - reference)
        records.append(
            {
                "mask": mask,
                "method": method,
                "reference_rmse": reference,
                "reproduced_rmse": current,
                "absolute_difference": error,
                "passes": error <= 0.002,
            }
        )
    result = pd.DataFrame(records)
    result.to_csv(output_dir / "reproduction_check.csv", index=False)
    print("\nOriginal RMSE reproduction:")
    print(result.to_string(index=False))
    if not bool(result["passes"].all()):
        raise RuntimeError(
            "Original RMSEs did not reproduce. Investigate before interpreting structural metrics."
        )
    return result


def paired_comparisons(per_well, metrics):
    """Matched well/seed contrasts. For error metrics, negative favors CLP."""
    keys = ["seed", "mask", "region", "well_id"]
    records = []
    for metric in metrics:
        for competitor in ("interp", "csgm"):
            left = per_well.loc[per_well["method"] == "clp", keys + [metric]].rename(
                columns={metric: "clp_value"}
            )
            right = per_well.loc[per_well["method"] == competitor, keys + [metric]].rename(
                columns={metric: "other_value"}
            )
            joined = left.merge(right, on=keys, validate="one_to_one")
            joined["comparison"] = "clp_minus_{0}".format(competitor)
            joined["metric"] = metric
            joined["difference"] = joined["clp_value"] - joined["other_value"]
            if metric == "delta_corr":
                joined["clp_wins"] = joined["difference"] > 0
            else:
                joined["clp_wins"] = joined["difference"] < 0
            records.append(joined)
    return pd.concat(records, ignore_index=True)


def bootstrap_wells(paired, n_boot=5000):
    """Resample wells after averaging repeated seeds per well."""
    well_mean = paired.groupby(
        ["mask", "region", "comparison", "metric", "well_id"],
        as_index=False,
    )["difference"].mean()
    rng = np.random.default_rng(2026)
    records = []
    group_keys = ["mask", "region", "comparison", "metric"]
    for name, group in well_mean.groupby(group_keys):
        values = group["difference"].dropna().to_numpy()
        n = len(values)
        if n == 0:
            continue
        samples = rng.choice(values, size=(n_boot, n), replace=True)
        means = samples.mean(axis=1)
        if isinstance(name, tuple):
            labels = dict(zip(group_keys, name))
        else:
            labels = {group_keys[0]: name}
        win_rule = values > 0 if labels["metric"] == "delta_corr" else values < 0
        records.append(
            {
                **labels,
                "n_wells": n,
                "mean_clp_minus_competitor": float(values.mean()),
                "median_well_difference": float(np.median(values)),
                "wins_by_well": int(np.sum(win_rule)),
                "ci_025": float(np.quantile(means, 0.025)),
                "ci_975": float(np.quantile(means, 0.975)),
            }
        )
    return pd.DataFrame(records), well_mean


def seed_consistency(paired):
    """A consistent well win requires the same sign in all three seeds."""
    keys = ["mask", "region", "comparison", "metric", "well_id"]
    grouped = paired.groupby(keys, as_index=False).agg(
        n_seeds=("clp_wins", "size"),
        n_wins=("clp_wins", "sum"),
        mean_difference=("difference", "mean"),
    )
    grouped["consistent_clp_win"] = (grouped["n_seeds"] == 3) & (grouped["n_wins"] == 3)
    grouped["consistent_clp_loss"] = (grouped["n_seeds"] == 3) & (grouped["n_wins"] == 0)
    return grouped


def plot_example(seed, mask, data, outdir):
    """Fixed first exported window. Selection does not depend on the scores."""
    y = np.asarray(data["y_true"][0], dtype=float)
    obs = np.asarray(data["observed"][0], dtype=bool)
    x = np.arange(64)
    curves = {
        "CLP": np.asarray(data["pred_clp"][0], dtype=float),
        "CSGM": np.asarray(data["pred_csgm"][0], dtype=float),
        "Interpolation": np.asarray(data["pred_interp"][0], dtype=float),
    }
    if "y_measured" in data:
        measured = np.asarray(data["y_measured"][0], dtype=float)[obs]
        obs_label = "Noisy measurements"
    else:
        measured = y[obs]
        obs_label = "Observed target locations (truth)"
    well = str(np.asarray(data["well_id"]).astype(str)[0])
    window = str(np.asarray(data["window_id"]).astype(str)[0])
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(x, y, label="Truth", linewidth=2)
    for label, values in curves.items():
        axes[0].plot(x, values, label=label)
    axes[0].scatter(x[obs], measured, c="black", s=28, label=obs_label, zorder=5)
    axes[0].set_title(
        "GeoLink fold 0 | seed={0} | mask={1} | well={2} | window={3} | export row 0".format(
            seed, mask, well, window
        )
    )
    axes[0].set_ylabel("Normalized target")
    axes[0].legend(ncol=3)
    for label, values in curves.items():
        error = np.abs(values - y)
        axes[1].plot(x, np.where(~obs, error, np.nan), label=label)
    axes[1].set_ylabel("Absolute error (unobserved)")
    axes[1].legend()
    valid_edges = (~obs[:-1]) & (~obs[1:]) & ~BOUNDARY_EDGE
    if mask == "contiguous_gap":
        valid_edges &= CENTER[:-1] & CENTER[1:]
    for label, values in {"Truth": y, **curves}.items():
        delta = np.diff(values).astype(float)
        delta[~valid_edges] = np.nan
        axes[2].plot(x[:-1], delta, label=label)
    axes[2].set_ylabel("Consecutive difference")
    axes[2].set_xlabel("Decimated sample position")
    axes[2].legend(ncol=4)
    for ax in axes:
        ax.axvspan(16, 47, alpha=0.10)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "first_window_seed{0}_{1}.png".format(seed, mask), dpi=180)
    plt.close(fig)


def plot_well_panel(data, outdir):
    """Seed 7, first exported window of every test well. Not chosen by error."""
    if int(data["seed"].item()) != 7:
        return
    mask = str(data["mask"].item())
    wells = np.asarray(data["well_id"]).astype(str)
    windows = np.asarray(data["window_id"]).astype(str)
    y = np.asarray(data["y_true"], dtype=float)
    obs = np.asarray(data["observed"], dtype=bool)
    clp = np.asarray(data["pred_clp"], dtype=float)
    csgm = np.asarray(data["pred_csgm"], dtype=float)
    intr = np.asarray(data["pred_interp"], dtype=float)
    first = []
    seen = set()
    for i, well in enumerate(wells.tolist()):
        if well in seen:
            continue
        seen.add(well)
        first.append(i)
    fig, axes = plt.subplots(4, 3, figsize=(14, 12), sharex=True)
    x = np.arange(64)
    for ax, i in zip(axes.ravel(), first):
        ax.plot(x, y[i], color="black", linewidth=1.2, label="Truth")
        ax.plot(x, clp[i], label="CLP")
        ax.plot(x, csgm[i], label="CSGM")
        ax.plot(x, intr[i], label="Interpolation")
        ax.scatter(x[obs[i]], y[i, obs[i]], c="black", s=12, zorder=5)
        ax.axvspan(16, 47, alpha=0.10)
        ax.set_title("well={0} | window={1}".format(wells[i], windows[i]), fontsize=8)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(
        "GeoLink fold 0 | seed=7 | mask={0} | first exported window of each test well".format(mask)
    )
    fig.tight_layout()
    fig.savefig(outdir / "wells_seed7_{0}.png".format(mask), dpi=160)
    plt.close(fig)


def plot_summaries(seed_summary, well_mean, positions, outdir):
    """Aggregate figures. Points are wells or seeds, not selected examples."""
    methods = ["clp", "csgm", "interp", "prior", "ridge"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    for col, mask in enumerate(("random", "contiguous_gap")):
        for row, region in enumerate(("central", "external")):
            ax = axes[row, col]
            sub = seed_summary[
                (seed_summary["mask"] == mask) & (seed_summary["region"] == region)
            ]
            means = [float(sub.loc[sub["method"] == method, "rmse"].mean()) for method in methods]
            ax.bar(np.arange(len(methods)), means)
            ax.set_xticks(np.arange(len(methods)))
            ax.set_xticklabels([DISPLAY[m] for m in methods], rotation=25, ha="right")
            ax.set_title("GeoLink | {0} | {1} RMSE".format(mask, region))
            ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "rmse_central_vs_external.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    panels = (
        (0, 0, "random", "rmse"),
        (0, 1, "contiguous_gap", "rmse"),
        (1, 0, "random", "delta_rmse"),
        (1, 1, "contiguous_gap", "delta_rmse"),
    )
    focus = well_mean[well_mean["region"] == "all"]
    for row, col, mask, metric in panels:
        ax = axes[row, col]
        for comparison, xpos in (("clp_minus_interp", 0), ("clp_minus_csgm", 1)):
            vals = focus[
                (focus["mask"] == mask)
                & (focus["metric"] == metric)
                & (focus["comparison"] == comparison)
            ]["difference"].to_numpy()
            ax.scatter(np.full(len(vals), xpos), vals, s=28)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["CLP-interp", "CLP-CSGM"])
        ax.set_title("GeoLink | {0} | {1} | region=all".format(mask, metric))
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "paired_by_well.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, mask in zip(axes, ("random", "contiguous_gap")):
        left = focus[
            (focus["mask"] == mask)
            & (focus["metric"] == "rmse")
            & (focus["comparison"] == "clp_minus_interp")
        ][["well_id", "difference"]].rename(columns={"difference": "rmse_diff"})
        right = focus[
            (focus["mask"] == mask)
            & (focus["metric"] == "delta_rmse")
            & (focus["comparison"] == "clp_minus_interp")
        ][["well_id", "difference"]].rename(columns={"difference": "delta_diff"})
        joined = left.merge(right, on="well_id", validate="one_to_one")
        ax.scatter(joined["rmse_diff"], joined["delta_diff"])
        ax.axhline(0, color="black", linewidth=1)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("CLP minus interpolation RMSE")
        ax.set_ylabel("CLP minus interpolation delta RMSE")
        ax.set_title("GeoLink | {0} | one point per test well".format(mask))
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "rmse_vs_delta_by_well.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, mask in zip(axes, ("random", "contiguous_gap")):
        for method in ("clp", "csgm", "interp"):
            sub = positions[(positions["mask"] == mask) & (positions["method"] == method)]
            curve = sub.groupby("position")["sum_sqerr"].sum() / sub.groupby("position")["n"].sum()
            ax.plot(np.sqrt(curve.to_numpy()), label=DISPLAY[method])
        ax.axvspan(16, 47, alpha=0.10)
        ax.set_title("GeoLink fold 0 | {0} | RMSE by position, seeds pooled".format(mask))
        ax.grid(alpha=0.25)
        ax.legend()
    axes[-1].set_xlabel("Decimated sample position")
    fig.tight_layout()
    fig.savefig(outdir / "position_rmse.png", dpi=160)
    plt.close(fig)


def main():
    """Read the six bundles, gate on reproduction, then write tables and figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    figdir = args.figures if args.figures is not None else args.output.parent / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.predictions.glob("*.npz"))
    if len(files) != 6:
        raise ValueError("Expected exactly 6 NPZ files, got {0}".format(len(files)))

    records = []
    position_records = []
    combinations = set()
    reference_ids = None
    bundles = []
    for file in files:
        seed, mask, data = read_bundle(file)
        if (seed, mask) in combinations:
            raise ValueError("Duplicated seed/mask combination")
        combinations.add((seed, mask))
        y = np.asarray(data["y_true"], dtype=float)
        obs = np.asarray(data["observed"], dtype=bool)
        wells = np.asarray(data["well_id"]).astype(str)
        windows = np.asarray(data["window_id"]).astype(str)
        identities = list(zip(wells.tolist(), windows.tolist()))
        if reference_ids is None:
            reference_ids = identities
        elif identities != reference_ids:
            raise ValueError("Test-window order differs between exported bundles")
        bundles.append((seed, mask, data))
        for method, key in METHODS.items():
            prediction = np.asarray(data[key], dtype=float)
            for i in range(72):
                for region_name, region in REGIONS.items():
                    row = score(y[i], prediction[i], obs[i], region)
                    records.append(
                        {
                            "seed": seed,
                            "mask": mask,
                            "method": method,
                            "region": region_name,
                            "well_id": wells[i],
                            "window_id": windows[i],
                            **row,
                        }
                    )
            sqerr = (prediction - y) ** 2
            abserr = np.abs(prediction - y)
            for pos in range(64):
                valid = ~obs[:, pos]
                position_records.append(
                    {
                        "seed": seed,
                        "mask": mask,
                        "method": method,
                        "position": pos,
                        "n": int(valid.sum()),
                        "sum_sqerr": float(sqerr[valid, pos].sum()) if valid.any() else 0.0,
                        "sum_abserr": float(abserr[valid, pos].sum()) if valid.any() else 0.0,
                    }
                )

    if combinations != {
        (seed, mask) for seed in (7, 23, 41) for mask in ("random", "contiguous_gap")
    }:
        raise ValueError("Incomplete experiment grid")

    raw = pd.DataFrame(records)
    per_window = derived_metrics(raw)
    seed_summary = aggregate(raw, ["seed", "mask", "method", "region"])
    holdout_points = seed_summary[seed_summary["region"] == "all"]["n_points"]
    if not bool((holdout_points == 72 * 51).all()):
        raise RuntimeError("Region all does not score 51 unobserved positions in every window")
    check_reproduction(seed_summary, args.output)

    for seed, mask, data in bundles:
        plot_example(seed, mask, data, figdir)
        plot_well_panel(data, figdir)

    per_well = aggregate(raw, ["seed", "mask", "method", "region", "well_id"])
    paired_error = paired_comparisons(per_well, ("rmse", "delta_rmse"))
    paired_corr = paired_comparisons(per_well, ("delta_corr",))
    ci_error, well_mean_error = bootstrap_wells(paired_error)
    ci_corr, well_mean_corr = bootstrap_wells(paired_corr)
    consistency = pd.concat(
        [seed_consistency(paired_error), seed_consistency(paired_corr)],
        ignore_index=True,
    )
    positions = pd.DataFrame(position_records)
    positions["rmse"] = np.sqrt(positions["sum_sqerr"] / positions["n"].replace(0, np.nan))
    positions["mae"] = positions["sum_abserr"] / positions["n"].replace(0, np.nan)

    per_window.to_csv(args.output / "per_window_metrics.csv", index=False)
    per_well.to_csv(args.output / "per_well_metrics.csv", index=False)
    seed_summary.to_csv(args.output / "seed_summary.csv", index=False)
    paired_error.to_csv(args.output / "paired_comparisons.csv", index=False)
    well_mean_error.to_csv(args.output / "paired_well_mean.csv", index=False)
    ci_error.to_csv(args.output / "well_bootstrap_ci.csv", index=False)
    paired_corr.to_csv(args.output / "paired_correlation.csv", index=False)
    well_mean_corr.to_csv(args.output / "paired_correlation_well_mean.csv", index=False)
    ci_corr.to_csv(args.output / "well_bootstrap_correlation.csv", index=False)
    consistency.to_csv(args.output / "well_seed_consistency.csv", index=False)
    positions.to_csv(args.output / "position_rmse.csv", index=False)
    plot_summaries(seed_summary, well_mean_error, positions, figdir)

    print("\nStructural summary by seed:")
    print(
        seed_summary[
            [
                "seed",
                "mask",
                "method",
                "region",
                "rmse",
                "mae",
                "delta_rmse",
                "delta_corr",
                "amplitude_ratio",
            ]
        ].to_string(index=False)
    )
    print("\nPaired comparisons, clustered by well:")
    print(ci_error.to_string(index=False))
    print("\nResults written to {0}".format(args.output))


if __name__ == "__main__":
    main()

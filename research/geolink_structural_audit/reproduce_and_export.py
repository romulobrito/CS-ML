"""Reproduce the frozen GeoLink gap pilot and export every test prediction.

The training, masks, noise, lambda search, and latent optimization follow
research/geolink_structural_audit/frozen/run_geolink_gap_pilot.py.
This file does not change those calculations. It only keeps the source-row
identity of each test window and writes the reconstructions after they exist.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn

from export_predictions import save_prediction_bundle


os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[2]
FROZEN = Path(__file__).resolve().parent / "frozen" / "run_geolink_gap_pilot.py"
FROZEN_SHA256 = "695d015ebacf2effb1a59513753979133b32426501a9836c5b943f232637ce57"
DATA = (
    REPO
    / "data_local"
    / "geolink"
    / "public-processed-datasets"
    / "pgeoprj"
    / "godeep"
    / "dados"
    / "l2_datasets"
    / "imputation-processed-datasets"
)
REFERENCE = REPO / "data_local" / "pilot" / "geo_gap_pilot"
OUT = REPO / "outputs" / "exploratory" / "geolink_structural_audit"
PRED = OUT / "predictions"
REPRO = OUT / "reproduction"

torch.set_num_threads(1)
SEEDS = [7, 23, 41]
NTRAIN = 56
L = 64
M = 13
GAP = np.arange(16, 48)
LAMBDAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
EPOCHS = 180
NITERS = 220
RESTARTS = 2
SCORE_ATOL = 0.002


class AE(nn.Module):
    """Frozen decoder: 64 -> 64 -> 16 -> 64 -> 64 with ReLU."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(L, 64), nn.ReLU(), nn.Linear(64, 16))
        self.dec = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, L))

    def forward(self, y):
        return self.dec(self.enc(y))


def fit_ae(y_n, seed):
    """Train the autoencoder on normalized training windows only."""
    torch.manual_seed(seed)
    net = AE()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    yt = torch.tensor(y_n, dtype=torch.float32)
    for _ep in range(EPOCHS):
        for ix in torch.randperm(len(yt)).split(32):
            loss = ((net(yt[ix]) - yt[ix]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


def recover(ae, ymean, yscale, z0, obs, b, lam, seed):
    """Batch latent fit with shared restarts and the best batch objective."""
    obs = np.asarray(obs)
    zt = torch.tensor(z0, dtype=torch.float32)
    bt = torch.tensor((b - ymean[obs][None, :]) / yscale[obs][None, :], dtype=torch.float32)
    zscale = torch.tensor(yscale[obs], dtype=torch.float32)
    noise = torch.Generator().manual_seed(seed + 190)
    best = np.inf
    ypred = None
    for restart in range(RESTARTS):
        if restart == 0:
            zi = zt.clone()
        else:
            zi = zt + 0.05 * torch.randn(zt.shape, generator=noise)
        zi = zi.detach().requires_grad_()
        opt = torch.optim.Adam([zi], lr=0.05)
        for _k in range(NITERS):
            yp = ae.dec(zi)
            data = ((yp[:, obs] - bt) * zscale).square().sum() / len(zi)
            prior = lam * (zi - zt).square().sum() / len(zi)
            loss = data + prior
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            yp = ae.dec(zi)
            loss = (
                ((yp[:, obs] - bt) * zscale).square().sum() + lam * (zi - zt).square().sum()
            ) / len(zi)
            if float(loss.item()) < best:
                best = float(loss.item())
                ypred = yp.numpy().copy()
    if ypred is None:
        raise RuntimeError("Latent recovery produced no candidate")
    return ypred * yscale[None, :] + ymean[None, :]


def rms(y, yhat, mask):
    """Pooled RMSE on the selected positions."""
    return float(np.sqrt(np.mean((y[:, mask] - yhat[:, mask]) ** 2)))


def interp(y, obs, b):
    """Linear interpolation on decimated indices, with endpoint hold."""
    yp = np.empty_like(y)
    x = np.arange(L)
    for i in range(len(y)):
        yp[i] = np.interp(x, obs, b[i], left=b[i, 0], right=b[i, -1])
    return yp


def mask_for(seed, kind):
    """One shared 13-position mask. The gap mask leaves indices 16:48 empty."""
    rng = np.random.default_rng(3000 + seed)
    if kind == "random":
        return np.sort(rng.choice(L, M, replace=False))
    left = np.sort(rng.choice(np.arange(0, 16), 6, replace=False))
    right = np.sort(rng.choice(np.arange(48, 64), M - 6, replace=False))
    return np.r_[left, right]


def load_fold():
    """Rebuild the frozen 56/55/72 window split and keep source-row identity."""
    train_path = DATA / "geolink_fold_0_well_log_sliced_train.npy"
    val_path = DATA / "geolink_fold_0_well_log_sliced_val.npy"
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError("GeoLink fold 0 arrays are not at {0}".format(DATA))
    tr = np.load(train_path, mmap_mode="r")
    va = np.load(val_path, mmap_mode="r")
    with open(DATA / "geolink_fold_0_well_log_slices_meta_train.json", encoding="utf-8") as handle:
        trmeta = json.load(handle)
    with open(DATA / "geolink_fold_0_well_log_slices_meta_val.json", encoding="utf-8") as handle:
        vameta = json.load(handle)
    if len(tr) != len(trmeta) or len(va) != len(vameta):
        raise ValueError("Slice metadata does not align with the arrays")
    if {v[0] for v in trmeta} & {v[0] for v in vameta}:
        raise ValueError("Train and validation files share a well name")
    rng = np.random.default_rng(10203)
    wells = np.unique([m[0] for m in trmeta])
    rng.shuffle(wells)
    chosen = np.array(
        [
            rng.choice([i for i, row in enumerate(trmeta) if row[0] == well])
            for well in wells[:NTRAIN]
        ]
    )
    vwells = np.unique([m[0] for m in vameta])
    rng.shuffle(vwells)
    valwell = set(vwells[:10])
    testwell = set(vwells[10:22])

    def sample_ix(group, quota):
        out = []
        for well in sorted(group):
            ix = [i for i, row in enumerate(vameta) if row[0] == well]
            out += list(rng.choice(ix, min(len(ix), quota), replace=False))
        return np.asarray(out, dtype=int)

    vi = sample_ix(valwell, 6)
    ti = sample_ix(testwell, 6)

    def extract(array, idx):
        chunk = np.asarray(array[idx][:, ::4, :], dtype=np.float64)
        target = chunk[:, :, 0]
        aux = chunk[:, :, 1:3].reshape(len(idx), -1)
        return target, aux

    yt, xt = extract(tr, chosen)
    yv, xv = extract(va, vi)
    ye, xe = extract(va, ti)
    meta = dict(
        ntrain=len(chosen),
        nval=len(vi),
        ntest=len(ti),
        train_wells=sorted(wells[:NTRAIN].tolist()),
        val_wells=sorted(valwell),
        test_wells=sorted(testwell),
    )
    identity = {
        "source_index": np.asarray(ti, dtype=np.int64),
        "well_id": np.array([vameta[i][0] for i in ti], dtype=str),
        "meta_c1": np.array([int(vameta[i][1]) for i in ti], dtype=np.int64),
        "meta_c2": np.array([int(vameta[i][2]) for i in ti], dtype=np.int64),
    }
    return (yt, xt), (yv, xv), (ye, xe), meta, identity


def assert_frozen_script():
    """Refuse to run if the packaged original script was edited."""
    digest = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    if digest != FROZEN_SHA256:
        raise RuntimeError("Frozen pilot script hash changed: {0}".format(digest))


def assert_reference_split(meta):
    """The well lists are part of the frozen protocol."""
    reference = json.loads((REFERENCE / "config.json").read_text(encoding="utf-8"))
    split = reference["split_wells"]
    if meta["train_wells"] != split["train"]:
        raise RuntimeError("Training wells do not match the packaged config")
    if meta["val_wells"] != split["validation"]:
        raise RuntimeError("Validation wells do not match the packaged config")
    if meta["test_wells"] != split["test"]:
        raise RuntimeError("Test wells do not match the packaged config")
    if meta["ntrain"] != 56 or meta["nval"] != 55 or meta["ntest"] != 72:
        raise RuntimeError("Window counts are not 56/55/72")


def compare_with_reference(results, lambda_rows):
    """Per-seed holdout RMSE and selected lambda against the packaged pilot."""
    reference = pd.read_csv(REFERENCE / "results.csv")
    merged = results.merge(
        reference,
        on=["seed", "mask", "method"],
        suffixes=("", "_ref"),
        validate="one_to_one",
    )
    merged["abs_diff_holdout"] = (merged["rmse_holdout"] - merged["rmse_holdout_ref"]).abs()
    merged["abs_diff_central"] = (
        merged["rmse_central_heldout"] - merged["rmse_central_heldout_ref"]
    ).abs()
    ref_lambda = reference[reference["method"] == "CLP"][
        ["seed", "mask", "lambda_selected"]
    ].rename(columns={"lambda_selected": "lambda_ref"})
    got_lambda = pd.DataFrame(lambda_rows)
    # The training loop uses strict <, which keeps the earlier grid value on a tie.
    picked = []
    for (seed, mask), group in got_lambda.groupby(["seed", "mask"]):
        best = np.inf
        lam_star = None
        for row in group.itertuples(index=False):
            if row.validation_rmse < best:
                best = float(row.validation_rmse)
                lam_star = float(row.lambda_value)
        picked.append({"seed": int(seed), "mask": mask, "lambda_selected": lam_star})
    picked = pd.DataFrame(picked).merge(ref_lambda, on=["seed", "mask"], validate="one_to_one")
    picked["lambda_match"] = np.isclose(picked["lambda_selected"], picked["lambda_ref"])
    out = merged[
        [
            "seed",
            "mask",
            "method",
            "rmse_holdout",
            "rmse_holdout_ref",
            "abs_diff_holdout",
            "rmse_central_heldout",
            "rmse_central_heldout_ref",
            "abs_diff_central",
        ]
    ]
    out.to_csv(REPRO / "per_seed_reproduction.csv", index=False)
    picked.to_csv(REPRO / "lambda_reproduction.csv", index=False)
    neural = out["method"].isin(["CLP", "CSGM_sem_prior", "prior_only"])
    nonneural = ~neural
    if float(out.loc[nonneural, "abs_diff_holdout"].max()) > 1e-8:
        raise RuntimeError("Interpolation or direct ridge does not match the packaged pilot")
    if not bool(picked["lambda_match"].all()):
        raise RuntimeError("Selected lambda does not match the packaged pilot")
    later = out["seed"].isin([23, 41]) & neural
    if float(out.loc[later, "abs_diff_holdout"].max()) > 1e-4:
        raise RuntimeError("Seeds 23 and 41 do not match the packaged neural results")
    means = out.groupby(["mask", "method"])["rmse_holdout"].mean()
    published = {
        ("random", "CLP"): 0.4957,
        ("random", "CSGM_sem_prior"): 0.4950,
        ("random", "interpolacao"): 0.3316,
        ("random", "ridge_aux"): 0.9838,
        ("random", "prior_only"): 1.2131,
        ("contiguous_gap", "CLP"): 0.6592,
        ("contiguous_gap", "CSGM_sem_prior"): 0.6264,
        ("contiguous_gap", "interpolacao"): 0.4242,
        ("contiguous_gap", "ridge_aux"): 0.9862,
        ("contiguous_gap", "prior_only"): 1.2126,
    }
    mean_gaps = []
    for key, reference_value in published.items():
        gap = abs(float(means.loc[key]) - reference_value)
        mean_gaps.append(gap)
        if gap > SCORE_ATOL:
            raise RuntimeError("Mean RMSE exceeds the 0.002 tolerance for {0}".format(key))
    worst = float(out["abs_diff_holdout"].max())
    worst_central = float(out["abs_diff_central"].max())
    return worst, worst_central, float(max(mean_gaps))


def main():
    """Run the frozen protocol once and write predictions plus reproduction tables."""
    assert_frozen_script()
    if (REPRO / "results.csv").exists() or any(PRED.glob("*.npz")):
        raise RuntimeError("Refusing to overwrite an existing GeoLink structural run")
    REPRO.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    (yt, xt), (yv, xv), (ye, xe), meta, identity = load_fold()
    assert_reference_split(meta)
    if not np.isfinite(yt).all() or not np.isfinite(yv).all() or not np.isfinite(ye).all():
        raise ValueError("Non-finite target window")
    y_scaler = StandardScaler().fit(yt)
    xn_scaler = StandardScaler().fit(xt)
    ymean = y_scaler.mean_
    yscale = y_scaler.scale_
    if np.any(yscale == 0):
        raise ValueError("Target scale has a zero entry")
    xf = xn_scaler.transform(xt)
    vf = xn_scaler.transform(xv)
    ef = xn_scaler.transform(xe)
    aux = Ridge(alpha=10).fit(xf, yt)
    ridge_pred = aux.predict(ef)
    rows = []
    pred_rows = []
    lambda_rows = []
    start = time.time()
    order = pd.DataFrame(
        {
            "row": np.arange(len(identity["well_id"])),
            "well_id": identity["well_id"],
            "window_id": identity["source_index"].astype(str),
            "source_index": identity["source_index"],
            "meta_c1": identity["meta_c1"],
            "meta_c2": identity["meta_c2"],
        }
    )
    order.to_csv(REPRO / "test_window_index.csv", index=False)
    for seed in SEEDS:
        print("seed {0} start".format(seed), flush=True)
        ae = fit_ae(y_scaler.transform(yt), seed)
        with torch.no_grad():
            zt = ae.enc(torch.tensor(y_scaler.transform(yt), dtype=torch.float32)).numpy()
        zsc = StandardScaler().fit(zt)
        ridge_lat = Ridge(alpha=1.0).fit(xf, zsc.transform(zt))
        zv = zsc.inverse_transform(ridge_lat.predict(vf))
        ze = zsc.inverse_transform(ridge_lat.predict(ef))
        for kind in ("random", "contiguous_gap"):
            obs = mask_for(seed, "random" if kind == "random" else "gap")
            hold = np.array([i for i in range(L) if i not in set(obs.tolist())])
            if kind == "contiguous_gap" and set(obs.tolist()) & set(GAP.tolist()):
                raise RuntimeError("Gap mask observed the central interval")
            if len(obs) != M or len(hold) != L - M:
                raise RuntimeError("Mask does not have 13 observations")
            rnga = np.random.default_rng(5900 + seed)
            noise_val = rnga.normal(0, 0.01, size=yv.shape)
            noise_test = rnga.normal(0, 0.01, size=ye.shape)
            bval = yv[:, obs] + noise_val[:, obs]
            btest = ye[:, obs] + noise_test[:, obs]
            bestval = np.inf
            lam_star = None
            for lam in LAMBDAS:
                yh = recover(ae, ymean, yscale, zv, obs, bval, lam, seed)
                valrmse = rms(yv, yh, hold)
                lambda_rows.append(
                    dict(seed=seed, mask=kind, lambda_value=lam, validation_rmse=valrmse)
                )
                if valrmse < bestval:
                    bestval = valrmse
                    lam_star = lam
            if lam_star is None:
                raise RuntimeError("Lambda selection failed")
            clp = recover(ae, ymean, yscale, ze, obs, btest, lam_star, seed)
            free = recover(ae, ymean, yscale, np.zeros_like(ze), obs, btest, 0, seed)
            intr = interp(ye, obs, btest)
            prior = (
                ae.dec(torch.tensor(ze, dtype=torch.float32)).detach().numpy() * yscale[None, :]
                + ymean[None, :]
            )
            preds = {
                "CLP": clp,
                "CSGM_sem_prior": free,
                "interpolacao": intr,
                "ridge_aux": ridge_pred,
                "prior_only": prior,
            }
            for label, pr in preds.items():
                rows.append(
                    dict(
                        seed=seed,
                        mask=kind,
                        method=label,
                        rmse_holdout=rms(ye, pr, hold),
                        rmse_central_heldout=rms(ye, pr, np.intersect1d(hold, GAP)),
                        lambda_selected=lam_star if label == "CLP" else np.nan,
                        n_train=len(yt),
                        n_test=len(ye),
                        n_holdout=len(hold),
                    )
                )
            observed = np.zeros_like(ye, dtype=bool)
            observed[:, obs] = True
            measured = np.full(ye.shape, np.nan, dtype=np.float64)
            measured[:, obs] = btest
            save_prediction_bundle(
                PRED / "seed{0}_{1}.npz".format(seed, kind),
                seed=seed,
                mask=kind,
                well_id=identity["well_id"],
                window_id=identity["source_index"].astype(str),
                y_true=ye,
                observed=observed,
                predictions={
                    "pred_clp": clp,
                    "pred_csgm": free,
                    "pred_interp": intr,
                    "pred_prior": prior,
                    "pred_ridge": ridge_pred,
                },
                lambda_clp=lam_star,
                y_measured=measured,
                extras={
                    "source_index": identity["source_index"],
                    "meta_c1": identity["meta_c1"],
                    "meta_c2": identity["meta_c2"],
                },
            )
            idx = 0
            for j in range(L):
                record = dict(
                    seed=seed,
                    mask=kind,
                    index=j,
                    truth=ye[idx, j],
                    observed=int(j in set(obs.tolist())),
                )
                for label, pr in preds.items():
                    record[label] = pr[idx, j]
                pred_rows.append(record)
        print("seed {0} elapsed {1}".format(seed, round(time.time() - start, 1)), flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(REPRO / "results.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(REPRO / "first_heldout_window.csv", index=False)
    pd.DataFrame(lambda_rows).to_csv(REPRO / "lambda_validation.csv", index=False)
    summary = (
        results.groupby(["mask", "method"])[["rmse_holdout", "rmse_central_heldout"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(REPRO / "summary.csv", index=False)
    checks = dict(
        train_val_wells_disjoint=not bool(set(meta["train_wells"]) & set(meta["val_wells"])),
        train_test_wells_disjoint=not bool(set(meta["train_wells"]) & set(meta["test_wells"])),
        val_test_wells_disjoint=not bool(set(meta["val_wells"]) & set(meta["test_wells"])),
        n_train=len(yt),
        n_validation=len(yv),
        n_test=len(ye),
        all_finite=bool(np.isfinite(results.rmse_holdout).all()),
        gap_zero_observations=all(not set(mask_for(s, "gap").tolist()) & set(GAP.tolist()) for s in SEEDS),
        all_mask_counts=all(len(mask_for(s, k)) == M for s in SEEDS for k in ("random", "gap")),
    )
    if not all(v for v in checks.values() if isinstance(v, bool)):
        raise RuntimeError("Protocol checks failed: {0}".format(checks))
    config = dict(
        dataset="GeoLink fold 0 public-processed-datasets.tar(2).gz",
        dataset_origin="user-provided archive",
        dataset_resolved_path=str(DATA),
        target="channel index 0; physical meaning unavailable in metadata",
        aux_channels=[1, 2],
        unused_channel=3,
        train_budget=56,
        seeds=SEEDS,
        window_length=L,
        original_window_length=256,
        subsampling_stride=4,
        observations_per_window=M,
        contiguous_gap_positions=[16, 47],
        gap_observation_rule="6 positions from 0:16 plus 7 from 48:64",
        random_rule="13 positions uniform without replacement",
        ae_arch="64-64-16 encoder, 16-64-64 decoder, ReLU",
        ae_epochs=EPOCHS,
        ae_weight_decay=1e-5,
        latent_iters=NITERS,
        latent_restarts=RESTARTS,
        lambda_grid=LAMBDAS,
        lambda_selection="validation holdout RMSE",
        n_train=len(yt),
        n_validation=len(yv),
        n_test=len(ye),
        noise_std=0.01,
        direct_ridge_alpha=10,
        latent_ridge_alpha=1.0,
        comparison="independent GeoLink proxy, NOT original CS-ML F3 run",
        split_wells={
            "train": meta["train_wells"],
            "validation": meta["val_wells"],
            "test": meta["test_wells"],
        },
        test_windows_come_from="geolink_fold_0_well_log_sliced_val.npy",
        interpolation_axis="decimated index 0..63, endpoint hold",
    )
    (REPRO / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (REPRO / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    worst, worst_central, mean_gap = compare_with_reference(results, lambda_rows)
    manifest = dict(
        frozen_sha256=FROZEN_SHA256,
        head=os.popen("git rev-parse HEAD").read().strip(),
        branch=os.popen("git branch --show-current").read().strip(),
        elapsed_s=round(time.time() - start, 3),
        torch=torch.__version__,
        numpy=np.__version__,
        worst_abs_holdout_diff=worst,
        worst_abs_central_diff=worst_central,
        worst_mean_gap_vs_four_decimals=mean_gap,
        score_atol=SCORE_ATOL,
        continuation_audit_zip="absent",
        data_zip_used="public-processed-datasets.zip",
        data_zip_requested_name="public-processed-datasets(1).zip",
    )
    (REPRO / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        "DONE seconds {0} worst_holdout_diff {1}".format(round(time.time() - start, 1), worst),
        flush=True,
    )


if __name__ == "__main__":
    main()

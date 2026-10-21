#!/usr/bin/env python3
"""F3 uncertainty study: uncertainty-weighted CLP, extended lambda and robust Huber.
Audit extension of scripts/f3_gap_audit/run_crosswell_gap.py.
The original cross-well F03-4 test well stays unseen in model and parameter selection.
"""
from __future__ import annotations
import argparse,csv,json,math,os,time
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from csgm_m2_module import train_ae_generator,encode_y,RidgePrior,decode_latent_prior
from scripts.f3_gap_audit.run_crosswell_gap import (
    data,positions,Mulberry32,metrics,selection,gp_pred,savecsv,
    L,SEEDS,MODES,NOISES,RHO,LAM,
)
from types import SimpleNamespace

EXT_LAM=tuple(LAM)+(.3,1.,3.)
VARIANTS=("canonical","extended","weighted_prior","huber")
CASES=(("gaussian_low",.01),("gaussian_high",.05),("contaminated",.01))
HUBER_DELTA_SIGMA=1.5

def latent_precision(ztrain,zfit):
    """Train-only shrunken diagonal precision; no validation or test labels."""
    residual=np.asarray(ztrain,dtype=float)-np.asarray(zfit,dtype=float)
    var=np.mean(residual**2,axis=0)+1e-8
    pooled=float(np.mean(var))
    shrunk=.50*var+.50*pooled
    weight=pooled/shrunk
    weight=np.clip(weight,.25,4.)
    return weight/weight.mean()

def recover_grid(ae,ids,measurements,z0,y_mean,y_scale,
                 lams,variant,weights,sigma,seed,iters=400,restarts=3):
    """Batched lambdas, independent Adam trajectories, canonical sum-loss scaling.

    For canonical/extended, objective is the original residual squared norm
    plus lambda*||z-z0||^2. One different parameter dimension per lambda.
    For weighted_prior, only latent quadratic penalty changes.
    For huber, data penalty is 2*Huber_delta(r) with delta=1.5*sigma.
    In all cases the objective is summed across rows, divided by row count.
    """
    n=len(z0); d=len(lams); m=len(ids)
    device="cpu"
    z0=torch.as_tensor(z0,dtype=torch.float32,device=device)
    obs=torch.as_tensor(measurements,dtype=torch.float32,device=device)
    lamsT=torch.as_tensor(lams,dtype=torch.float32).view(d,1,1)
    weightsT=torch.as_tensor(weights,dtype=torch.float32).view(1,1,-1)
    ymu=torch.as_tensor(y_mean[ids],dtype=torch.float32).view(1,1,m)
    yscale=torch.as_tensor(y_scale[ids],dtype=torch.float32).view(1,1,m)
    fixed=(obs.unsqueeze(0)-ymu)
    generator=torch.Generator(device=device).manual_seed(int(seed))
    best_loss=np.full(d,np.inf); best_predictions=[None for _ in range(d)]
    ae.eval()
    for restart in range(restarts):
        initial=z0 if restart==0 else (z0+.05*torch.randn(z0.shape,generator=generator,device=device))
        z=initial.unsqueeze(0).expand(d,-1,-1).clone().detach().requires_grad_(True)
        optimizer=torch.optim.Adam([z],lr=.05)
        def loss_components():
            decoded=ae.decode(z)
            residual=decoded[:,:,ids]*yscale-fixed
            if variant=="huber":
                delta=HUBER_DELTA_SIGMA*sigma
                dr=residual.abs()
                dat=torch.where(dr<=delta,residual**2,2*delta*dr-delta**2)
            else:
                dat=residual**2
            diff=(z-z0.unsqueeze(0))**2
            if variant=="weighted_prior":diff=diff*weightsT
            objective=dat.sum(dim=(1,2))/n + lamsT.view(-1)*diff.sum(dim=(1,2))/n
            return objective
        for _ in range(iters):
            optimizer.zero_grad()
            objectives=loss_components()
            objectives.sum().backward()
            optimizer.step()
        with torch.no_grad():
            final_losses=loss_components().cpu().numpy()
            output=ae.decode(z).cpu().numpy()*y_scale[None,None,:]+y_mean[None,None,:]
        for q in range(d):
            if final_losses[q]<best_loss[q]:
                best_loss[q]=float(final_losses[q]);best_predictions[q]=output[q].copy()
    assert all(pred is not None for pred in best_predictions)
    return np.stack(best_predictions)

def get_observations(Yv,Yt,ids,seed,mode,sigma,kind):
    gen=Mulberry32(seed+322+int(round(RHO*100))*131+len(mode)*19)
    def inject(Y):
        rows=[]
        for yy in Y:
            b=[]
            for pos in ids:
                normal=sigma*gen.gaussian()
                if kind=="contaminated":
                    # 12% of observed coordinates receive an additional 12*sigma error
                    trigger=gen.random()
                    if trigger<.12:normal+=12*sigma*gen.gaussian()
                b.append(yy[pos]+normal)
            rows.append(b)
        return np.asarray(rows)
    return inject(Yv),inject(Yt)

def run(output,quick=False):
    start=time.time()
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    np.set_printoptions(precision=9)
    repo=Path(__file__).resolve().parents[2]
    Xtr,Xv,Xt,Ytr,Yv,Yt=data(repo)
    cfg=SimpleNamespace(csgm_latent_dim=16,csgm_hidden_dim=128,
                        csgm_batch_size=64,csgm_ae_lr=.001,csgm_weight_decay=1e-5,
                        csgm_ae_epochs=200 if not quick else 15,
                        csgm_iters=400 if not quick else 30,
                        csgm_opt_lr=.05,csgm_restarts=3 if not quick else 1,
                        csgm_ridge_alpha=1.)
    seeds=SEEDS if not quick else (7,)
    modes=MODES if not quick else ("spread",)
    cases=CASES if not quick else (("gaussian_low",.01),)
    allrows=[]; validation=[]; details=[]; consistency=[]
    for seed in seeds:
        scaler=StandardScaler().fit(Ytr)
        yn=scaler.transform(Ytr)
        ae=train_ae_generator(yn,cfg,seed,"cpu")
        for p in ae.parameters():p.requires_grad_(False)
        ztr=encode_y(ae,yn,"cpu")
        latentprior=RidgePrior(alpha=1.,random_state=seed+7000).fit(Xtr,ztr)
        z0train=latentprior.predict(Xtr)
        z0v=latentprior.predict(Xv);z0t=latentprior.predict(Xt)
        pval=decode_latent_prior(ae,z0v,scaler.mean_,scaler.scale_,"cpu")
        ptest=decode_latent_prior(ae,z0t,scaler.mean_,scaler.scale_,"cpu")
        weights=latent_precision(ztr,z0train)
        print("PRIOR",json.dumps({"seed":seed,"weights":weights.tolist(),
                                 "ae_epochs":cfg.csgm_ae_epochs}),flush=True)
        for mode in modes:
            ids=positions(max(4,round(RHO*L)),mode,seed+int(round(RHO*100))*19)
            hidden=np.setdiff1d(np.arange(L),ids)
            for kind,sigma in cases:
                bv,bt=get_observations(Yv,Yt,ids,seed,mode,sigma,kind)
                best_gp=selection(bv,ids,Yv,hidden,None)
                p_gp=gp_pred(bt,ids,best_gp[3],None)
                pred_interp=np.stack([np.interp(np.arange(L),ids,b) for b in bt])
                pred_mean=np.broadcast_to(bt.mean(axis=1,keepdims=True),Yt.shape)
                best_base=min([(n,metrics(P,Yt,hidden)["rmse"]) for n,P in
                               [("interpolation",pred_interp),("GP_smooth",p_gp),
                                ("observed_mean",pred_mean)]],key=lambda kv:kv[1])
                row={"seed":seed,"mode":mode,"noise_case":kind,"noise_sigma":sigma,
                     "m":len(ids),"observations":";".join(map(str,ids)),
                     "n_test_windows":len(Yt),"n_scored":Yt[:,hidden].size,
                     "GP_length":best_gp[1],"GP_regularizer":best_gp[2],
                     "best_simple_name":best_base[0],"best_simple_rmse":best_base[1]}
                baselines={"prior_only":ptest,"interpolation":pred_interp,
                           "GP_smooth":p_gp,"observed_mean":pred_mean}
                for name,P in baselines.items():
                    for k,v in metrics(P,Yt,hidden).items():row[f"{name}_{k}"]=v
                grid_preds={}
                for variant in VARIANTS:
                    lam_grid=LAM if variant=="canonical" else EXT_LAM
                    vpreds=recover_grid(ae,ids,bv,z0v,scaler.mean_,scaler.scale_,
                                       lam_grid,variant,weights,sigma,seed,
                                       cfg.csgm_iters,cfg.csgm_restarts)
                    errors=np.mean((vpreds[:,:,hidden]-Yv[None,:,hidden])**2,axis=(1,2))
                    k=int(np.argmin(errors))
                    validation.extend([{"seed":seed,"mode":mode,"noise_case":kind,
                        "variant":variant,"lambda":float(lam),
                        "val_hidden_rmse":math.sqrt(float(err))}
                        for lam,err in zip(lam_grid,errors)])
                    # Refit none. Only infer using validation-selected lambda.
                    bestlam=float(lam_grid[k])
                    testpred=recover_grid(ae,ids,bt,z0t,scaler.mean_,scaler.scale_,
                                          [bestlam],variant,weights,sigma,seed,
                                          cfg.csgm_iters,cfg.csgm_restarts)[0]
                    grid_preds[variant]=testpred
                    row[f"{variant}_lambda"]=bestlam
                    row[f"{variant}_val_rmse"]=math.sqrt(float(errors[k]))
                    for metric,value in metrics(testpred,Yt,hidden).items():
                        row[f"{variant}_{metric}"]=value
                # Guarantee expanded reproduces original candidate if selected lambda in old grid
                if row["canonical_lambda"]==row["extended_lambda"]:
                    delta=float(np.max(np.abs(grid_preds["canonical"]-grid_preds["extended"])))
                    if delta>5e-5:raise AssertionError("canonical/extended reproducibility "+str(delta))
                allrows.append(row)
                savecsv(output/"cell_metrics.csv",allrows)
                savecsv(output/"lambda_validation.csv",validation)
                for wi in range(len(Yt)):
                    for p in hidden:
                        rec={"seed":seed,"mode":mode,"noise_case":kind,
                             "window":wi,"coordinate":int(p),"truth":float(Yt[wi,p])}
                        for name,P in {**baselines,**grid_preds}.items():rec[f"pred_{name}"]=float(P[wi,p])
                        details.append(rec)
                savecsv(output/"predictions_hidden.csv",details)
                print("CELL",json.dumps(row,sort_keys=True),flush=True)
    manifest={
        "repo_sha":os.environ.get("GITHUB_SHA"),"quick":quick,"elapsed_seconds":time.time()-start,
        "data":"original 30dB F3 tables in this repository",
        "train_wells":["F02-1","F03-2","F06-1"],"heldout_test_well":"F03-4",
        "train_windows":41,"val_windows":8,"test_windows":49,"window_len":64,
        "embargo_windows":3,"rho":RHO,"seeds":list(seeds),"modes":list(modes),
        "cases":list(cases),"model":"original TinyAE+RidgePrior and Adam",
        "epochs":cfg.csgm_ae_epochs,"iters":cfg.csgm_iters,"restarts":cfg.csgm_restarts,
        "lambda_original":list(LAM),"lambda_extended":list(EXT_LAM),
        "latent_precision":"0.5 train residual variance + 0.5 pooled, clipped to [0.25,4], then normalized",
        "robust_loss":"2 x Huber with delta = 1.5 sigma",
        "contamination":"12% probability of additional Gaussian 12*sigma per observed coordinate",
        "hyperparameter_validation":"only heldout coordinates on embargoed 8 development windows",
        "test_score":"only hidden coordinates; test windows overlap; only one test well",
        "caution":"uncertainty scales are known by simulation; not instrument-calibrated",
        "aim":"test if improvements versus prior translate to improvements versus GP smoothing"
    }
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("DONE",len(allrows),"cells",round(time.time()-start,1),"seconds",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("outputs/f3_clp_uncertainty"))
    p.add_argument("--quick",action="store_true")
    a=p.parse_args();run(a.out,a.quick)

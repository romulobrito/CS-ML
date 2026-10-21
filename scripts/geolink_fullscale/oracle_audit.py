#!/usr/bin/env python3
"""Diagnostic only — Geolink fold 0 decoder oracle/prior/conditional/shift audit.

Uses exactly the original benchmark's full fold-0 arrays and the same group
train/validation split, TinyAE, and RidgePrior as the earlier full-scale run.
Unlike operational reconstruction, encoder and full-target latent oracle are
permitted to inspect held-out DTC, ONLY to diagnose generator representability.
No novel model selection, no changes to the original sparse CLP method.
"""
from __future__ import annotations
import argparse,csv,json,math,os,time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from csgm_m2_module import train_ae_generator,encode_y,decode_latent_prior,RidgePrior,csgm_recover_with_prior
from scripts.geolink_fullscale.run_fold0 import (
  full_arrays,split_dev_wells,aux_features,obs_positions,gp_adjust,choose_gp,
  hidden_stats,csv_write,SEEDS,OBS_COUNTS,L,ARCHIVE_SHA256
)

def numerics(v):
    a=np.asarray(v,dtype=float).ravel()
    return dict(n=int(a.size),mean=float(np.mean(a)),sd=float(np.std(a)),
      min=float(np.min(a)),p01=float(np.quantile(a,.01)),p50=float(np.median(a)),
      p95=float(np.quantile(a,.95)),p99=float(np.quantile(a,.99)),
      p999=float(np.quantile(a,.999)),max=float(np.max(a)),
      count_gt_2=int(np.sum(a>2)),count_gt_3=int(np.sum(a>3)),
      count_gt_5=int(np.sum(a>5)))
def met(yhat,y,hid=None):
    e=(np.asarray(yhat,dtype=float)-np.asarray(y,dtype=float))
    if hid is not None:e=e[:,hid]
    sq=e*e
    a=np.abs(e).ravel()
    s=np.sort(sq.ravel())[::-1]
    top=max(1,int(np.ceil(.01*len(s))))
    return dict(rmse=float(np.sqrt(np.mean(sq))),mae=float(np.mean(a)),
      abs_p50=float(np.quantile(a,.5)),abs_p90=float(np.quantile(a,.9)),
      abs_p99=float(np.quantile(a,.99)),abs_p999=float(np.quantile(a,.999)),
      abs_max=float(np.max(a)),top1pct_sse_frac=float(np.sum(s[:top])/np.sum(s)) if np.sum(s)>0 else 0.)

def latent_oracle(ae, y, scaler, z_encoded,z_prior, seed, steps=450,lr=.035):
    """Diagnostic use of full truth: minimize original target units with 3 starts.

    Independent Adam optimization of codes for each sample, with per-window
    best restart/initial candidate selection. Approximately solves projection
    into learned decoder range, not a certified global optimum.
    """
    ae.eval()
    for param in ae.parameters():param.requires_grad_(False)
    truth=torch.as_tensor(y,dtype=torch.float32,device="cpu")
    mean=torch.as_tensor(scaler.mean_,dtype=torch.float32)
    scale=torch.as_tensor(scaler.scale_,dtype=torch.float32)
    enc=torch.as_tensor(z_encoded,dtype=torch.float32)
    prior=torch.as_tensor(z_prior,dtype=torch.float32)
    gen=torch.Generator(device="cpu").manual_seed(seed+11994)
    best_sse=np.full(len(y),np.inf,dtype=np.float64)
    best_pred=np.empty_like(y,dtype=np.float64)
    best_z=np.empty_like(z_encoded,dtype=np.float64)
    for name,init in [("encoded",enc),
                      ("prior",prior),
                      ("encoded_jitter",enc+.10*torch.randn(enc.shape,generator=gen))]:
        z=init.clone().detach().requires_grad_(True)
        optimizer=torch.optim.Adam([z],lr=lr)
        def compute():
            pred=ae.decode(z)*scale+mean
            return torch.mean((pred-truth)**2,dim=1),pred
        # initial solutions are also candidates: optimized oracle cannot be worse
        with torch.no_grad():
            err,pred=compute()
            a=err.detach().cpu().numpy()
            replace=a<best_sse
            best_sse[replace]=a[replace]
            best_pred[replace]=pred.detach().cpu().numpy()[replace]
            best_z[replace]=z.detach().cpu().numpy()[replace]
        for it in range(steps):
            optimizer.zero_grad()
            errors,_=compute()
            errors.sum().backward()
            optimizer.step()
        with torch.no_grad():
            errors,pred=compute()
            a=errors.detach().cpu().numpy()
            replace=a<best_sse
            best_sse[replace]=a[replace]
            best_pred[replace]=pred.detach().cpu().numpy()[replace]
            best_z[replace]=z.detach().cpu().numpy()[replace]
        print("ORACLE_RESTART",name,"n",len(y),
            "rmse",float(np.sqrt(np.mean(best_sse))),flush=True)
    return best_pred,best_z,np.sqrt(best_sse)

def well_rows(seed,split,wells,y,prior,encoder,oracle,z0,zencoded,zoracle,clp_by_m,gp_by_m):
    rows=[]
    for well in sorted(set(wells)):
        indices=np.flatnonzero(wells==well)
        sub=y[indices]
        row=dict(seed=seed,split=split,well_id=well,n_windows=len(indices),
                 y_min=float(np.min(sub)),y_max=float(np.max(sub)),
                 y_p99=float(np.quantile(sub,.99)))
        for name,pre in [('prior',prior),('encoded',encoder),('oracle',oracle)]:
            row[name+"_rmse_full"]=met(pre[indices],sub)['rmse']
            row[name+"_mae_full"]=met(pre[indices],sub)['mae']
        # Calculate mismatch of codes; scale-free values should be interpreted
        # relatively, not as physical distance.
        row["latent_prior_to_encoder_rms"]=float(np.sqrt(np.mean((z0[indices]-zencoded[indices])**2)))
        row["latent_prior_to_oracle_rms"]=float(np.sqrt(np.mean((z0[indices]-zoracle[indices])**2)))
        for m in sorted(clp_by_m):
            hid=clp_by_m[m][1]
            row[f"clp_{m}_rmse_hidden"]=met(clp_by_m[m][0][indices],sub,hid)['rmse']
            row[f"oracle_{m}_rmse_hidden"]=met(oracle[indices],sub,hid)['rmse']
            row[f"gp_{m}_rmse_hidden"]=met(gp_by_m[m][indices],sub,hid)['rmse']
        rows.append(row)
    return rows

def bootstrap(deltas,seed=24680,rep=5000):
    d=np.asarray(deltas,dtype=np.float64)
    rng=np.random.default_rng(seed)
    mat=rng.choice(d,size=(rep,len(d)),replace=True)
    q=np.quantile(np.mean(mat,axis=1),[.025,.975])
    return dict(mean=float(np.mean(d)),ci95_low=float(q[0]),ci95_high=float(q[1]),n=len(d))

def run(archive,out,quick=False):
    ts=time.time();torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=True)
    dev,external,wdev,wext,mt,me=full_arrays(archive)
    it,iv=split_dev_wells(wdev)
    train,val=dev[it],dev[iv]
    wval=wdev[iv]
    Xtr,mux,sx=aux_features(train)
    Xv,*_=aux_features(val,mux,sx)
    Xt,*_=aux_features(external,mux,sx)
    Ytr,Yv,Yt=[a[:,:,3].copy() for a in (train,val,external)]
    assert not(set(wdev)&set(wext)) and not (set(wdev[it])&set(wdev[iv]))
    distributions={"train":numerics(Ytr),"validation":numerics(Yv),
                   "external":numerics(Yt)}
    manifest=dict(archive_sha256=ARCHIVE_SHA256,source_fold=0,
      train_n=len(train),validation_n=len(val),external_n=len(external),
      train_wells=len(set(wdev[it])),validation_wells=len(set(wval)),
      external_wells=len(set(wext)),seed_list=[7] if quick else list(SEEDS),
      y_standardization="per-position StandardScaler fit exclusively on train",
      full_target_oracle="DIAGNOSTIC ONLY: uses complete external DTC for inversion",
      representation_oracle="3 starts encoded/prior/jitter; original-units squared error, 450 Adam steps, per-window best, not certified global optimum",
      oracle_scope="best approximately attainable decoder representation; not deployable",
      operational_CLP="unchanged canonical module with validation-selected lambda from previous run",
      caveats=["Original Geolink source normalization may use global statistics",
               "Investigated regime chosen on earlier Geolink pilot; not independent confirmation",
               "Test fold windows are separate by well but from 36 wells only",
               "oracle optimization can remain above global range optimum"])
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (out/"distributions.json").write_text(json.dumps(distributions,indent=2)+"\n")
    print("DATA_AUDIT",json.dumps(dict(splits={k:len(a) for k,a in [('train',Ytr),('validation',Yv),('external',Yt)]},
         distributions=distributions)),flush=True)
    cfg=SimpleNamespace(csgm_latent_dim=16,csgm_hidden_dim=128,
       csgm_batch_size=64,csgm_ae_lr=.001,csgm_weight_decay=1e-5,
       csgm_ae_epochs=12 if quick else 200,csgm_iters=35 if quick else 400,
       csgm_opt_lr=.05,csgm_restarts=1 if quick else 3)
    summary=[];well_metrics=[];per_window=[];diagnostics=[]
    selected_lambdas={
      7:{32:.003,64:.003},
      23:{32:.003,64:.003},
      41:{32:.001,64:.003},
    }
    for seed in ([7] if quick else list(SEEDS)):
        m_start=time.time()
        scaler=StandardScaler().fit(Ytr)
        ae=train_ae_generator(scaler.transform(Ytr),cfg,seed,"cpu")
        for param in ae.parameters():param.requires_grad_(False)
        ztrain=encode_y(ae,scaler.transform(Ytr),"cpu")
        pred=RidgePrior(alpha=1.,random_state=seed+7000).fit(Xtr,ztrain)
        groups={}
        for split,arr,X,w in (("train",Ytr,Xtr,wdev[it]),("validation",Yv,Xv,wval),("external",Yt,Xt,wext)):
            z0=pred.predict(X)
            zinc=encode_y(ae,scaler.transform(arr),"cpu")
            encoded=decode_latent_prior(ae,zinc,scaler.mean_,scaler.scale_,"cpu")
            cond=decode_latent_prior(ae,z0,scaler.mean_,scaler.scale_,"cpu")
            groups[split]=dict(z0=z0,encoded_z=zinc,encoder=encoded,prior=cond,
              Y=arr,wells=w)
            print("RECON",seed,split,"encoded",met(encoded,arr),
                  "prior",met(cond,arr),flush=True)
        for split in ("validation","external"):
            gr=groups[split]
            oracle,zbest,er=latent_oracle(
              ae,gr["Y"],scaler,gr["encoded_z"],gr["z0"],seed,
              steps=35 if quick else 450)
            gr["oracle"]=oracle;gr["oracle_z"]=zbest;gr["oracle_rmse_rows"]=er
            print("ORACLE",seed,split,"orig_units",met(oracle,gr["Y"]),
                  "encoded",met(gr["encoder"],gr["Y"]),flush=True)
        ext=groups["external"]
        for split in ("train","validation","external"):
            gr=groups[split]
            row=dict(seed=seed,split=split,n_windows=len(gr["Y"]),
                 n_wells=len(set(gr["wells"])),
                 **{"encoded_"+k:v for k,v in met(gr["encoder"],gr["Y"]).items()},
                 **{"prior_"+k:v for k,v in met(gr["prior"],gr["Y"]).items()})
            if "oracle" in gr:
                row.update({"oracle_"+k:v for k,v in met(gr["oracle"],gr["Y"]).items()})
                row["oracle_better_encoder_windows"]=int(np.sum(
                    np.mean((gr["oracle"]-gr["Y"])**2,axis=1) <
                    np.mean((gr["encoder"]-gr["Y"])**2,axis=1)-1e-10))
            diagnostics.append(row)
        for m in OBS_COUNTS:
            ids=obs_positions(m,seed);hid=np.setdiff1d(np.arange(L),ids)
            ztest=ext["z0"]
            M=np.eye(L)[ids];bt=Yt[:,ids]
            # Canonical recovery unchanged, with previously validation-selected
            # lambda; recompute for trustworthy paired diagnostics and audit.
            lam=selected_lambdas[seed][m]
            clp=csgm_recover_with_prior(ae,M,bt,ztest,scaler.mean_,scaler.scale_,
                            lam,cfg.csgm_iters,cfg.csgm_opt_lr,cfg.csgm_restarts,
                            "cpu",seed)
            gp_choice=choose_gp(Yv,Yv[:,ids],ids,hid,None)
            gp=gp_adjust(bt,ids,gp_choice[3],None)
            interp=np.stack([np.interp(np.arange(L),ids,b) for b in bt])
            constant=np.broadcast_to(bt.mean(axis=1,keepdims=True),Yt.shape)
            mres=dict(seed=seed,observed=m,support_start=int(ids[0]),
              support_end=int(ids[-1]),lambda_original=lam,
              gp_length=gp_choice[1],gp_regularization=gp_choice[2],
              external_n=len(Yt),n_wells=len(set(wext)))
            for name,P in [("clp",clp),("oracle",ext["oracle"]),
                           ("encoded",ext["encoder"]),("prior",ext["prior"]),
                           ("gp",gp),("interpolation",interp),("constant",constant)]:
                mres.update({name+"_"+k:v for k,v in met(P,Yt,hid).items()})
            mres["oracle_vs_gp_rmse_ratio"]=mres["oracle_rmse"]/mres["gp_rmse"]
            mres["clp_vs_oracle_rmse_ratio"]=mres["clp_rmse"]/max(1e-9,mres["oracle_rmse"])
            summary.append(mres)
            print("CELL",json.dumps(mres),flush=True)
            ext.setdefault("clp_m",{})[m]=(clp,hid)
            ext.setdefault("gp_m",{})[m]=gp
            val=groups["validation"]
            for i,well in enumerate(wext):
                rec=dict(seed=seed,observed=m,well_id=str(well),window=i,
                     oracle_full_rmse=float(np.sqrt(np.mean((ext["oracle"][i]-Yt[i])**2))),
                     encoded_full_rmse=float(np.sqrt(np.mean((ext["encoder"][i]-Yt[i])**2))),
                     prior_full_rmse=float(np.sqrt(np.mean((ext["prior"][i]-Yt[i])**2))),
                     clp_hidden_rmse=float(np.sqrt(np.mean((clp[i,hid]-Yt[i,hid])**2))),
                     oracle_hidden_rmse=float(np.sqrt(np.mean((ext["oracle"][i,hid]-Yt[i,hid])**2))),
                     gp_hidden_rmse=float(np.sqrt(np.mean((gp[i,hid]-Yt[i,hid])**2))),
                     y_max=float(np.max(Yt[i])),y_min=float(np.min(Yt[i])))
                per_window.append(rec)
            csv_write(out/"metrics_by_seed.csv",summary)
            csv_write(out/"per_window.csv",per_window)
        well_metrics.extend(well_rows(seed,"external",wext,Yt,ext["prior"],
           ext["encoder"],ext["oracle"],ext["z0"],ext["encoded_z"],ext["oracle_z"],
           ext["clp_m"],ext["gp_m"]))
        csv_write(out/"well_diagnostics.csv",well_metrics)
        csv_write(out/"representation_by_split.csv",diagnostics)
        print("SEED_DONE",seed,round(time.time()-m_start,1),flush=True)
    # Per-well paired CLP vs approx oracle, CLP vs GP, approx oracle vs GP
    bstrap=[]
    for m in OBS_COUNTS:
        wells=sorted(set(wext))
        for name,a,b in [
            ("oracle_minus_gp",f"oracle_{m}_rmse_hidden",f"gp_{m}_rmse_hidden"),
            ("clp_minus_oracle",f"clp_{m}_rmse_hidden",f"oracle_{m}_rmse_hidden"),
            ("clp_minus_gp",f"clp_{m}_rmse_hidden",f"gp_{m}_rmse_hidden")]:
            diffs=[]
            for well in wells:
                q=[z for z in well_metrics if z["well_id"]==well]
                diffs.append(np.mean([z[a]-z[b] for z in q]))
            bstrap.append(dict(observed=m,contrast=name,**bootstrap(diffs,seed=1031+m+len(name))))
    csv_write(out/"bootstrap_by_well.csv",bstrap)
    # Top wells ranked by full-scale CLP error, averaged over 3 seeds.
    profiles=[]
    for well in sorted(set(wext)):
        a=[z for z in well_metrics if z["well_id"]==well]
        outrow=dict(well_id=well,n_windows=a[0]["n_windows"],y_max=float(np.max([q["y_max"] for q in a])),
           y_p99=float(np.mean([q["y_p99"] for q in a])))
        for k in ["oracle_rmse_full","encoded_rmse_full","prior_rmse_full",
                  "latent_prior_to_oracle_rms",
                  "clp_32_rmse_hidden","clp_64_rmse_hidden",
                  "gp_32_rmse_hidden","gp_64_rmse_hidden",
                  "oracle_32_rmse_hidden","oracle_64_rmse_hidden"]:
            outrow[k]=float(np.mean([q[k] for q in a]))
        profiles.append(outrow)
    profiles.sort(key=lambda x:x["clp_32_rmse_hidden"],reverse=True)
    csv_write(out/"well_ranking.csv",profiles)
    (out/"diagnostic_summary.json").write_text(json.dumps({
      "representation":diagnostics,
      "sparse_support":summary,
      "bootstrap":bstrap,
      "top_wells":profiles[:12],
      "elapsed_seconds":time.time()-ts,
      "warnings":["Approximate oracle uses external full target and is NOT a deployable method",
                  "Oracle is not guaranteed global optimum",
                  "Do not select hyperparameters or best well by external evaluation"]},indent=2)+"\n")
    print("SUMMARY",json.dumps({"representation":diagnostics,"sparse":summary,
       "bootstrap":bstrap,"top_wells":profiles[:12]}),flush=True)
    print("DONE",len(summary),"cells",round(time.time()-ts,1),"sec",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--archive",type=Path,default=Path("/tmp/geolink/public-processed-datasets.tar.gz"))
    p.add_argument("--out",type=Path,default=Path("outputs/geolink_oracle_audit"))
    p.add_argument("--quick",action="store_true")
    a=p.parse_args()
    run(a.archive,a.out,a.quick)

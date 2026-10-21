#!/usr/bin/env python3
"""Investigation 19: Geolink fold-0 stratified *development* validation and
latent-recovery initializations. Research diagnostics, not manuscript evidence.

Frozen 32/64-consecutive-support protocol, fixed 36 outer-test wells.
Test target is never used for hyperparameter choice or operational initialization.
"encoded_oracle_init" uses full external DTC ONLY as a diagnostic control.
"""
from __future__ import annotations
import argparse,csv,json,math,time,os
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from csgm_m2_module import (RidgePrior,train_ae_generator,encode_y,
    decode_latent_prior,csgm_recover_with_prior)
from scripts.geolink_fullscale.run_fold0 import (
    full_arrays,split_dev_wells,aux_features,obs_positions,choose_gp,gp_adjust,
    hidden_stats,SEEDS,OBS_COUNTS,L,ARCHIVE_SHA256
)

LAMS=(.0001,.0003,.001,.003,.01,.03,.1)
def emit_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with open(path,"w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def describe(y):
    x=np.asarray(y).reshape(-1)
    return dict(n=len(x),mean=float(np.mean(x)),sd=float(np.std(x)),
      p50=float(np.quantile(x,.5)),p99=float(np.quantile(x,.99)),
      max=float(np.max(x)),count_gt2=int(np.sum(x>2)))

def stratified_by_well(wells,target,fraction=.2,seed=20261021):
    unique=np.unique(wells)
    max_per={w:float(np.max(target[wells==w])) for w in unique}
    high=[w for w in unique if max_per[w]>2.]
    selection_rule="well_max_DTC>2_in_DEVELOPMENT_only"
    # If high-value wells are extremely rare, avoid a validation set with no
    # elevated-target wells by widening the DEVELOPMENT-ONLY top-tail threshold.
    if len(high)<2:
        cut=float(np.quantile(list(max_per.values()),.8))
        high=[w for w in unique if max_per[w]>=cut]
        selection_rule="fallback_top20pct_well_max_DTC_development_only"
    lo=[w for w in unique if w not in set(high)]
    rng=np.random.default_rng(seed)
    hi=np.array(high,dtype=str);lo=np.array(lo,dtype=str)
    rng.shuffle(hi);rng.shuffle(lo)
    # Ensure both model-training and validation wells contain upper-tail wells.
    nv_total=int(round(len(unique)*fraction))
    hi_valid=max(1,min(len(hi)-1,int(round(len(hi)*fraction))))
    low_valid=nv_total-hi_valid
    assert 0<hi_valid<len(hi) and 0<low_valid<len(lo)
    val=set(hi[:hi_valid])|set(lo[:low_valid])
    mask=np.array([w in val for w in wells])
    assert not (set(wells[mask])&set(wells[~mask]))
    meta=dict(rule=selection_rule,top_tail_wells_total=len(hi),
      top_tail_wells_val=hi_valid,top_tail_wells_train=len(hi)-hi_valid,
      val_wells=len(set(wells[mask])),train_wells=len(set(wells[~mask])),
      train_extreme_entries=int(np.sum(target[~mask]>2)),
      val_extreme_entries=int(np.sum(target[mask]>2)),
      high_well_list=[str(w) for w in sorted(high)],
      val_high_well_list=[str(w) for w in sorted(val&set(high))])
    return ~mask,mask,meta

def perwell_rmse(y,ypred,hidden,wells):
    vals=[]
    for w in np.unique(wells):
        idx=wells==w
        vals.append(float(np.sqrt(np.mean((ypred[idx][:,hidden]-y[idx][:,hidden])**2))))
    return float(np.mean(vals)),vals

def sparse_optimizer(ae,obs,ids,z0,mean,scale,lam,train_z,seed,
                     mode,steps=400,lr=.05,n_restarts=3,oracle_z=None):
    """Independent per-window best objective, same per-window CLP objective.
    Original's global restart selector is NOT used in this controlled variant.
    """
    n=len(obs)
    zprior=torch.tensor(z0,dtype=torch.float32)
    truth_obs=torch.tensor(obs,dtype=torch.float32)
    mean=torch.tensor(mean,dtype=torch.float32)
    scale=torch.tensor(scale,dtype=torch.float32)
    rng=np.random.default_rng(seed+99211+len(ids)*71)
    generator=torch.Generator().manual_seed(seed+17001+len(ids)*13)
    latent_pool=np.asarray(train_z,dtype=np.float32)
    best=np.full(n,np.inf)
    winner=np.zeros((n,L),dtype=np.float64)
    winner_z=np.zeros_like(z0,dtype=np.float64)
    original_y=np.asarray(obs)
    for restart in range(n_restarts):
        if mode=="prior_jitter":
            init=zprior.clone() if restart==0 else zprior+0.05*torch.randn(zprior.shape,generator=generator)
        elif mode=="train_latent_restart":
            if restart==0:init=zprior.clone()
            else:
                ids_pool=rng.integers(0,len(latent_pool),size=n)
                init=torch.tensor(latent_pool[ids_pool])
        elif mode=="oracle_encoder":
            assert oracle_z is not None and restart==0
            init=torch.as_tensor(oracle_z,dtype=torch.float32).clone()
        else:raise ValueError(mode)
        z=init.detach().requires_grad_(True)
        opt=torch.optim.Adam([z],lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            recon=ae.decode(z)*scale+mean
            data=torch.sum((recon[:,ids]-truth_obs)**2,dim=1)
            prior=float(lam)*torch.sum((z-zprior)**2,dim=1)
            (data+prior).sum().backward()
            opt.step()
        with torch.no_grad():
            recon=ae.decode(z)*scale+mean
            losses=(torch.sum((recon[:,ids]-truth_obs)**2,dim=1)+
                  float(lam)*torch.sum((z-zprior)**2,dim=1)).numpy()
            re=losses<best
            best[re]=losses[re]
            winner[re]=recon.numpy()[re]
            winner_z[re]=z.detach().numpy()[re]
    return winner,best,winner_z

def bootstrap(values,seed=20261021,nboot=3000):
    rng=np.random.default_rng(seed)
    d=np.asarray(values,float)
    b=np.mean(rng.choice(d,size=(nboot,len(d)),replace=True),axis=1)
    return dict(mean=float(np.mean(d)),ci95lo=float(np.quantile(b,.025)),
                ci95hi=float(np.quantile(b,.975)),n=len(d))

def run(archive,output,quick=False):
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    tic=time.monotonic()
    dev,ext,wdev,wext,_,_=full_arrays(archive)
    ydev=dev[:,:,3]
    yext=ext[:,:,3]
    assert len(ext)==1019 and len(np.unique(wext))==36
    ntr,nva=split_dev_wells(wdev)
    strtr,strva,sm=stratified_by_well(wdev,ydev)
    original_val_regime={k:describe(ydev[q]) for k,q in
       [('train',ntr),('validation',nva)]}
    strat_val_regime={k:describe(ydev[q]) for k,q in
       [('train',strtr),('validation',strva)]}
    print("SPLIT_REGIMES",json.dumps(dict(original=original_val_regime,
       stratified=strat_val_regime,meta=sm,external=describe(yext))),flush=True)
    meta=dict(input_hash=ARCHIVE_SHA256,fold=0,
      outer_test_wells=len(np.unique(wext)),outer_test_windows=len(ext),
      dev_wells=len(np.unique(wdev)),dev_windows=len(dev),support=[32,64],
      seeds=[7] if quick else list(SEEDS),stratification=sm,
      original_split=original_val_regime,stratified_split=strat_val_regime,
      validation_only_selection="lambda original grid, global RMSE and well-macro RMSE separately",
      oracle_warning="EXTERNAL full DTC enters ONLY diagnostic encoder initialization. Nonoperational.",
      modes=["original_canonical_batch_restart","same_starts_per_window_selector",
             "three_starts_including_train_latent_codes","full_truth_encoder_start_diagnostic",
             "full_truth_encoder_start_lambda_zero_diagnostic"],
      caveats=["source-preprocessed Geolink may contain global normalization",
               "outer fold already viewed, so results remain EXPLORATORY",
               "group-disjoint split does not prove geological representativeness",
               "full_target_encoder_start is not a feasible deployment"])
    (output/"manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
    config=SimpleNamespace(csgm_latent_dim=16,csgm_hidden_dim=128,
      csgm_batch_size=64,csgm_ae_lr=.001,csgm_weight_decay=1e-5,
      csgm_ae_epochs=12 if quick else 200,
      csgm_iters=35 if quick else 400,csgm_opt_lr=.05,
      csgm_restarts=1 if quick else 3)
    aggregate=[];wal=[];valscore=[]
    regimes={"original":(ntr,nva),"stratified":(strtr,strva)}
    for regime in regimes:
        trainmask,valmask=regimes[regime]
        train,val=dev[trainmask],dev[valmask]
        wellval=wdev[valmask]
        Ytr,Yv=train[:,:,3],val[:,:,3]
        Xtr,mu,sd=aux_features(train)
        Xv,*_=aux_features(val,mu,sd)
        Xt,*_=aux_features(ext,mu,sd)
        for seed in ([7] if quick else SEEDS):
            stamp=time.monotonic()
            scale=StandardScaler().fit(Ytr)
            ae=train_ae_generator(scale.transform(Ytr),config,seed,"cpu")
            for p in ae.parameters():p.requires_grad_(False)
            ztr=encode_y(ae,scale.transform(Ytr),"cpu")
            prior=RidgePrior(alpha=1.,random_state=seed+7000).fit(Xtr,ztr)
            zv,zt=prior.predict(Xv),prior.predict(Xt)
            # External complete DTC only for diagnosis; forbidden for deployed
            # models or validation selection.
            zfull=encode_y(ae,scale.transform(yext),"cpu")
            yenc=decode_latent_prior(ae,zfull,scale.mean_,scale.scale_,"cpu")
            yprior=decode_latent_prior(ae,zt,scale.mean_,scale.scale_,"cpu")
            print("MODEL",regime,seed,"dev",len(train),"val",len(val),
                  "external full encoder RMSE",np.sqrt(np.mean((yenc-yext)**2)),
                  "external prior RMSE",np.sqrt(np.mean((yprior-yext)**2)),flush=True)
            for nobs in OBS_COUNTS:
                ids=obs_positions(nobs,seed)
                hid=np.setdiff1d(np.arange(L),ids)
                M=np.eye(L)[ids]
                bval=Yv[:,ids]
                bt=yext[:,ids]
                # No held-out test truth used for any choice.
                choices=[]
                for lam in ([.003] if quick else LAMS):
                    pred=csgm_recover_with_prior(
                        ae,M,bval,zv,scale.mean_,scale.scale_,lam,
                        config.csgm_iters,config.csgm_opt_lr,
                        config.csgm_restarts,"cpu",seed)
                    gl=hidden_stats(Yv,pred,hid)["rmse"]
                    macro,ws=perwell_rmse(Yv,pred,hid,wellval)
                    choices.append((gl,macro,lam))
                    vals=dict(regime=regime,seed=seed,observed=nobs,lambda_value=lam,
                        global_validation_rmse=gl,well_macro_rmse=macro,
                        well_validation_max=float(np.max(ws)),
                        validation_high_entries=int(np.sum(Yv[:,hid]>2)))
                    valscore.append(vals)
                best_global=min(choices,key=lambda z:(z[0],z[2]))[2]
                best_macro=min(choices,key=lambda z:(z[1],z[2]))[2]
                gp_choice=choose_gp(Yv,bval,ids,hid,None)
                gp=gp_adjust(bt,ids,gp_choice[3],None)
                interp=np.stack([np.interp(np.arange(L),ids,b) for b in bt])
                const=np.broadcast_to(bt.mean(1,keepdims=True),yext.shape)
                base={"GP":gp,"interpolation":interp,"constant":const,
                      "prior_only":yprior,"full_truth_AE_encoder_oracle":yenc}
                lamconfigs=[("global_lambda",best_global)]
                if best_macro!=best_global:
                    lamconfigs.append(("macro_lambda",best_macro))
                for lbl,lam in lamconfigs:
                    canonical=csgm_recover_with_prior(
                        ae,M,bt,zt,scale.mean_,scale.scale_,lam,
                        config.csgm_iters,config.csgm_opt_lr,
                        config.csgm_restarts,"cpu",seed)
                    best_window,objective,zout=sparse_optimizer(
                        ae,bt,ids,zt,scale.mean_,scale.scale_,lam,ztr,seed,
                        mode="prior_jitter",steps=config.csgm_iters,
                        n_restarts=config.csgm_restarts)
                    broad,_,_=sparse_optimizer(
                        ae,bt,ids,zt,scale.mean_,scale.scale_,lam,ztr,seed,
                        mode="train_latent_restart",steps=config.csgm_iters,
                        n_restarts=config.csgm_restarts)
                    # Diagnostic only, forbidden as deployable algorithm.
                    zstart,_,_=sparse_optimizer(
                        ae,bt,ids,zt,scale.mean_,scale.scale_,lam,ztr,seed,
                        mode="oracle_encoder",steps=config.csgm_iters,
                        n_restarts=1,oracle_z=zfull)
                    zstart_free,_,_=sparse_optimizer(
                        ae,bt,ids,zt,scale.mean_,scale.scale_,0.,ztr,seed,
                        mode="oracle_encoder",steps=config.csgm_iters,
                        n_restarts=1,oracle_z=zfull)
                    methods={**base,"canonical_CLP":canonical,
                      "perwindow_CLP":best_window,
                      "broad_restart_CLP":broad,
                      "diagnostic_oracle_initialized_CLP":zstart,
                      "diagnostic_oracle_initialized_no_prior":zstart_free}
                    entry=dict(regime=regime,seed=seed,observed=nobs,
                      lambda_selection=lbl,selected_lambda=lam,
                      val_lambda_global=best_global,val_lambda_macro=best_macro,
                      val_best_global=min(x[0] for x in choices),
                      val_best_macro=min(x[1] for x in choices),
                      n_train=len(train),n_val=len(val),n_test=len(ext),
                      support_start=int(ids[0]),support_end=int(ids[-1]))
                    for method,preds in methods.items():
                        metrics=hidden_stats(yext,preds,hid)
                        for k,v in metrics.items():entry[method+"_"+k]=v
                        entry[method+"_observed_rmse"]=float(np.sqrt(np.mean(
                            (preds[:,ids]-bt)**2)))
                    aggregate.append(entry)
                    print("CELL",json.dumps(entry),flush=True)
                    # well bootstrap units; every window belongs to exactly one well
                    for well in sorted(set(wext)):
                        idx=wext==well
                        record=dict(regime=regime,seed=seed,observed=nobs,
                            lambda_selection=lbl,well_id=str(well),
                            n_windows=int(np.sum(idx)),has_gt2=bool(np.any(yext[idx]>2)))
                        for method,preds in methods.items():
                            record[method+"_rmse"]=hidden_stats(
                                yext[idx],preds[idx],hid)["rmse"]
                        wal.append(record)
                    emit_csv(output/"metrics_by_cell.csv",aggregate)
                    emit_csv(output/"metrics_by_well.csv",wal)
                    emit_csv(output/"validation_lambda.csv",valscore)
                print("REGIME_SEED",regime,seed,nobs,
                     "elapsed_s",round(time.monotonic()-stamp,1),flush=True)
    # Aggregate across independent wells (same well repeated across seeds)
    contrasts=[]
    for regime in regimes:
        for nobs in OBS_COUNTS:
            sub=[r for r in wal if r["regime"]==regime and
              r["observed"]==nobs and r["lambda_selection"]=="global_lambda"]
            for other in ("GP","interpolation","constant",
                    "perwindow_CLP","broad_restart_CLP",
                    "diagnostic_oracle_initialized_CLP",
                    "diagnostic_oracle_initialized_no_prior"):
                dd=[]
                for well in sorted(set(wext)):
                    rw=[r for r in sub if r["well_id"]==well]
                    dd.append(float(np.mean([r["canonical_CLP_rmse"]-
                              r[other+"_rmse"] for r in rw])))
                contrasts.append(dict(regime=regime,observed=nobs,
                    contrast="canonical_minus_"+other,**bootstrap(dd)))
    emit_csv(output/"well_bootstrap.csv",contrasts)
    meta["n_cells"]=len(aggregate)
    meta["seconds_elapsed"]=round(time.monotonic()-tic,2)
    (output/"manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
    (output/"summary.json").write_text(json.dumps({
       "cells":aggregate,"validation_selection":valscore,
       "bootstrap":contrasts,"stratified_split":sm},indent=2)+"\n")
    print("ALL_DONE",len(aggregate),"elapsed_s",meta["seconds_elapsed"],
          "bootstrap",json.dumps(contrasts),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--archive",type=Path,default=Path("/tmp/geolink/public-processed-datasets.tar.gz"))
    p.add_argument("--out",type=Path,default=Path("outputs/geolink_recovery_decomposition19"))
    p.add_argument("--quick",action="store_true")
    a=p.parse_args()
    run(a.archive,a.out,quick=a.quick)

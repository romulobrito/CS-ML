#!/usr/bin/env python3
"""Investigation 20: Is the Geolink CLP conditional prior a harmful latent anchor?

Full fold-0 real wells, frozen test 36 wells, frozen 32/64 consecutive measurements.
All lambda selection occurs on development-only, well-disjoint validation data.
The outer fold has ALREADY been examined: research diagnostic, NOT independent
confirmation. Complete DTC is used ONLY for the retrospective encoder diagnostic.
"""
from __future__ import annotations
import argparse,csv,json,os,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from csgm_m2_module import RidgePrior, train_ae_generator, encode_y, decode_latent_prior
from scripts.geolink_fullscale.run_fold0 import (
  full_arrays,split_dev_wells,aux_features,obs_positions,hidden_stats,
  SEEDS,OBS_COUNTS,L,ARCHIVE_SHA256,choose_gp,gp_adjust)
from scripts.geolink_fullscale.investigation19 import stratified_by_well

LAM_GRID=(0.,1e-6,1e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,1.)
NUM_RESTARTS=3

def write_csv(path,rows):
    if not rows:return
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=columns)
        w.writeheader()
        w.writerows(rows)

def well_macro(y,ypred,hid,wells):
    return float(np.mean([
      hidden_stats(y[wells==w],ypred[wells==w],hid)["rmse"]
      for w in sorted(set(wells))]))

def perwell_records(y,wells,hid,methods,metadata):
    rows=[]
    for w in sorted(set(wells)):
        ix=wells==w
        row={**metadata,"well_id":str(w),"n_windows":int(np.sum(ix)),
             "max_target_full":float(np.max(y[ix])),
             "has_extreme_gt2":bool(np.max(y[ix])>2)}
        for name,pred in methods.items():
            sc=hidden_stats(y[ix],pred[ix],hid)
            for k,v in sc.items():row[name+"_"+k]=v
        rows.append(row)
    return rows

def choose_initials(mode,train_z,zprior,seed,nobs,nstarts=NUM_RESTARTS):
    """Predictable initialization; never reads the hidden target."""
    n,dim=zprior.shape
    rng=np.random.default_rng(110007+seed*101+nobs*337)
    if mode=="conditional":
        gen=torch.Generator().manual_seed(99003+seed*307+nobs)
        t=torch.as_tensor(zprior,dtype=torch.float32)
        return [t.clone() if i==0 else t+0.05*torch.randn(t.shape,generator=gen)
                for i in range(nstarts)]
    if mode=="unconditional":
        gen=torch.Generator().manual_seed(23501+seed*307+nobs)
        pool=np.asarray(train_z)
        mean=pool.mean(axis=0)
        return [torch.tensor(np.broadcast_to(mean,(n,dim)).copy(),dtype=torch.float32)] + [
          torch.tensor(pool[rng.integers(0,len(pool),size=n)].copy(),dtype=torch.float32)
          for _ in range(nstarts-1)]
    raise ValueError(mode)

def recover(ae,y_obs,positions,zprior,train_z,mean,scale,lam,
            mode,seed,steps=400,lr=.05,restarts=NUM_RESTARTS):
    """Per-window model-objective restart selection, identical across arms.

    Conditional λ>0: prior penalty + conditional starts.
    Conditional λ=0: NO regularization but same conditional starts.
    Unconditional λ=0: NO regularization and starts independent of X.
    All three arms use ONLY the same sparse measurements in the loss.
    """
    ae.eval()
    torch.set_num_threads(1)
    obs=torch.as_tensor(y_obs,dtype=torch.float32)
    zp=torch.as_tensor(zprior,dtype=torch.float32)
    m=torch.as_tensor(mean,dtype=torch.float32)
    s=torch.as_tensor(scale,dtype=torch.float32)
    inits=choose_initials(mode,train_z,zprior,seed,len(positions),restarts)
    best=np.full(len(obs),np.inf)
    pred_best=np.zeros((len(obs),L),dtype=np.float64)
    codes_best=np.zeros_like(zprior,dtype=np.float64)
    fit_best=np.zeros(len(obs),dtype=np.float64)
    for st,zinit in enumerate(inits):
        z=zinit.detach().clone().requires_grad_(True)
        optimizer=torch.optim.Adam([z],lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            out=ae.decode(z)*s+m
            miss=out[:,positions]-obs
            fit=torch.sum(miss*miss,dim=1)
            penalty=float(lam)*torch.sum((z-zp)**2,dim=1)
            torch.sum(fit+penalty).backward()
            optimizer.step()
        with torch.no_grad():
            out=ae.decode(z)*s+m
            fit=torch.sum((out[:,positions]-obs)**2,dim=1).numpy()
            penalty=(float(lam)*torch.sum((z-zp)**2,dim=1)).numpy()
            objective=fit+penalty
            improving=objective<best
            best[improving]=objective[improving]
            fit_best[improving]=fit[improving]
            pred_best[improving]=out.numpy()[improving]
            codes_best[improving]=z.detach().numpy()[improving]
    return pred_best,codes_best,dict(mean_obs_fit=float(np.mean(fit_best)),
                                     mean_objective=float(np.mean(best)))

def prior_latent_diagnostics(train_z,val_z,val_prior,val_y,ext_z,ext_prior,ext_y,
                             seed,regime):
    mu=np.mean(train_z,axis=0)
    sig=np.maximum(np.std(train_z,axis=0),1e-8)
    rows=[]
    for split,target,z,zp in [
        ("validation",val_y,val_z,val_prior),
        ("external",ext_y,ext_z,ext_prior)]:
        e=(zp-z)/sig
        for k in range(z.shape[1]):
            corr=float(np.corrcoef(z[:,k],zp[:,k])[0,1]) if np.std(zp[:,k])>1e-12 and np.std(z[:,k])>1e-12 else None
            rows.append(dict(seed=seed,regime=regime,split=split,latent_dim=k,
                train_mu=float(mu[k]),train_sigma=float(sig[k]),
                encoder_mu=float(np.mean(z[:,k])),predicted_mu=float(np.mean(zp[:,k])),
                standardized_bias=float(np.mean(e[:,k])),
                standardized_mae=float(np.mean(np.abs(e[:,k]))),
                standardized_rmse=float(np.sqrt(np.mean(e[:,k]**2))),
                correlation=corr))
        high=np.max(target,axis=1)>2.
        for group,mask in [("ordinary",~high),("upper_tail",high)]:
            if not np.any(mask):continue
            dist=np.sqrt(np.mean(e[mask]**2,axis=1))
            rows.append(dict(seed=seed,regime=regime,split=split,
                latent_dim=group,number_windows=int(np.sum(mask)),
                standardized_latent_rmse=float(np.sqrt(np.mean(e[mask]**2))),
                standardized_latent_distance_mean=float(np.mean(dist)),
                standardized_latent_distance_p90=float(np.quantile(dist,.9))))
    return rows

def paired_ci(rows,regime,nobs,name,baseline,nboot=3000):
    group=[r for r in rows if r["regime"]==regime and r["observed"]==nobs
             and r["selection"]=="global"]
    wells=sorted(set(r["well_id"] for r in group))
    diff=[]
    for w in wells:
        rr=[r for r in group if r["well_id"]==w]
        diff.append(np.mean([r[name+"_rmse"]-r[baseline+"_rmse"] for r in rr]))
    d=np.array(diff)
    rng=np.random.default_rng(4050+len(name)*17+nobs)
    draws=rng.choice(d,size=(nboot,len(d)),replace=True).mean(axis=1)
    return dict(regime=regime,n_obs=nobs,model=name,comparison=baseline,
      delta_rmse_model_minus_baseline=float(np.mean(d)),
      lo=float(np.quantile(draws,.025)),hi=float(np.quantile(draws,.975)),
      n_wells=len(wells),wins_wells=int(np.sum(d<0)))

def main(archive,output,quick=False):
    started=time.time()
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    dev,ext,wdev,wext,_,_=full_arrays(archive)
    origin_train,origin_val=split_dev_wells(wdev)
    strat_train,strat_val,strat_meta=stratified_by_well(wdev,dev[:,:,3])
    splits=[("original",origin_train,origin_val),
            ("stratified",strat_train,strat_val)]
    options=SimpleNamespace(csgm_latent_dim=16,csgm_hidden_dim=128,
      csgm_batch_size=64,csgm_ae_lr=.001,csgm_weight_decay=1e-5,
      csgm_ae_epochs=12 if quick else 200)
    steps=30 if quick else 400
    lams=(0.,.003) if quick else LAM_GRID
    manifest=dict(study="prior-versus-no-prior in full-fold Geolink",
      input_sha256=ARCHIVE_SHA256,train_outer_windows=3904,
      test_outer_windows=1019,test_wells=int(len(set(wext))),
      seed_list=[7] if quick else list(SEEDS),
      support_counts=list(OBS_COUNTS),all_lambda_values=list(lams),
      restart_count=NUM_RESTARTS,steps=steps,
      loss="sum of squared errors at observed coordinates + lambda*sum squared latent distance to conditioned prior",
      selection="development-only validation: global RMSE and per-well macro RMSE",
      noise="no added noise",stratification=strat_meta,
      honesty=["external fold is repeatedly inspected: exploratory not confirmatory",
        "full external truth encoder used ONLY for retrospective diagnosis",
        "unconditional uses the same number of restarts; train-latent mean and training codes, not target-derived",
        "all methods share fixed masks for a given (seed, support)",
        "no retuning based on outer-test outcomes"])
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    summaries=[];validation=[];wells_out=[];latent_out=[]
    for regime,tr_mask,v_mask in splits:
        A=dev[tr_mask];V=dev[v_mask]
        wval=wdev[v_mask]
        yt=A[:,:,3];yv=V[:,:,3];ye=ext[:,:,3]
        xtr,mux,sigx=aux_features(A)
        xv,*_=aux_features(V,mux,sigx)
        xe,*_=aux_features(ext,mux,sigx)
        print("SPLIT",regime,
             "train_windows",len(A),"val_windows",len(V),
             "train_wells",len(set(wdev[tr_mask])),
             "val_wells",len(set(wval)),
             "extreme_validation_windows",int(np.sum(np.max(yv,axis=1)>2)),
             flush=True)
        for seed in ([7] if quick else SEEDS):
            scaler=StandardScaler().fit(yt)
            ae=train_ae_generator(scaler.transform(yt),options,seed,"cpu")
            for p in ae.parameters():p.requires_grad_(False)
            ztr=encode_y(ae,scaler.transform(yt),"cpu")
            prior=RidgePrior(alpha=1.,random_state=seed+7000).fit(xtr,ztr)
            zv,ze=prior.predict(xv),prior.predict(xe)
            # IMPORTANT: full target encoding ONLY for testing if conditional
            # codes are biased; it NEVER initializes operational reconstruction.
            zenc_v=encode_y(ae,scaler.transform(yv),"cpu")
            zenc_e=encode_y(ae,scaler.transform(ye),"cpu")
            latent_out.extend(prior_latent_diagnostics(
                ztr,zenc_v,zv,yv,zenc_e,ze,ye,seed,regime))
            prior_val=decode_latent_prior(ae,zv,scaler.mean_,scaler.scale_,"cpu")
            prior_ext=decode_latent_prior(ae,ze,scaler.mean_,scaler.scale_,"cpu")
            for nobs in OBS_COUNTS:
                ids=obs_positions(nobs,seed)
                hidden=np.setdiff1d(np.arange(L),ids)
                bval=yv[:,ids];bext=ye[:,ids]
                v_candidates=[]
                for lam in lams:
                    yh,_,valobs=recover(ae,bval,ids,zv,ztr,
                       scaler.mean_,scaler.scale_,lam,"conditional",seed,steps=steps)
                    sc=hidden_stats(yv,yh,hidden)
                    macro=well_macro(yv,yh,hidden,wval)
                    rec=dict(regime=regime,seed=seed,observed=nobs,
                             lam=lam,rmse_val=sc["rmse"],
                             macro_rmse_val=macro,
                             observed_fitting_loss=valobs["mean_obs_fit"],
                             upper_tail_validation_windows=int(np.sum(np.max(yv,axis=1)>2)))
                    validation.append(rec)
                    v_candidates.append(rec)
                    print("VAL",json.dumps(rec),flush=True)
                choice_global=min(v_candidates,key=lambda r:(r["rmse_val"],r["lam"]))["lam"]
                choice_macro=min(v_candidates,key=lambda r:(r["macro_rmse_val"],r["lam"]))["lam"]
                gp_val=choose_gp(yv,bval,ids,hidden,None)
                gp_ext=gp_adjust(bext,ids,gp_val[3],None)
                interp=np.stack([np.interp(np.arange(L),ids,row) for row in bext])
                constant=np.broadcast_to(bext.mean(1,keepdims=True),ye.shape).copy()
                pred_uncond,_,uncond_fit=recover(ae,bext,ids,ze,ztr,
                    scaler.mean_,scaler.scale_,0.,"unconditional",seed,steps=steps)
                pred_zinit0,_,zinit_fit=recover(ae,bext,ids,ze,ztr,
                    scaler.mean_,scaler.scale_,0.,"conditional",seed,steps=steps)
                cache={0.:pred_zinit0}
                for label,lam in [("global",choice_global),
                                  ("macro",choice_macro)]:
                    if label=="macro" and choice_macro==choice_global:continue
                    if lam not in cache:
                        cache[lam],_,_=recover(ae,bext,ids,ze,ztr,
                            scaler.mean_,scaler.scale_,lam,"conditional",seed,steps=steps)
                    predictions={"CLP_selected":cache[lam],
                        "conditional_init_no_penalty":pred_zinit0,
                        "unconditional_CSGM_no_prior":pred_uncond,
                        "prior_only":prior_ext,
                        "GP_smooth":gp_ext,
                        "interpolation":interp,
                        "observed_mean":constant}
                    row=dict(regime=regime,seed=seed,observed=nobs,selection=label,
                       chosen_lambda=lam,lambda_global=choice_global,
                       lambda_macro=choice_macro,
                       n_train=len(A),n_validation=len(V),n_external=len(ext),
                       support_start=int(ids[0]),support_end=int(ids[-1]),
                       uncond_observed_fit=uncond_fit["mean_obs_fit"],
                       conditional_no_penalty_observed_fit=zinit_fit["mean_obs_fit"])
                    for name,ypred in predictions.items():
                        for key,value in hidden_stats(ye,ypred,hidden).items():
                            row[name+"_"+key]=value
                        row[name+"_observed_rmse"]=float(np.sqrt(np.mean((ypred[:,ids]-bext)**2)))
                    row["CLP_wins_GP"]=bool(row["CLP_selected_rmse"]<row["GP_smooth_rmse"])
                    row["CLP_wins_uncond"]=bool(row["CLP_selected_rmse"]<row["unconditional_CSGM_no_prior_rmse"])
                    row["CLP_wins_no_penalty"]=bool(row["CLP_selected_rmse"]<row["conditional_init_no_penalty_rmse"])
                    summaries.append(row)
                    wells_out.extend(perwell_records(ye,wext,hidden,predictions,
                       {k:row[k] for k in ("regime","seed","observed","selection","chosen_lambda")}))
                    print("RESULT",json.dumps(row),flush=True)
                    write_csv(output/"metric_cells.csv",summaries)
                    write_csv(output/"metric_wells.csv",wells_out)
                    write_csv(output/"validation_regularization.csv",validation)
                    write_csv(output/"latent_bias_dims.csv",latent_out)
            print("SEED_DONE",regime,seed,"elapsed",round(time.time()-started,2),flush=True)
    contrasts=[]
    for regime,_,_ in splits:
        for count in OBS_COUNTS:
            for name,baseline in [
              ("CLP_selected","unconditional_CSGM_no_prior"),
              ("CLP_selected","conditional_init_no_penalty"),
              ("CLP_selected","GP_smooth"),
              ("conditional_init_no_penalty","unconditional_CSGM_no_prior")]:
                contrasts.append(paired_ci(wells_out,regime,count,name,baseline))
    write_csv(output/"paired_bootstrap_wells.csv",contrasts)
    manifest["duration_seconds"]=time.time()-started
    manifest["n_test_cells"]=len(summaries)
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (output/"summary.json").write_text(json.dumps(dict(
        cells=summaries,contrasts=contrasts,validation=validation,
        latent_bias=latent_out),indent=2)+"\n")
    print("FINISHED",json.dumps(dict(cells=summaries,bootstrap=contrasts)),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--archive",type=Path,
      default=Path("/tmp/geolink/public-processed-datasets.tar.gz"))
    parser.add_argument("--out",type=Path,
      default=Path("outputs/geolink_prior_regularization20"))
    parser.add_argument("--quick",action="store_true")
    a=parser.parse_args()
    main(a.archive,a.out,a.quick)

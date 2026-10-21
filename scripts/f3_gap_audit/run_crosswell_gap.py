#!/usr/bin/env python3
"""Paired F3 cross-well CLP test on grouped-gap masks, using canonical csgm_m2_module.
Scientific protocol: F02-1, F03-2, F06-1 development / F03-4 held out;
64-sample windows at step 16; 3-window train/validation embargo per well;
all methods scored on hidden coordinates only; validation-selected hyperparameters.
"""
from __future__ import annotations
import argparse,csv,json,math,os,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from csgm_m2_module import train_ae_generator,encode_y,RidgePrior,csgm_recover_with_prior,decode_latent_prior

L=64; STEP=16; SEEDS=(7,23,41); MODES=("spread","cluster","two_clusters")
NOISES=(.01,.05); RHO=.10; GP_L=(3,7,15,30,60); GP_A=(.001,.01,.1,1,10)
LAM=(.0001,.0003,.001,.003,.01,.03,.1)
class Mulberry32:
    def __init__(self,seed): self.state=int(seed)&0xffffffff
    def random(self):
        self.state=(self.state+0x6d2b79f5)&0xffffffff
        t=self.state
        t=((t^(t>>15))*(t|1))&0xffffffff
        t^=(t+(((t^(t>>7))*(t|61))&0xffffffff))&0xffffffff
        return ((t^(t>>14))&0xffffffff)/(2**32)
    def gaussian(self):
        return math.sqrt(-2*math.log(max(1e-12,self.random())))*math.cos(2*math.pi*self.random())
def positions(m,mode,seed):
    r=Mulberry32(seed)
    if mode=="spread":
        a=list(range(L))
        for k in range(L-1,0,-1):
            j=int(r.random()*(k+1));a[k],a[j]=a[j],a[k]
        ids=a[:m]
    elif mode=="cluster":
        start=int(r.random()*(L-m+1));ids=list(range(start,start+m))
    elif mode=="two_clusters":
        n1=m//2;n2=m-n1
        s1=2+int(r.random()*8);s2=L-2-n2-int(r.random()*8)
        ids=list(range(s1,s1+n1))+list(range(s2,s2+n2))
    else:raise ValueError(mode)
    return np.asarray(sorted(ids),dtype=int)
def windows(wells):
    out=[]
    for wid,arr in enumerate(wells):
        for st in range(0,len(arr)-L+1,STEP):
            xs=np.concatenate([arr[st:st+L,j] for j in (1,2,4,5)])
            yy=arr[st:st+L,6]
            out.append((wid,xs,yy))
    return out
def data(repo):
    a=np.loadtxt(repo/"data/F02-1,F03-2,F06-1_6logs_30dB.txt",skiprows=1,delimiter="\t")
    b=np.loadtxt(repo/"data/F03-4_6logs_30dB.txt",skiprows=1,delimiter="\t")
    assert a.shape==(1098,7) and b.shape==(832,7),(a.shape,b.shape)
    allw=windows([a[:165],a[165:976],a[976:]])
    tr=[];va=[]
    for wid in range(3):
        rows=[x for x in allw if x[0]==wid]
        n_val=max(1,math.ceil(len(rows)*.12))
        tr.extend(rows[:len(rows)-n_val-3])
        va.extend(rows[-n_val:])
    te=windows([b])
    assert (len(tr),len(va),len(te))==(41,8,49)
    Xs=[np.stack([r[1] for r in rows]) for rows in (tr,va,te)]
    Ys=[np.stack([r[2] for r in rows]) for rows in (tr,va,te)]
    Xtr=Xs[0].reshape(-1,4,L)
    mu=Xtr.mean(axis=(0,2));sd=Xtr.std(axis=(0,2))
    xx=[((x.reshape(-1,4,L)-mu[None,:,None])/sd[None,:,None]).reshape(-1,4*L) for x in Xs]
    return *xx,*Ys
def gp_operator(ids,length,ridge):
    a=np.exp(-.5*((ids[:,None]-ids[None,:])/length)**2)+ridge*np.eye(len(ids))
    b=np.exp(-.5*((np.arange(L)[:,None]-ids[None,:])/length)**2)
    return np.linalg.solve(a,b.T).T
def gp_pred(obs,ids,A,prior=None):
    if prior is None:
        base=np.broadcast_to(obs.mean(axis=1,keepdims=True),(len(obs),L))
    else:base=prior
    return base+(obs-base[:,ids])@A.T
def metrics(pred,truth,hidden):
    d=pred[:,hidden]-truth[:,hidden]
    mse=float(np.mean(d*d))
    y=truth[:,hidden]
    var=float(np.var(y))
    return dict(rmse=math.sqrt(mse),r2=1-mse/var,mae=float(np.mean(np.abs(d))))
def selection(obs,ids,Yv,hidden,priorv=None):
    best=None
    for ell in GP_L:
        for alpha in GP_A:
            A=gp_operator(ids,ell,alpha)
            p=gp_pred(obs,ids,A,priorv)
            mse=float(np.mean((p[:,hidden]-Yv[:,hidden])**2))
            if best is None or mse<best[0]:
                best=(mse,ell,alpha,A)
    return best
def savecsv(path,rows):
    if not rows:return
    with open(path,"w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
def run(out,quick=False):
    torch.set_num_threads(1)
    repo=Path(__file__).resolve().parents[2]
    out.mkdir(parents=True,exist_ok=True)
    Xtr,Xv,Xt,Ytr,Yv,Yt=data(repo)
    cfg=SimpleNamespace(csgm_latent_dim=16,csgm_hidden_dim=128,
                        csgm_batch_size=64,csgm_ae_lr=.001,csgm_weight_decay=1e-5,
                        csgm_ae_epochs=200 if not quick else 12,
                        csgm_iters=400 if not quick else 40,
                        csgm_opt_lr=.05,csgm_restarts=3 if not quick else 1,
                        csgm_ridge_alpha=1.)
    rows=[];predrows=[];vscores=[]
    t0=time.time()
    for seed in (SEEDS if not quick else (7,)):
        ys=StandardScaler().fit(Ytr)
        ae=train_ae_generator(ys.transform(Ytr),cfg,seed,"cpu")
        for p in ae.parameters():p.requires_grad_(False)
        ztr=encode_y(ae,ys.transform(Ytr),"cpu")
        prior=RidgePrior(alpha=1.,random_state=seed+7000).fit(Xtr,ztr)
        zv=prior.predict(Xv);zt=prior.predict(Xt)
        yp=decode_latent_prior(ae,zt,ys.mean_,ys.scale_,"cpu")
        ypv=decode_latent_prior(ae,zv,ys.mean_,ys.scale_,"cpu")
        for mode in (MODES if not quick else ("spread","cluster")):
            for sigma in (NOISES if not quick else (.01,)):
                m=max(4,round(RHO*L))
                ids=positions(m,mode,seed+int(round(RHO*100))*19)
                hidden=np.setdiff1d(np.arange(L),ids)
                rng=Mulberry32(seed+322+int(round(RHO*100))*131+len(mode)*19)
                bv=Yv[:,ids]+sigma*np.asarray([[rng.gaussian() for _ in ids] for _ in range(len(Yv))])
                bt=Yt[:,ids]+sigma*np.asarray([[rng.gaussian() for _ in ids] for _ in range(len(Yt))])
                M=np.eye(L,dtype=float)[ids]
                vl=[]
                for lam in LAM:
                    pred=csgm_recover_with_prior(ae,M,bv,zv,ys.mean_,ys.scale_,float(lam),
                                                 cfg.csgm_iters,cfg.csgm_opt_lr,
                                                 cfg.csgm_restarts,"cpu",seed)
                    vm=float(np.mean((pred[:,hidden]-Yv[:,hidden])**2))
                    vl.append((vm,lam))
                    vscores.append(dict(seed=seed,mode=mode,noise=sigma,rho=RHO,lam=lam,val_hidden_rmse=math.sqrt(vm)))
                valbest,lambd=min(vl,key=lambda t:(t[0],t[1]))
                yc=csgm_recover_with_prior(ae,M,bt,zt,ys.mean_,ys.scale_,
                                           float(lambd),cfg.csgm_iters,cfg.csgm_opt_lr,
                                           cfg.csgm_restarts,"cpu",seed)
                gpS=selection(bv,ids,Yv,hidden,None)
                gpC=selection(bv,ids,Yv,hidden,ypv)
                ysmooth=gp_pred(bt,ids,gpS[3],None)
                ycontext=gp_pred(bt,ids,gpC[3],yp)
                yinterp=np.stack([np.interp(np.arange(L),ids,o) for o in bt])
                ymean=np.broadcast_to(bt.mean(axis=1,keepdims=True),Yt.shape)
                candidates={"CLP":yc,"prior_only":yp,"interpolation":yinterp,
                            "GP_smooth":ysmooth,"observed_mean":ymean,
                            "prior_GP_residual":ycontext}
                outrow=dict(seed=seed,mode=mode,noise=sigma,rho=RHO,m=m,
                     observations=";".join(map(str,ids)),n_test_windows=len(Yt),
                     scored_entries=Yt[:,hidden].size,lambda_selected=lambd,
                     gp_smooth_length=gpS[1],gp_smooth_ridge=gpS[2],
                     residual_gp_length=gpC[1],residual_gp_ridge=gpC[2],
                     val_CLP_hidden_rmse=math.sqrt(valbest))
                for name,P in candidates.items():
                    for measure,value in metrics(P,Yt,hidden).items():
                        outrow[f"{name}_{measure}"]=value
                    outrow[f"{name}_full_window_rmse"]=float(np.sqrt(np.mean((P-Yt)**2)))
                rows.append(outrow)
                savecsv(out/"cell_metrics.csv",rows)
                savecsv(out/"lambda_validation.csv",vscores)
                for wi in range(len(Yt)):
                    for j in hidden:
                        rec=dict(seed=seed,mode=mode,noise=sigma,rho=RHO,
                                 window=wi,coordinate=int(j),is_observed=False,
                                 y_true=float(Yt[wi,j]))
                        for name,P in candidates.items():rec[f"pred_{name}"]=float(P[wi,j])
                        predrows.append(rec)
                savecsv(out/"predictions_hidden.csv",predrows)
                print("CELL",json.dumps(outrow,sort_keys=True),flush=True)
    (out/"manifest.json").write_text(json.dumps(dict(git_commit=os.environ.get("GITHUB_SHA"),
       elapsed_seconds=time.time()-t0,quick=quick,source_data="original GitHub F3 30dB tables",
       heldout_well="F03-4",development_wells=["F02-1","F03-2","F06-1"],
       training_windows=41,validation_windows=8,testing_windows=49,
       embargo=3,observed_fraction=.10,scored="hidden coordinates only",
       masks="Mulberry32 fixed by seed, rho and mode",
       noise="Box-Muller Gaussian; shared across methods",generator="original csgm_m2_module",
       generator_epochs=cfg.csgm_ae_epochs,generator_latent_dim=16,
       optimizer_iters=cfg.csgm_iters,restarts=cfg.csgm_restarts,
       lambda_candidates=LAM,lambda_selected_on="validation hidden coordinates",
       note="One held-out well; overlapping test windows are not independent samples."),
       indent=2)+"\n")
    print("DONE",len(rows),"cells",round(time.time()-t0,1),"seconds",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=Path("outputs/f3_gap_clp_audit"))
    p.add_argument("--quick",action="store_true")
    a=p.parse_args();run(a.out,a.quick)

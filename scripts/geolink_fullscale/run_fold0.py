#!/usr/bin/env python3
"""Full-scale, well-disjoint, fold-0 Geolink DTC reconstruction audit.
Uses original CLP generator/prior/recovery implementation csgm_m2_module.py.
All model choices use development wells only. No access to held-out DTC to
choose gaps, models, lambda, GP parameters or stopping settings.
"""
from __future__ import annotations
import argparse,csv,io,json,math,os,sys,tarfile,time,hashlib
from pathlib import Path
from types import SimpleNamespace
from collections import Counter,defaultdict
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from csgm_m2_module import RidgePrior,train_ae_generator,encode_y,decode_latent_prior,csgm_recover_with_prior

L=256
SEEDS=(7,23,41)
OBS_COUNTS=(32,64)
LAM_GRID=(.0001,.0003,.001,.003,.01,.03,.1)
GP_LENGTHS=(8.,32.,64.,128.)
GP_REGULARIZERS=(.01,.1,1.)
ARCHIVE_SHA256="67a659043770fab643d60d4d8fd674da50d178797f49e664d8c9abdc532a7aa6"

def csv_write(path,rows):
    if not rows:return
    with open(path,"w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)

def full_arrays(path):
    sha=hashlib.sha256()
    with open(path,"rb") as f:
        for buf in iter(lambda:f.read(1<<20),b""):sha.update(buf)
    assert sha.hexdigest()==ARCHIVE_SHA256,("Archive checksum mismatch",sha.hexdigest())
    print("VERIFIED_ARCHIVE",sha.hexdigest(),flush=True)
    with tarfile.open(path,"r:gz") as tf:
        ms={Path(m.name).name:m for m in tf.getmembers() if m.isfile() and
            "geolink_fold_0_well_log_" in Path(m.name).name}
        def load(name):
            key="geolink_fold_0_well_log_"+name
            assert key in ms,key
            contents=tf.extractfile(ms[key]).read()
            return np.load(io.BytesIO(contents),allow_pickle=False) if name.endswith(".npy") else json.loads(contents)
        train=load("sliced_train.npy")
        test=load("sliced_val.npy")
        mt=load("slices_meta_train.json")
        ms=load("slices_meta_val.json")
    assert train.shape==(3904,256,4),train.shape
    assert test.shape==(1019,256,4),test.shape
    assert np.isfinite(train).all() and np.isfinite(test).all()
    wt=np.array([str(q[0]) for q in mt])
    we=np.array([str(q[0]) for q in ms])
    assert len(wt)==len(train) and len(we)==len(test)
    assert not (set(wt)&set(we)), "train-test well leakage!"
    return train,test,wt,we,mt,ms

def split_dev_wells(wt,seed=20261021):
    unique=np.unique(wt).copy()
    rng=np.random.default_rng(seed);rng.shuffle(unique)
    val=set(unique[:max(1,int(round(len(unique)*.2)))])
    va=np.array([str(x) in val for x in wt])
    assert not (set(wt[~va])&set(wt[va]))
    return ~va,va

def aux_features(A,mu=None,sd=None):
    U=A[:,:,:3].transpose(0,2,1)
    if mu is None:mu=U.mean((0,2));sd=U.std((0,2));sd=np.maximum(sd,1e-8)
    X=((U-mu[None,:,None])/sd[None,:,None]).reshape(len(U),-1)
    return X.astype(np.float64),mu,sd

def obs_positions(n,seed):
    rng=np.random.default_rng(314159+seed*103+n*37)
    start=int(rng.integers(0,L-n+1))
    return np.arange(start,start+n,dtype=int)

def predicted_baselines(Y,obs,positions,direct=None,extra=None):
    ymean=obs.mean(axis=1)[:,None]
    constant=np.broadcast_to(ymean,(len(obs),L)).copy()
    interp=np.stack([np.interp(np.arange(L),positions,obsrow) for obsrow in obs])
    candidates={"constant":constant,"interpolation":interp}
    if direct is not None:candidates["ridge_direct"]=direct
    if extra is not None:candidates["extra_trees_direct"]=extra
    return candidates

def gp_matrix(ids,length,ridge):
    distances=(ids[:,None]-ids[None,:])/length
    km=np.exp(-0.5*distances*distances)+ridge*np.eye(len(ids))
    kx=np.exp(-.5*((np.arange(L)[:,None]-ids[None,:])/length)**2)
    return np.linalg.solve(km,kx.T).T

def gp_adjust(obs,ids,kmat,base):
    if base is None:base=np.broadcast_to(obs.mean(1,keepdims=True),(len(obs),L))
    return base+(obs-base[:,ids])@kmat.T

def hidden_stats(ytrue,ypred,hid):
    err=ypred[:,hid]-ytrue[:,hid]
    mse=float(np.mean(err**2))
    variance=float(np.var(ytrue[:,hid]))
    return {"rmse":math.sqrt(mse),"mae":float(np.mean(np.abs(err))),
            "r2":1-mse/variance if variance>1e-12 else float("nan")}

def choose_gp(yval,obs,ids,hidden,base=None):
    winner=None
    for length in GP_LENGTHS:
        for reg in GP_REGULARIZERS:
            matrix=gp_matrix(ids,length,reg)
            z=gp_adjust(obs,ids,matrix,base)
            rmse=hidden_stats(yval,z,hidden)["rmse"]
            if winner is None or rmse<winner[0]:
                winner=(rmse,length,reg,matrix)
    return winner

def per_well_records(seed,m,well_ids,true,hidden,predictions,selected):
    rows=[]
    for well in sorted(set(well_ids)):
        ix=np.flatnonzero(well_ids==well)
        record={"seed":seed,"n_observations":m,"well_id":well,"n_windows":len(ix),
                "selected_simple":selected}
        for name,p in predictions.items():
            for metric,value in hidden_stats(true[ix],p[ix],hidden).items():
                record[f"{name}_{metric}"]=value
        rows.append(record)
    return rows

def fit_simple_predictors(Xtrain,Ytrain,Xval,Xtest,seed):
    direct=Ridge(alpha=100.,solver="cholesky").fit(Xtrain,Ytrain)
    valridge=direct.predict(Xval)
    testridge=direct.predict(Xtest)
    tree=ExtraTreesRegressor(n_estimators=64,max_depth=18,min_samples_leaf=3,
                             max_features=.5,n_jobs=2,random_state=seed)
    tree.fit(Xtrain,Ytrain)
    return valridge,testridge,tree.predict(Xval),tree.predict(Xtest)

def run(datafile,out,quick=False,seed_filter=None):
    os.environ["OMP_NUM_THREADS"]="1"
    os.environ["MKL_NUM_THREADS"]="1"
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=True)
    start=time.time()
    dev,ext,wdev,wext,mdev,mext=full_arrays(datafile)
    it,iv=split_dev_wells(wdev)
    train,val=dev[it],dev[iv]
    wtrain,wval=wdev[it],wdev[iv]
    assert set(wtrain).isdisjoint(set(wval)) and set(wtrain).isdisjoint(set(wext))
    assert set(wval).isdisjoint(set(wext))
    Xtr,mu,sd=aux_features(train)
    Xv,*_=aux_features(val,mu,sd)
    Xt,*_=aux_features(ext,mu,sd)
    Ytr=train[:,:,3].copy()
    Yv=val[:,:,3].copy()
    Yt=ext[:,:,3].copy()
    manifest={"archive_sha256":ARCHIVE_SHA256,
      "dataset":"GEOLINK processed original Zenodo record 10987946",
      "fold":0,"official_dev_windows":len(dev),"official_outer_test_windows":len(ext),
      "train_windows":len(train),"val_windows":len(val),"test_windows":len(ext),
      "train_wells":len(set(wtrain)),"val_wells":len(set(wval)),"test_wells":len(set(wext)),
      "train_test_well_overlap":0,"val_test_well_overlap":0,"validation_group_seed":20261021,
      "aux_channels":"first three source channels (GR,RHOB,NPHI)","target":"fourth source channel DTC",
      "window_len":L,"support_n":list(OBS_COUNTS),
      "support_policy":"fixed consecutive mask positions selected from seed and n_observations, no target dependence",
      "primary_metric":"hidden-coordinate RMSE; secondary hidden MAE and R2",
      "prior":"canonical RidgePrior (alpha=1); original autoencoder TinyAE",
      "lambda_grid":list(LAM_GRID),"selection":"internal well-disjoint validation only",
      "source_preprocessing_caveat":"published benchmark preprocessing, potentially global; no new cross-well scaling leakage",
      "confirmatory_caveat":"target regime chosen during earlier Geolink explorations; independent new dataset needed for final claim",
      "torch":torch.__version__,"quick":bool(quick)}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("AUDIT_SPLITS",json.dumps(manifest),flush=True)
    ys=StandardScaler().fit(Ytr)
    yn=ys.transform(Ytr)
    rows=[]; wellrows=[]; valrows=[]
    seeds=(SEEDS if seed_filter is None else tuple(seed_filter))
    if quick:seeds=(seeds[0],)
    cfg=SimpleNamespace(csgm_latent_dim=16,csgm_hidden_dim=128,
                         csgm_batch_size=64,csgm_ae_lr=.001,csgm_weight_decay=1e-5,
                         csgm_ae_epochs=12 if quick else 200,
                         csgm_iters=25 if quick else 400,csgm_opt_lr=.05,
                         csgm_restarts=1 if quick else 3)
    for seed in seeds:
        t0=time.time()
        ae=train_ae_generator(yn,cfg,seed,"cpu")
        for p in ae.parameters():p.requires_grad_(False)
        ztr=encode_y(ae,yn,"cpu")
        prior=RidgePrior(alpha=1.,random_state=seed+7000).fit(Xtr,ztr)
        zv,zt=prior.predict(Xv),prior.predict(Xt)
        priorval=decode_latent_prior(ae,zv,ys.mean_,ys.scale_,"cpu")
        priortest=decode_latent_prior(ae,zt,ys.mean_,ys.scale_,"cpu")
        print("MODEL_SEED",seed,"AE train seconds",round(time.time()-t0,1),flush=True)
        vdir,tdir,vext,text=fit_simple_predictors(Xtr,Ytr,Xv,Xt,seed)
        for m in OBS_COUNTS:
            ids=obs_positions(m,seed)
            hid=np.setdiff1d(np.arange(L),ids)
            bv=Yv[:,ids].copy()
            bt=Yt[:,ids].copy()
            simple_val=predicted_baselines(Yv,bv,ids,vdir,vext)
            simple_test=predicted_baselines(Yt,bt,ids,tdir,text)
            gp=choose_gp(Yv,bv,ids,hid,None)
            simple_val["GP_smooth"]=gp_adjust(bv,ids,gp[3],None)
            simple_test["GP_smooth"]=gp_adjust(bt,ids,gp[3],None)
            gp_res=choose_gp(Yv,bv,ids,hid,vdir)
            simple_val["ridge_GP_residual"]=gp_adjust(bv,ids,gp_res[3],vdir)
            simple_test["ridge_GP_residual"]=gp_adjust(bt,ids,gp_res[3],tdir)
            val_rank={key:hidden_stats(Yv,P,hid)["rmse"] for key,P in simple_val.items()}
            selected=min(val_rank,key=lambda k:(val_rank[k],k))
            M=np.eye(L,dtype=float)[ids]
            lam_scores=[]
            for lam in LAM_GRID:
                pred_val=csgm_recover_with_prior(
                    ae,M,bv,zv,ys.mean_,ys.scale_,lam,
                    cfg.csgm_iters,cfg.csgm_opt_lr,cfg.csgm_restarts,"cpu",seed)
                sc=hidden_stats(Yv,pred_val,hid)["rmse"]
                lam_scores.append((sc,lam))
                valrows.append({"seed":seed,"n_observations":m,"method":"CLP",
                                "lambda":lam,"validation_rmse":sc,
                                "n_val_windows":len(val)})
            lam_selected=min(lam_scores,key=lambda a:(a[0],a[1]))[1]
            clp=csgm_recover_with_prior(
                ae,M,bt,zt,ys.mean_,ys.scale_,lam_selected,
                cfg.csgm_iters,cfg.csgm_opt_lr,cfg.csgm_restarts,"cpu",seed)
            methods={"CLP":clp,"prior_only":priortest,**simple_test}
            result={"seed":seed,"observed_samples":m,"support_start":int(ids[0]),
                    "support_end":int(ids[-1]),"lambda":lam_selected,
                    "dev_validation_clp_rmse":min(sc for sc,la in lam_scores),
                    "selected_simple":selected,"selected_val_rmse":val_rank[selected],
                    "train_windows":len(train),"val_windows":len(val),
                    "outer_test_windows":len(Yt),"n_test_wells":len(set(wext))}
            for name,P in methods.items():
                for k,v in hidden_stats(Yt,P,hid).items():result[f"{name}_{k}"]=v
            result["clp_better_selected_simple"]=result["CLP_rmse"]<result[selected+"_rmse"]
            result["clp_better_interpolation"]=result["CLP_rmse"]<result["interpolation_rmse"]
            result["clp_better_observed_mean"]=result["CLP_rmse"]<result["constant_rmse"]
            rows.append(result)
            wellrows.extend(per_well_records(seed,m,wext,Yt,hid,methods,selected))
            csv_write(out/"metrics_by_seed.csv",rows)
            csv_write(out/"metrics_by_well.csv",wellrows)
            csv_write(out/"lambda_val.csv",valrows)
            print("RESULT_CELL",json.dumps(result,sort_keys=True),flush=True)
        print("SEED_DONE",seed,round(time.time()-t0,1),flush=True)
    # well-wise paired bootstrap; same well is a statistical cluster
    comparisons=[]
    for m in OBS_COUNTS:
        for name in ("interpolation","constant","GP_smooth","ridge_direct","extra_trees_direct","ridge_GP_residual"):
            group=[q for q in wellrows if q["n_observations"]==m]
            wells=sorted(set(q["well_id"] for q in group))
            diffs=[]
            for well in wells:
                per=[q for q in group if q["well_id"]==well]
                diffs.append(np.mean([q[name+"_rmse"]-q["CLP_rmse"] for q in per]))
            d=np.array(diffs)
            rng=np.random.default_rng(919911+m+len(name))
            sample=rng.choice(d,size=(2000,len(d)),replace=True).mean(axis=1)
            comparisons.append(dict(n_observations=m,method=name,n_wells=len(wells),
                                    mean_delta=float(d.mean()),
                                    ci95_low=float(np.quantile(sample,.025)),
                                    ci95_high=float(np.quantile(sample,.975)),
                                    win_wells=int(np.sum(d>0)),mean_clp_minus_baseline=float(-d.mean())))
    csv_write(out/"bootstrap_wells.csv",comparisons)
    manifest["elapsed_seconds"]=time.time()-start
    manifest["n_cells"]=len(rows)
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("FINAL_SUMMARY",json.dumps({"cells":rows,"bootstrap":comparisons}),flush=True)
    print("DONE",len(rows),"configurations",round(time.time()-start,1),"s",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--archive",type=Path,default=Path("/tmp/geolink/public-processed-datasets.tar.gz"))
    p.add_argument("--out",type=Path,default=Path("outputs/geolink_full_fold0"))
    p.add_argument("--quick",action="store_true")
    p.add_argument("--seed",type=int,action="append")
    args=p.parse_args()
    run(args.archive,args.out,quick=args.quick,seed_filter=args.seed)

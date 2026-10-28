#!/usr/bin/env python3
"""Independent in-chat GeoLink experiment, NOT a reproduction of CS-ML F3.
Source: user-provided public-processed-datasets.tar(2).gz, GeoLink fold 0.
Use 56 train windows from distinct training wells, validation/test on disjoint wells.
Target is dataset channel 0, auxiliaries channels 1 and 2 (names not provided).
Target windows downsampled to length 64. Observe 13 indices per window.
Compare random observation positions versus central contiguous 32-index gap.
"""
import os,json,time,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'pgeoprj/godeep/dados/l2_datasets/imputation-processed-datasets'
OUT=ROOT/'geo_gap_pilot';OUT.mkdir(exist_ok=True)
torch.set_num_threads(1)
SEEDS=[7,23,41]; NTRAIN=56; L=64; M=13; GAP=np.arange(16,48)
LAMBDAS=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
EPOCHS=180; NITERS=220; RESTARTS=2
class AE(nn.Module):
 def __init__(self):
  super().__init__();self.enc=nn.Sequential(nn.Linear(L,64),nn.ReLU(),nn.Linear(64,16));self.dec=nn.Sequential(nn.Linear(16,64),nn.ReLU(),nn.Linear(64,L))
 def forward(self,y):return self.dec(self.enc(y))

def fit_ae(y_n,seed):
 torch.manual_seed(seed);net=AE();opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-5)
 yt=torch.tensor(y_n,dtype=torch.float32)
 for ep in range(EPOCHS):
  for ix in torch.randperm(len(yt)).split(32):
   loss=((net(yt[ix])-yt[ix])**2).mean();opt.zero_grad();loss.backward();opt.step()
 return net

def recover(ae, ymean,yscale, z0, obs,b,lam,seed):
 """CS-ML-style batch objective with shared restarts, best batch-level objective."""
 obs=np.asarray(obs);zt=torch.tensor(z0,dtype=torch.float32)
 bt=torch.tensor((b-ymean[obs][None,:])/yscale[obs][None,:],dtype=torch.float32)
 zscale=torch.tensor(yscale[obs],dtype=torch.float32)
 noise=torch.Generator().manual_seed(seed+190)
 best=np.inf;ypred=None
 for r in range(RESTARTS):
  zi=zt.clone() if r==0 else zt+0.05*torch.randn(zt.shape,generator=noise)
  zi=zi.detach().requires_grad_();opt=torch.optim.Adam([zi],lr=0.05)
  for k in range(NITERS):
   yp=ae.dec(zi)
   data=((yp[:,obs]-bt)*zscale).square().sum()/len(zi)
   prior=lam*(zi-zt).square().sum()/len(zi)
   loss=data+prior;opt.zero_grad();loss.backward();opt.step()
  with torch.no_grad():
   yp=ae.dec(zi)
   loss=(((yp[:,obs]-bt)*zscale).square().sum()+lam*(zi-zt).square().sum())/len(zi)
   if loss.item()<best:
    best=loss.item();ypred=yp.numpy().copy()
 return ypred*yscale[None,:]+ymean[None,:]

def rms(y,yhat,mask):return float(np.sqrt(np.mean((y[:,mask]-yhat[:,mask])**2)))
def interp(y,obs,b):
 yp=np.empty_like(y)
 x=np.arange(L)
 for i in range(len(y)):
  yp[i]=np.interp(x,obs,b[i],left=b[i,0],right=b[i,-1])
 return yp

def load_fold():
 tr=np.load(DATA/'geolink_fold_0_well_log_sliced_train.npy',mmap_mode='r')
 va=np.load(DATA/'geolink_fold_0_well_log_sliced_val.npy',mmap_mode='r')
 trmeta=json.load(open(DATA/'geolink_fold_0_well_log_slices_meta_train.json'))
 vameta=json.load(open(DATA/'geolink_fold_0_well_log_slices_meta_val.json'))
 assert len(tr)==len(trmeta) and len(va)==len(vameta)
 assert not {v[0] for v in trmeta}&{v[0] for v in vameta}
 rng=np.random.default_rng(10203)
 # One window per unique training well; fixed training examples across 3 seeds.
 wells=np.unique([m[0] for m in trmeta]);rng.shuffle(wells)
 chosen=np.array([rng.choice([i for i,row in enumerate(trmeta) if row[0]==well]) for well in wells[:NTRAIN]])
 # Disjoint held-out wells for lambda validation and model testing.
 vwells=np.unique([m[0] for m in vameta]);rng.shuffle(vwells)
 valwell=set(vwells[:10]);testwell=set(vwells[10:22])
 def sample_ix(group,quota):
  out=[]
  for w in sorted(group):
   ix=[i for i,row in enumerate(vameta) if row[0]==w]
   out+=list(rng.choice(ix,min(len(ix),quota),replace=False))
  return np.asarray(out,dtype=int)
 vi=sample_ix(valwell,6);ti=sample_ix(testwell,6)
 def extract(a,idx):
  chunk=np.asarray(a[idx][:,::4,:],dtype=np.float64)
  return chunk[:,:,0],chunk[:,:,1:3].reshape(len(idx),-1)
 yt,xt=extract(tr,chosen);yv,xv=extract(va,vi);ye,xe=extract(va,ti)
 return (yt,xt),(yv,xv),(ye,xe), dict(ntrain=len(chosen),nval=len(vi),ntest=len(ti),train_wells=sorted(wells[:NTRAIN].tolist()),val_wells=sorted(valwell),test_wells=sorted(testwell))

def mask_for(seed,kind):
 rng=np.random.default_rng(3000+seed)
 # Both masks sample exactly 13 positions; gap has zero observations in center 16:48.
 if kind=='random':return np.sort(rng.choice(L,M,replace=False))
 left=np.sort(rng.choice(np.arange(0,16),6,replace=False));right=np.sort(rng.choice(np.arange(48,64),M-6,replace=False))
 return np.r_[left,right]

(yt,xt),(yv,xv),(ye,xe),meta=load_fold()
y_scaler=StandardScaler().fit(yt);xn_scaler=StandardScaler().fit(xt)
ymean=y_scaler.mean_;yscale=y_scaler.scale_
assert not np.any(yscale==0)
xf=xn_scaler.transform(xt);vf=xn_scaler.transform(xv);ef=xn_scaler.transform(xe)
# Simple direct auxiliary-only baseline, fixed regularization and training set.
aux=Ridge(alpha=10).fit(xf,yt);ridge_pred=aux.predict(ef)
rows=[];pred_rows=[];lambda_rows=[];start=time.time()
for seed in SEEDS:
 print('seed',seed,'start',flush=True)
 ae=fit_ae(y_scaler.transform(yt),seed)
 with torch.no_grad():zt=ae.enc(torch.tensor(y_scaler.transform(yt),dtype=torch.float32)).numpy()
 zsc=StandardScaler().fit(zt)
 ridge_lat=Ridge(alpha=1.0).fit(xf,zsc.transform(zt))
 zv=zsc.inverse_transform(ridge_lat.predict(vf));ze=zsc.inverse_transform(ridge_lat.predict(ef))
 for kind in ('random','contiguous_gap'):
  obs=mask_for(seed,'random' if kind=='random' else 'gap');hold=np.array([i for i in range(L) if i not in obs]);
  if kind=='contiguous_gap':assert not set(obs)&set(GAP)
  assert len(obs)==M and len(hold)==L-M
  rnga=np.random.default_rng(5900+seed)
  noise_val=rnga.normal(0,0.01,size=yv.shape);noise_test=rnga.normal(0,0.01,size=ye.shape)
  bval=yv[:,obs]+noise_val[:,obs];btest=ye[:,obs]+noise_test[:,obs]
  # Lambda chosen only on validation held-out locations.
  bestval=np.inf;lam_star=None
  for lam in LAMBDAS:
   yh=recover(ae,ymean,yscale,zv,obs,bval,lam,seed)
   valrmse=rms(yv,yh,hold)
   lambda_rows.append(dict(seed=seed,mask=kind,lambda_value=lam,validation_rmse=valrmse))
   if valrmse<bestval:bestval=valrmse;lam_star=lam
  clp=recover(ae,ymean,yscale,ze,obs,btest,lam_star,seed)
  free=recover(ae,ymean,yscale,np.zeros_like(ze),obs,btest,0,seed)
  intr=interp(ye,obs,btest)
  prior=ae.dec(torch.tensor(ze,dtype=torch.float32)).detach().numpy()*yscale[None,:]+ymean[None,:]
  preds={'CLP':clp,'CSGM_sem_prior':free,'interpolacao':intr,'ridge_aux':ridge_pred,'prior_only':prior}
  for label,pr in preds.items():
   rows.append(dict(seed=seed,mask=kind,method=label,rmse_holdout=rms(ye,pr,hold),rmse_central_heldout=rms(ye,pr,np.intersect1d(hold,GAP)),lambda_selected=lam_star if label=='CLP' else np.nan,n_train=len(yt),n_test=len(ye),n_holdout=len(hold)))
  # Selected fixed first heldout test row for plotting, independent of ranking.
  idx=0
  for j in range(L):
   d=dict(seed=seed,mask=kind,index=j,truth=ye[idx,j],observed=int(j in obs))
   for label,pr in preds.items():d[label]=pr[idx,j]
   pred_rows.append(d)
 print('seed',seed,'elapsed',round(time.time()-start,1),flush=True)

results=pd.DataFrame(rows);results.to_csv(OUT/'results.csv',index=False)
pd.DataFrame(pred_rows).to_csv(OUT/'first_heldout_window.csv',index=False)
pd.DataFrame(lambda_rows).to_csv(OUT/'lambda_validation.csv',index=False)
# Group summaries and paired gains: no window treated as independent seed.
summary=results.groupby(['mask','method'])[['rmse_holdout','rmse_central_heldout']].agg(['mean','std']).reset_index()
summary.to_csv(OUT/'summary.csv',index=False)
for kind in ['random','contiguous_gap']:
 sub=results[results['mask']==kind]
 print('\n'+kind)
 print(sub.pivot(index='seed',columns='method',values='rmse_holdout').round(4).to_string())
 print('mean heldout',sub.groupby('method').rmse_holdout.mean().round(4).to_dict())
 print('mean central heldout',sub.groupby('method').rmse_central_heldout.mean().round(4).to_dict())

# Plot mean by mask grouped, with all three seed values to show variability.
fig,axes=plt.subplots(1,2,figsize=(13,4.8),sharey=True)
methods=['CLP','CSGM_sem_prior','interpolacao','ridge_aux','prior_only']
for ax,kind in zip(axes,['random','contiguous_gap']):
 for i,method in enumerate(methods):
  sub=results[(results['mask']==kind)&(results['method']==method)]
  ax.bar(i,sub.rmse_holdout.mean(),alpha=0.65)
  ax.scatter([i-0.12,i,i+0.12],sub.sort_values('seed').rmse_holdout.values,s=18,zorder=4)
 ax.set_xticks(range(len(methods)));ax.set_xticklabels(['CLP','CSGM livre','Interpolação','Ridge aux','Prior-only'],rotation=23,ha='right')
 ax.set_title('Amostragem aleatória' if kind=='random' else 'Lacuna central contígua (32/64)')
 ax.grid(axis='y',alpha=0.25)
axes[0].set_ylabel('RMSE em 51 posições não observadas (escala normalizada)')
fig.suptitle('GeoLink — piloto independente, 3 sementes, 56 exemplos de treino')
fig.tight_layout();fig.savefig(OUT/'rmse_comparacao.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
for ax,kind in zip(axes,['random','contiguous_gap']):
 sub=pd.DataFrame(pred_rows);sub=sub[(sub.seed==7)&(sub['mask']==kind)].sort_values('index')
 for method in ['truth','CLP','CSGM_sem_prior','interpolacao']:
  ax.plot(sub['index'],sub[method],label=method,linewidth=1.5 if method=='truth' else 1.1)
 obs=sub[sub.observed==1];ax.scatter(obs['index'],obs['truth'],c='black',s=20,zorder=5)
 if kind=='contiguous_gap':ax.axvspan(16,47,alpha=0.15,color='gray',label='Lacuna sem medições')
 ax.set_title('Semente 7 — '+kind);ax.grid(alpha=0.25)
axes[-1].set_xlabel('Posição na janela (256 → 64 por subamostragem)')
axes[0].set_ylabel('Canal 0 padronizado');axes[1].set_ylabel('Canal 0 padronizado')
axes[0].legend(loc='upper right',ncol=5,fontsize=8);fig.tight_layout();fig.savefig(OUT/'exemplo_perfil.png',dpi=160);plt.close(fig)

checks=dict(train_val_wells_disjoint=not bool(set(meta['train_wells'])&set(meta['val_wells'])),train_test_wells_disjoint=not bool(set(meta['train_wells'])&set(meta['test_wells'])),val_test_wells_disjoint=not bool(set(meta['val_wells'])&set(meta['test_wells'])),n_train=len(yt),n_validation=len(yv),n_test=len(ye),all_finite=bool(np.isfinite(results.rmse_holdout).all()),gap_zero_observations=all(not set(mask_for(s,'gap'))&set(GAP) for s in SEEDS),all_mask_counts=all(len(mask_for(s,k))==M for s in SEEDS for k in ('random','gap')))
assert all(v for k,v in checks.items() if isinstance(v,bool))
config=dict(dataset='GeoLink fold 0 public-processed-datasets.tar(2).gz',dataset_origin='user-provided archive',target='channel index 0; physical meaning unavailable in metadata',aux_channels=[1,2],train_budget=56,seeds=SEEDS,window_length=L,original_window_length=256,subsampling_stride=4,observations_per_window=M,contiguous_gap_positions=[16,47],gap_observation_rule='6 positions from 0:16 plus 7 from 48:64',random_rule='13 positions uniform without replacement',ae_arch='64-64-16 encoder, 16-64-64 decoder, ReLU',ae_epochs=EPOCHS,latent_iters=NITERS,latent_restarts=RESTARTS,lambda_grid=LAMBDAS,lambda_selection='validation holdout RMSE',n_train=len(yt),n_validation=len(yv),n_test=len(ye),noise_std=0.01,comparison='independent GeoLink proxy, NOT original CS-ML F3 run',split_wells={'train':meta['train_wells'],'validation':meta['val_wells'],'test':meta['test_wells']})
(OUT/'config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False));(OUT/'checks.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False))
print('DONE seconds',round(time.time()-start,1),'at',OUT)

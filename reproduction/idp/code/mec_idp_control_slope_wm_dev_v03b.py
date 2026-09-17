from __future__ import annotations
import argparse, json, math, pickle
from pathlib import Path
import numpy as np

def wrap(x):
    return (x+np.pi)%(2*np.pi)-np.pi

class IDP:
    def __init__(self,seed=0):
        import gymnasium as gym, mujoco
        self.mujoco=mujoco
        self.env=gym.make('InvertedDoublePendulum-v5')
        self.env.reset(seed=seed)
        self.model=self.env.unwrapped.model
        self.data=self.env.unwrapped.data
        slide=int(mujoco.mjtJoint.mjJNT_SLIDE)
        hinge=int(mujoco.mjtJoint.mjJNT_HINGE)
        self.slide=None; hs=[]
        for jid,t in enumerate(self.model.jnt_type):
            if int(t)==slide and self.slide is None: self.slide=jid
            if int(t)==hinge: hs.append(jid)
        assert self.slide is not None and len(hs)>=2
        self.hs=hs[:2]
        self.cart_q=int(self.model.jnt_qposadr[self.slide])
        self.cart_d=int(self.model.jnt_dofadr[self.slide])
        self.hq=[int(self.model.jnt_qposadr[j]) for j in self.hs]
        self.hd=[int(self.model.jnt_dofadr[j]) for j in self.hs]
        self.act=None
        for aid in range(self.model.nu):
            if int(self.model.actuator_trnid[aid,0])==self.slide:
                self.act=aid; break
        assert self.act is not None
        self.ctrlrange=np.asarray(self.model.actuator_ctrlrange[self.act],float)
        self.qpos0=self.model.qpos0.copy()

    def set_state(self,cart,hp,cv,hv):
        self.data.qpos[:]=self.qpos0; self.data.qvel[:]=0
        self.data.qpos[self.cart_q]=self.qpos0[self.cart_q]+float(cart)
        for q,v in zip(self.hq,hp):
            self.data.qpos[q]=self.qpos0[q]+float(v)
        self.data.qvel[self.cart_d]=float(cv)
        for d,v in zip(self.hd,hv):
            self.data.qvel[d]=float(v)
        self.data.ctrl[:]=0
        self.mujoco.mj_forward(self.model,self.data)

    def obs(self):
        return np.array([
            self.data.qpos[self.cart_q]-self.qpos0[self.cart_q],
            wrap(self.data.qpos[self.hq[0]]-self.qpos0[self.hq[0]]),
            wrap(self.data.qpos[self.hq[1]]-self.qpos0[self.hq[1]]),
            self.data.qvel[self.cart_d],self.data.qvel[self.hd[0]],self.data.qvel[self.hd[1]]
        ],float)

    def step_actual(self,actual):
        self.data.ctrl[:]=0
        self.data.ctrl[self.act]=float(actual)
        self.mujoco.mj_step(self.model,self.data)

    def close(self): self.env.close()

def feature(x,u_commanded):
    x=np.asarray(x,float)
    return np.array([x[0],math.sin(x[1]),math.cos(x[1]),math.sin(x[2]),math.cos(x[2]),
                     x[3],x[4],x[5],float(u_commanded)],np.float32)

def target_delta(x,y):
    d=np.asarray(y,float)-np.asarray(x,float)
    d[1]=wrap(d[1]); d[2]=wrap(d[2])
    return d.astype(np.float32)

class WM:
    def __init__(self,indim,outdim):
        import torch, torch.nn as nn
        self.torch=torch
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net=nn.Sequential(
            nn.Linear(indim,128),nn.SiLU(),
            nn.Linear(128,128),nn.SiLU(),
            nn.Linear(128,128),nn.SiLU(),
            nn.Linear(128,outdim)
        ).float().to(self.device)

    def load(self,b):
        self.net.load_state_dict(b['state_dict'])
        self.fmu=np.asarray(b['fmu'],np.float32); self.fsd=np.asarray(b['fsd'],np.float32)
        self.rmu=np.asarray(b['rmu'],np.float32); self.rsd=np.asarray(b['rsd'],np.float32)
        self.net.eval(); return self

    def pred(self,F):
        torch=self.torch
        F=np.asarray(F,np.float32)
        with torch.no_grad():
            X=torch.tensor((F-self.fmu)/self.fsd,dtype=torch.float32,device=self.device)
            p=self.net(X).cpu().numpy()
        return p*self.rsd+self.rmu

def nominal_stats(wm,Fc,Rc):
    e=Rc-wm.pred(Fc)
    return e.mean(0),e.std(0)+1e-6

def generate_pairs(cfg):
    d=cfg['fresh_pairs']; rng=np.random.default_rng(d['pair_root'])
    groups=[]; invalid=0
    for i in range(d['pairs']):
        sign=float(rng.choice([-1.,1.]))
        u1=sign*float(rng.uniform(*d['u1_abs_range']))
        du=sign*float(rng.uniform(*d['delta_u_abs_range']))
        u2=u1+du
        gain=float(rng.uniform(*d['gain_multiplier_range']))
        bias=(gain-1.0)*u1
        cart=float(rng.uniform(*d['cart_position_range']))
        cv=float(rng.uniform(*d['cart_velocity_range']))
        hp=rng.uniform(d['hinge_pose_range'][0],d['hinge_pose_range'][1],2)
        hv=rng.uniform(d['hinge_velocity_range'][0],d['hinge_velocity_range'][1],2)

        A=IDP(d['pair_root']+i*2+1); B=IDP(d['pair_root']+i*2+2)
        scale=max(abs(A.ctrlrange[0]),abs(A.ctrlrange[1]))
        vals=[gain*u1,u1+bias,gain*u2,u2+bias]
        if max(abs(v)/scale for v in vals)>d['max_fraction_of_actuator_range_used']+1e-12:
            invalid+=1; A.close(); B.close(); continue

        pair=[]
        for lab,kind,env in [(1,'GAIN',A),(0,'BIAS',B)]:
            FF=[]; RR=[]
            for u in [u1,u2]:
                env.set_state(cart,hp,cv,hv)
                x=env.obs()
                actual=(gain*u) if kind=='GAIN' else (u+bias)
                env.step_actual(actual)
                y=env.obs()
                FF.append(feature(x,u)); RR.append(target_delta(x,y))
            pair.append({'label':lab,'kind':kind,'F':np.asarray(FF,np.float32),
                         'R':np.asarray(RR,np.float32),'du':du})
        groups.append(pair); A.close(); B.close()
    return groups,invalid

def fit_eval(Xtr,ytr,Xte,yte):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score,accuracy_score
    models={
        'RF':RandomForestClassifier(n_estimators=500,min_samples_leaf=3,max_features='sqrt',
                                    class_weight='balanced',random_state=3107,n_jobs=-1),
        'SVM':make_pipeline(StandardScaler(),SVC(C=10.0,kernel='rbf',gamma='scale',
                                                 class_weight='balanced',probability=True,random_state=3107))
    }
    out={}
    for name,m in models.items():
        m.fit(Xtr,ytr)
        p=m.predict_proba(Xte)[:,1]
        pred=m.predict(Xte)
        auc=float(roc_auc_score(yte,p))
        auc=max(auc,1-auc)
        out[name]={'auc':auc,'accuracy':float(accuracy_score(yte,pred))}
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    ap.add_argument('--v02dir',required=True)
    ap.add_argument('--outdir',required=True)
    a=ap.parse_args()
    cfg=json.loads(Path(a.config).read_text())
    V=Path(a.v02dir); O=Path(a.outdir); O.mkdir(parents=True,exist_ok=True)

    assert cfg['hidden_unseal_authorized'] is False
    forbidden=set(cfg['hidden_seeds_forbidden']['world_model_seeds'])|set(cfg['hidden_seeds_forbidden']['scenario_roots'])
    used=set(cfg['world_models']['development_seeds'])|{cfg['fresh_pairs']['pair_root'],cfg['fresh_pairs']['group_split_seed']}
    assert not (forbidden & used)

    import torch
    nz=np.load(V/'nominal_data.npz')
    Fc,Rc=nz['Fc'],nz['Rc']
    v02=json.loads((V/'MEC_CROSSFAMILY_IDP_WM_THRESHOLD_DEV_V0_2_RESULT.json').read_text())
    assert v02['summary']['quality_models_passed']==8
    assert v02['hidden_unsealed'] is False and v02['hidden_seeds_used'] is False

    cache=O/'fresh_control_slope_pairs.pkl'
    if cache.exists():
        groups,invalid=pickle.loads(cache.read_bytes())
    else:
        groups,invalid=generate_pairs(cfg)
        cache.write_bytes(pickle.dumps((groups,invalid)))

    rng=np.random.default_rng(cfg['fresh_pairs']['group_split_seed'])
    ids=np.arange(cfg['fresh_pairs']['pairs']); rng.shuffle(ids)
    train=set(map(int,ids[:cfg['fresh_pairs']['train_groups']]))
    test=set(map(int,ids[cfg['fresh_pairs']['train_groups']:]))

    records={}; direct_d1={}; direct_d2={}
    for seed in cfg['world_models']['development_seeds']:
        b=torch.load(V/f'wm_{seed}.pt',map_location='cpu',weights_only=False)
        wm=WM(Fc.shape[1],Rc.shape[1]).load(b)
        mu,sd=nominal_stats(wm,Fc,Rc)
        rows=[]; d1diff=[]; d2diff=[]
        for gi,pair in enumerate(groups):
            rr=[]
            for row in pair:
                e=np.asarray(row['R'],float)-wm.pred(row['F'])
                z=(e-mu)/sd
                d1=z[0].copy()
                slope=(z[1]-z[0])/float(row['du'])
                rr.append((row['label'],d1,slope,e))
                rows.append({'group':gi,'label':row['label'],'D1':d1,'D2':slope})
            d1diff.append(float(np.linalg.norm(rr[0][3][0]-rr[1][3][0])))
            d2diff.append(float(np.linalg.norm(rr[0][3][1]-rr[1][3][1])))
        records[seed]=rows; direct_d1[seed]=np.asarray(d1diff); direct_d2[seed]=np.asarray(d2diff)

    folds=[]
    for held in cfg['world_models']['development_seeds']:
        src=[s for s in cfg['world_models']['development_seeds'] if s!=held]
        fold={'heldout_wm':held}
        for key in ['D1','D2']:
            Xtr=[];ytr=[];Xte=[];yte=[]
            for s in src:
                for r in records[s]:
                    if r['group'] in train:
                        Xtr.append(r[key]); ytr.append(r['label'])
            for r in records[held]:
                if r['group'] in test:
                    Xte.append(r[key]); yte.append(r['label'])
            fold[key]=fit_eval(np.asarray(Xtr),np.asarray(ytr),np.asarray(Xte),np.asarray(yte))
        folds.append(fold)

    g=cfg['gates']
    d1p95=max(float(np.quantile(direct_d1[s],.95)) for s in direct_d1)
    d2p05=min(float(np.quantile(direct_d2[s],.05)) for s in direct_d2)
    d1best=[max(f['D1']['RF']['auc'],f['D1']['SVM']['auc']) for f in folds]
    d2worst_auc=[min(f['D2']['RF']['auc'],f['D2']['SVM']['auc']) for f in folds]
    d2worst_acc=[min(f['D2']['RF']['accuracy'],f['D2']['SVM']['accuracy']) for f in folds]

    gates={
        'inherited_quality':v02['summary']['quality_models_passed']==g['inherited_world_model_quality_models'],
        'invalid_pairs':invalid<=g['invalid_pairs_max'],
        'D1_direct_match':d1p95<=g['D1_direct_residual_pair_diff_p95_max'],
        'D1_auc_median':float(np.median(d1best))<=g['D1_adversarial_best_auc_median_max'],
        'D1_folds':sum(x<=g['D1_adversarial_best_auc_median_max'] for x in d1best)>=g['D1_fold_passes_min'],
        'D2_direct_separation':d2p05>=g['D2_direct_residual_pair_diff_p05_min'],
        'D2_auc_median':float(np.median(d2worst_auc))>=g['D2_worst_classifier_auc_median_min'],
        'D2_acc_median':float(np.median(d2worst_acc))>=g['D2_worst_classifier_accuracy_median_min'],
        'D2_folds':sum((a>=g['D2_worst_classifier_auc_median_min'] and
                        b>=g['D2_worst_classifier_accuracy_median_min'])
                       for a,b in zip(d2worst_auc,d2worst_acc))>=g['D2_fold_passes_min']
    }
    summary={
        'inherited_quality_models':v02['summary']['quality_models_passed'],
        'invalid_pairs':invalid,
        'D1_direct_residual_pair_diff_p95_worst_wm':d1p95,
        'D1_adversarial_best_auc_median':float(np.median(d1best)),
        'D1_fold_passes':int(sum(x<=g['D1_adversarial_best_auc_median_max'] for x in d1best)),
        'D2_direct_residual_pair_diff_p05_worst_wm':d2p05,
        'D2_worst_classifier_auc_median':float(np.median(d2worst_auc)),
        'D2_worst_classifier_accuracy_median':float(np.median(d2worst_acc)),
        'D2_fold_passes':int(sum((a>=g['D2_worst_classifier_auc_median_min'] and
                                 b>=g['D2_worst_classifier_accuracy_median_min'])
                                for a,b in zip(d2worst_auc,d2worst_acc))),
        'gates':gates
    }
    decision='CONTROL_SLOPE_WM_DEV_PASS' if all(gates.values()) else 'CONTROL_SLOPE_WM_DEV_FAIL'
    result={'protocol_id':cfg['protocol_id'],'decision':decision,'hidden_unsealed':False,
            'hidden_seeds_used':False,'world_models_retrained':False,
            'source_v02_decision':v02['decision'],'folds':folds,'summary':summary}
    (O/'MEC_IDP_CONTROL_SLOPE_WM_DEV_V0_3B_RESULT.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'decision':decision,'summary':summary},indent=2))

if __name__=='__main__':
    main()

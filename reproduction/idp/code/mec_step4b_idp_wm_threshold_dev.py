from __future__ import annotations
import argparse, json, math, os, random, hashlib, pickle, time
from pathlib import Path
import numpy as np

def save_json(p,obj):
    Path(p).write_text(json.dumps(obj,indent=2))

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1<<20),b''): h.update(c)
    return h.hexdigest()

def seed_all(seed):
    random.seed(seed); np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True,warn_only=True)

def wrap(x):
    return (x+np.pi)%(2*np.pi)-np.pi

class IDP:
    def __init__(self, seed=0):
        import gymnasium as gym, mujoco
        self.mujoco=mujoco
        self.env=gym.make('InvertedDoublePendulum-v5')
        self.env.reset(seed=seed)
        self.model=self.env.unwrapped.model; self.data=self.env.unwrapped.data
        self.base_qpos0=self.model.qpos0.copy()
        self.base_damp=self.model.dof_damping.copy()
        slide=int(mujoco.mjtJoint.mjJNT_SLIDE); hinge=int(mujoco.mjtJoint.mjJNT_HINGE)
        self.slide_jid=None; h=[]
        for jid,t in enumerate(self.model.jnt_type):
            if int(t)==slide and self.slide_jid is None: self.slide_jid=jid
            if int(t)==hinge: h.append(jid)
        assert self.slide_jid is not None and len(h)>=2
        self.hinge_jids=h[:2]
        self.cart_q=int(self.model.jnt_qposadr[self.slide_jid]); self.cart_d=int(self.model.jnt_dofadr[self.slide_jid])
        self.hinge_q=[int(self.model.jnt_qposadr[j]) for j in self.hinge_jids]
        self.hinge_d=[int(self.model.jnt_dofadr[j]) for j in self.hinge_jids]
        self.cart_act=None
        for aid in range(self.model.nu):
            if int(self.model.actuator_trnid[aid,0])==self.slide_jid:
                self.cart_act=aid; break
        assert self.cart_act is not None
        self.lo=np.asarray(self.env.action_space.low,float); self.hi=np.asarray(self.env.action_space.high,float)

    def restore(self):
        self.model.qpos0[:]=self.base_qpos0; self.model.dof_damping[:]=self.base_damp
        self.mujoco.mj_setConst(self.model,self.data)

    def set_state(self, cart, hp, cv, hv):
        self.data.qpos[:]=self.model.qpos0; self.data.qvel[:]=0
        self.data.qpos[self.cart_q]=self.base_qpos0[self.cart_q]+float(cart)
        for q,v in zip(self.hinge_q,hp): self.data.qpos[q]=self.base_qpos0[q]+float(v)
        self.data.qvel[self.cart_d]=float(cv)
        for d,v in zip(self.hinge_d,hv): self.data.qvel[d]=float(v)
        self.data.ctrl[:]=0; self.mujoco.mj_forward(self.model,self.data)

    def obs(self, cart_bias=0.0):
        return np.array([
            self.data.qpos[self.cart_q]-self.base_qpos0[self.cart_q]+float(cart_bias),
            wrap(self.data.qpos[self.hinge_q[0]]-self.base_qpos0[self.hinge_q[0]]),
            wrap(self.data.qpos[self.hinge_q[1]]-self.base_qpos0[self.hinge_q[1]]),
            self.data.qvel[self.cart_d], self.data.qvel[self.hinge_d[0]], self.data.qvel[self.hinge_d[1]]
        ],float)

    def step(self, commanded, hidden_cart_bias_ctrl=0.0):
        u=np.zeros(self.model.nu,float)
        u[:len(commanded)]=np.asarray(commanded,float)
        u[self.cart_act]+=float(hidden_cart_bias_ctrl)
        self.data.ctrl[:]=u; self.mujoco.mj_step(self.model,self.data)

    def cart_tau_per_unit(self):
        oq=self.data.qpos.copy(); ov=self.data.qvel.copy(); oc=self.data.ctrl.copy()
        self.data.ctrl[:]=0; self.data.ctrl[self.cart_act]=1.0
        self.mujoco.mj_forward(self.model,self.data)
        tau=float(self.data.qfrc_actuator[self.cart_d])
        self.data.qpos[:]=oq; self.data.qvel[:]=ov; self.data.ctrl[:]=oc
        self.mujoco.mj_forward(self.model,self.data)
        return tau

    def close(self): self.env.close()

def feature(x,u):
    x=np.asarray(x,float); u=np.asarray(u,float)
    return np.array([x[0],math.sin(x[1]),math.cos(x[1]),math.sin(x[2]),math.cos(x[2]),x[3],x[4],x[5],u[0]],float)

def target_delta(x,y):
    d=np.asarray(y,float)-np.asarray(x,float)
    d[1]=wrap(d[1]); d[2]=wrap(d[2])
    return d

def sample_nominal(cfg,n,seed):
    r=cfg['nominal_data']['state_ranges']; rng=np.random.default_rng(seed)
    A=IDP(seed); A.restore()
    F=[]; R=[]
    for i in range(n):
        cart=rng.uniform(*r['cart_position'])
        hp=rng.uniform(*r['hinge_angle'],2)
        cv=rng.uniform(*r['cart_velocity'])
        hv=rng.uniform(*r['hinge_velocity'],2)
        frac=rng.uniform(*r['control_fraction'])
        u=np.array([frac*max(abs(A.lo[0]),abs(A.hi[0]))],float)
        A.set_state(cart,hp,cv,hv)
        x=A.obs(); A.step(u); y=A.obs()
        F.append(feature(x,u)); R.append(target_delta(x,y))
    A.close()
    return np.asarray(F,np.float32),np.asarray(R,np.float32)

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
        self.fmu=self.fsd=self.rmu=self.rsd=None

    def fit(self,F,R,cfg,seed):
        torch=self.torch; seed_all(seed)
        self.fmu=F.mean(0).astype(np.float32); self.fsd=(F.std(0)+1e-8).astype(np.float32)
        self.rmu=R.mean(0).astype(np.float32); self.rsd=(R.std(0)+1e-8).astype(np.float32)
        X=torch.tensor((F-self.fmu)/self.fsd,dtype=torch.float32)
        Y=torch.tensor((R-self.rmu)/self.rsd,dtype=torch.float32)
        opt=torch.optim.AdamW(self.net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        g=torch.Generator(device='cpu'); g.manual_seed(seed)
        for ep in range(cfg['epochs']):
            order=torch.randperm(len(X),generator=g)
            for st in range(0,len(X),cfg['batch_size']):
                j=order[st:st+cfg['batch_size']]
                xb=X[j].to(self.device); yb=Y[j].to(self.device)
                pred=self.net(xb); loss=((pred-yb)**2).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if (ep+1)%20==0: print('epoch',ep+1,'loss',float(loss.detach().cpu()),flush=True)
        return self

    def predict_delta(self,F):
        torch=self.torch
        F=np.asarray(F,np.float32)
        with torch.no_grad():
            X=torch.tensor((F-self.fmu)/self.fsd,dtype=torch.float32,device=self.device)
            p=self.net(X).cpu().numpy()
        return p*self.rsd+self.rmu

    def bundle(self):
        return {'state_dict':{k:v.detach().cpu() for k,v in self.net.state_dict().items()},
                'fmu':self.fmu,'fsd':self.fsd,'rmu':self.rmu,'rsd':self.rsd}

    def load(self,b):
        self.net.load_state_dict(b['state_dict']); self.fmu=b['fmu']; self.fsd=b['fsd']; self.rmu=b['rmu']; self.rsd=b['rsd']; return self

def quality(model,F,R):
    pred=model.predict_delta(F)
    rmse=float(np.sqrt(np.mean((pred-R)**2)))
    persist=float(np.sqrt(np.mean(R**2)))
    norm=float(np.mean(((pred-R)/(model.rsd+1e-8))**2))
    return {'rmse':rmse,'persistence_rmse':persist,'rmse_ratio':rmse/max(persist,1e-12),'normalized_residual_target_mse':norm}

def nominal_residual_stats(model,F,R):
    e=R-model.predict_delta(F)
    return e.mean(0),e.std(0)+1e-6

def static_raw_pairs(cfg):
    e=cfg['exact_pairs']['static']; N=cfg['exact_pairs']['pairs']; rng=np.random.default_rng(cfg['exact_pairs']['pair_root'])
    groups=[]
    for i in range(N):
        c=float(rng.uniform(*e['cart_offset_abs_range']))*float(rng.choice([-1,1]))
        hp=rng.uniform(e['hinge_pose_range'][0],e['hinge_pose_range'][1],2)
        hv=rng.uniform(e['hinge_velocity_range'][0],e['hinge_velocity_range'][1],2)
        cv=float(rng.uniform(-e['cart_velocity_abs_max'],e['cart_velocity_abs_max']))
        fracs=[]; left=0; frac=0.
        for t in range(e['horizon_steps']):
            if left<=0:
                frac=float(rng.uniform(*e['control_fraction_range'])); left=int(rng.integers(1,6))
            fracs.append(frac); left-=1
        rows=[]
        for lab,kind in [(1,'HOME'),(0,'SENSOR')]:
            A=IDP(cfg['exact_pairs']['pair_root']+10000+i*2+lab); A.restore()
            if kind=='HOME': A.set_state(c,hp,cv,hv); bias=0.0
            else: A.set_state(0.0,hp,cv,hv); bias=c
            F=[]; R=[]
            ref0=float(A.data.qpos[A.cart_q]-A.base_qpos0[A.cart_q])
            for frac in fracs:
                u=np.array([frac*max(abs(A.lo[0]),abs(A.hi[0]))],float)
                x=A.obs(bias); A.step(u); y=A.obs(bias)
                F.append(feature(x,u)); R.append(target_delta(x,y))
            rows.append({'label':lab,'kind':kind,'F':np.asarray(F,np.float32),'R':np.asarray(R,np.float32),'ref0':ref0})
            A.close()
        groups.append(rows)
    return groups

def dynamic_raw_pairs(cfg):
    e=cfg['exact_pairs']['dynamic']; N=cfg['exact_pairs']['pairs']; rng=np.random.default_rng(cfg['exact_pairs']['pair_root']+777)
    groups=[]; invalid=0
    for i in range(N):
        hp=rng.uniform(e['hinge_pose_range'][0],e['hinge_pose_range'][1],2); hv=np.zeros(2)
        b=float(rng.uniform(*e['b_abs_range']))*float(rng.choice([-1,1]))
        a=float(rng.uniform(*e['a_abs_range']))*float(rng.choice([-1,1]))
        mult=float(rng.uniform(*e['damping_multiplier_range']))
        Af=IDP(cfg['exact_pairs']['pair_root']+30000+i*2); Ab=IDP(cfg['exact_pairs']['pair_root']+30001+i*2)
        Af.restore(); Ab.restore()
        dnom=float(Af.model.dof_damping[Af.cart_d]); dfault=dnom*mult; delta=dfault-dnom
        Af.model.dof_damping[Af.cart_d]=dfault
        Ab.set_state(0,hp,b,hv); tau_per=Ab.cart_tau_per_unit()
        tau_bias=-delta*b; ubias=tau_bias/tau_per
        if ubias<Ab.lo[0]-1e-12 or ubias>Ab.hi[0]+1e-12:
            invalid+=1; Af.close(); Ab.close(); continue
        rows=[]
        for lab,kind,A,biasctrl in [(1,'DAMPING',Af,0.0),(0,'ACTUATOR',Ab,ubias)]:
            FF=[]; RR=[]
            for vel in [b,b+a]:
                A.set_state(0,hp,vel,hv); u=np.array([0.0])
                x=A.obs(); A.step(u,biasctrl); y=A.obs()
                FF.append(feature(x,u)); RR.append(target_delta(x,y))
            rows.append({'label':lab,'kind':kind,'F':np.asarray(FF,np.float32),'R':np.asarray(RR,np.float32)})
        groups.append(rows); Af.close(); Ab.close()
    return groups,invalid

def summary_features(z):
    z=np.asarray(z,float)
    return np.concatenate([z.mean(0),z.std(0),np.mean(np.abs(z),0),np.max(np.abs(z),0),
                           np.quantile(z,.25,axis=0),np.quantile(z,.50,axis=0),np.quantile(z,.75,axis=0)])

def residualize_pairs(model,mu,sd,static_groups,dynamic_groups):
    S=[]; D=[]; static_pairdiff=[]; dynamic_pairdiff=[]
    for gi,rows in enumerate(static_groups):
        feats=[]
        for row in rows:
            e=row['R']-model.predict_delta(row['F'])
            z=(e-mu)/sd
            s0=summary_features(z)
            s1=np.r_[s0,row['ref0']]
            feats.append((row['label'],s0,s1,row['ref0'],e))
            S.append({'group':gi,'label':row['label'],'S0':s0,'S1':s1,'ref0':row['ref0']})
        static_pairdiff.extend(np.max(np.abs(feats[0][4]-feats[1][4]),axis=1).tolist())
    for gi,rows in enumerate(dynamic_groups):
        rr=[]
        for row in rows:
            e=row['R']-model.predict_delta(row['F'])
            z=(e-mu)/sd
            d1=z[0].copy(); d2=z[:2].ravel().copy()
            rr.append((row['label'],d1,d2,e))
            D.append({'group':gi,'label':row['label'],'D1':d1,'D2':d2})
        dynamic_pairdiff.append(float(np.linalg.norm(rr[0][3][0]-rr[1][3][0])))
    return S,D,np.asarray(static_pairdiff),np.asarray(dynamic_pairdiff)

def fit_eval(Xtr,ytr,Xte,yte):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score,accuracy_score
    models={
      'RF':RandomForestClassifier(n_estimators=500,min_samples_leaf=3,max_features='sqrt',
                                  class_weight='balanced',random_state=1701,n_jobs=-1),
      'SVM':make_pipeline(StandardScaler(),SVC(C=10.0,kernel='rbf',gamma='scale',class_weight='balanced'))
    }
    out={}
    for name,m in models.items():
        m.fit(Xtr,ytr)
        score=m.predict_proba(Xte)[:,1] if hasattr(m,'predict_proba') else m.decision_function(Xte)
        pred=m.predict(Xte)
        auc=float(roc_auc_score(yte,score))
        auc=max(auc,1-auc) # adversarial orientation; pair labels are arbitrary
        out[name]={'auc':auc,'accuracy':float(accuracy_score(yte,pred))}
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True); ap.add_argument('--workdir',required=True)
    a=ap.parse_args(); cfg=json.loads(Path(a.config).read_text()); W=Path(a.workdir); W.mkdir(parents=True,exist_ok=True)
    assert cfg['hidden_unseal_authorized'] is False
    forbidden=set(cfg['hidden_seeds_forbidden']['world_model_seeds'])|set(cfg['hidden_seeds_forbidden']['scenario_roots'])
    used=set(cfg['world_models']['development_seeds'])|set(cfg['nominal_data']['seeds'].values())|{cfg['exact_pairs']['pair_root'],cfg['exact_pairs']['group_split_seed']}
    assert not (forbidden & used), 'hidden/development seed collision'

    import torch
    print('device', 'cuda' if torch.cuda.is_available() else 'cpu', flush=True)
    cache=W/'nominal_data.npz'
    if cache.exists():
        z=np.load(cache)
        Ftr,Rtr,Fc,Rc,Ft,Rt=[z[k] for k in ['Ftr','Rtr','Fc','Rc','Ft','Rt']]
    else:
        print('sampling nominal data',flush=True)
        Ftr,Rtr=sample_nominal(cfg,cfg['world_models']['train_transitions'],cfg['nominal_data']['seeds']['train'])
        Fc,Rc=sample_nominal(cfg,cfg['world_models']['calibration_transitions'],cfg['nominal_data']['seeds']['calibration'])
        Ft,Rt=sample_nominal(cfg,cfg['world_models']['test_transitions'],cfg['nominal_data']['seeds']['test'])
        np.savez_compressed(cache,Ftr=Ftr,Rtr=Rtr,Fc=Fc,Rc=Rc,Ft=Ft,Rt=Rt)

    static_cache=W/'static_raw.pkl'; dynamic_cache=W/'dynamic_raw.pkl'
    if static_cache.exists():
        static_groups=pickle.loads(static_cache.read_bytes())
    else:
        print('generating static pairs',flush=True)
        static_groups=static_raw_pairs(cfg); static_cache.write_bytes(pickle.dumps(static_groups))
    if dynamic_cache.exists():
        dynamic_groups,invalid=pickle.loads(dynamic_cache.read_bytes())
    else:
        print('generating dynamic pairs',flush=True)
        dynamic_groups,invalid=dynamic_raw_pairs(cfg); dynamic_cache.write_bytes(pickle.dumps((dynamic_groups,invalid)))

    # frozen group split
    rng=np.random.default_rng(cfg['exact_pairs']['group_split_seed'])
    ids=np.arange(cfg['exact_pairs']['pairs']); rng.shuffle(ids)
    train_groups=set(map(int,ids[:cfg['exact_pairs']['train_groups']]))
    test_groups=set(map(int,ids[cfg['exact_pairs']['train_groups']:]))

    allS={}; allD={}; quals=[]; spd={}; dpd={}
    for seed in cfg['world_models']['development_seeds']:
        mp=W/f'wm_{seed}.pt'
        wm=WM(Ftr.shape[1],Rtr.shape[1])
        if mp.exists():
            print('load model',seed,flush=True)
            wm.load(torch.load(mp,map_location=wm.device,weights_only=False))
        else:
            print('train model',seed,flush=True)
            wm.fit(Ftr,Rtr,cfg['world_models'],seed)
            torch.save(wm.bundle(),mp)
        q=quality(wm,Ft,Rt); q['seed']=seed; quals.append(q)
        mu,sd=nominal_residual_stats(wm,Fc,Rc)
        S,D,sdd,ddd=residualize_pairs(wm,mu,sd,static_groups,dynamic_groups)
        allS[seed]=S; allD[seed]=D; spd[seed]=sdd; dpd[seed]=ddd
        save_json(W/'progress.json',{'models_done':[x['seed'] for x in quals],'hidden_unsealed':False})

    folds=[]
    for held in cfg['world_models']['development_seeds']:
        source=[s for s in cfg['world_models']['development_seeds'] if s!=held]
        fold={'heldout_wm':held}
        for stage,store,key in [('S0',allS,'S0'),('S1',allS,'S1'),('D1',allD,'D1'),('D2',allD,'D2')]:
            Xtr=[];ytr=[];Xte=[];yte=[]
            for s in source:
                for r in store[s]:
                    if r['group'] in train_groups:
                        Xtr.append(r[key]); ytr.append(r['label'])
            for r in store[held]:
                if r['group'] in test_groups:
                    Xte.append(r[key]); yte.append(r['label'])
            fold[stage]=fit_eval(np.asarray(Xtr),np.asarray(ytr),np.asarray(Xte),np.asarray(yte))
        # structured static action: independent world reference, frozen threshold at half minimum fault magnitude.
        srows=[r for r in allS[held] if r['group'] in test_groups]
        pred=[1 if abs(r['ref0'])>=0.025 else 0 for r in srows]
        fold['S1_structured_accuracy']=float(np.mean(np.asarray(pred)==np.asarray([r['label'] for r in srows])))
        folds.append(fold)

    qc=cfg['world_models']['quality_gate']; gates=cfg['gates']
    qpass=[q['rmse_ratio']<=qc['rmse_vs_persistence_ratio_max'] and q['normalized_residual_target_mse']<=qc['normalized_residual_target_mse_max'] for q in quals]
    static_pair_p95=max(float(np.quantile(spd[s],.95)) for s in spd)
    dynamic_pair_p95=max(float(np.quantile(dpd[s],.95)) for s in dpd)

    s0best=[max(f['S0']['RF']['auc'],f['S0']['SVM']['auc']) for f in folds]
    s1worst=[min(f['S1']['RF']['auc'],f['S1']['SVM']['auc']) for f in folds]
    d1best=[max(f['D1']['RF']['auc'],f['D1']['SVM']['auc']) for f in folds]
    d2worst_auc=[min(f['D2']['RF']['auc'],f['D2']['SVM']['auc']) for f in folds]
    d2worst_acc=[min(f['D2']['RF']['accuracy'],f['D2']['SVM']['accuracy']) for f in folds]
    sacc=[f['S1_structured_accuracy'] for f in folds]

    gatevals={
      'quality':sum(qpass)>=gates['quality_models_passed_min'],
      'static_pair_diff':static_pair_p95<=gates['static_pair_residual_diff_p95_max'],
      'static_S0_median':float(np.median(s0best))<=gates['static_S0_adversarial_best_auc_median_max'],
      'static_S0_folds':sum(x<=gates['static_S0_adversarial_best_auc_median_max'] for x in s0best)>=gates['static_S0_fold_passes_min'],
      'static_S1_median':float(np.median(s1worst))>=gates['static_S1_worst_classifier_auc_median_min'],
      'static_S1_folds':sum(x>=gates['static_S1_worst_classifier_auc_median_min'] for x in s1worst)>=gates['static_S1_fold_passes_min'],
      'static_structured':float(np.median(sacc))>=gates['static_structured_action_accuracy_min'],
      'dynamic_pair_diff':dynamic_pair_p95<=gates['dynamic_D1_pair_residual_diff_p95_max'],
      'dynamic_D1_median':float(np.median(d1best))<=gates['dynamic_D1_adversarial_best_auc_median_max'],
      'dynamic_D1_folds':sum(x<=gates['dynamic_D1_adversarial_best_auc_median_max'] for x in d1best)>=gates['dynamic_D1_fold_passes_min'],
      'dynamic_D2_auc_median':float(np.median(d2worst_auc))>=gates['dynamic_D2_worst_classifier_auc_median_min'],
      'dynamic_D2_acc_median':float(np.median(d2worst_acc))>=gates['dynamic_D2_worst_classifier_accuracy_median_min'],
      'dynamic_D2_folds':sum((a>=gates['dynamic_D2_worst_classifier_auc_median_min'] and b>=gates['dynamic_D2_worst_classifier_accuracy_median_min']) for a,b in zip(d2worst_auc,d2worst_acc))>=gates['dynamic_D2_fold_passes_min'],
      'invalid_pairs':invalid<=gates['invalid_pair_count_max']
    }

    summary={
      'quality_models_passed':int(sum(qpass)),
      'static_pair_residual_diff_p95_worst_wm':static_pair_p95,
      'static_S0_adversarial_best_auc_median':float(np.median(s0best)),
      'static_S0_fold_passes':int(sum(x<=gates['static_S0_adversarial_best_auc_median_max'] for x in s0best)),
      'static_S1_worst_classifier_auc_median':float(np.median(s1worst)),
      'static_S1_fold_passes':int(sum(x>=gates['static_S1_worst_classifier_auc_median_min'] for x in s1worst)),
      'static_structured_action_accuracy_median':float(np.median(sacc)),
      'dynamic_D1_pair_residual_diff_p95_worst_wm':dynamic_pair_p95,
      'dynamic_D1_adversarial_best_auc_median':float(np.median(d1best)),
      'dynamic_D1_fold_passes':int(sum(x<=gates['dynamic_D1_adversarial_best_auc_median_max'] for x in d1best)),
      'dynamic_D2_worst_classifier_auc_median':float(np.median(d2worst_auc)),
      'dynamic_D2_worst_classifier_accuracy_median':float(np.median(d2worst_acc)),
      'dynamic_D2_fold_passes':int(sum((a>=gates['dynamic_D2_worst_classifier_auc_median_min'] and b>=gates['dynamic_D2_worst_classifier_accuracy_median_min']) for a,b in zip(d2worst_auc,d2worst_acc))),
      'invalid_pairs':invalid,
      'gates':gatevals
    }
    decision='DEVELOPMENT_THRESHOLD_PASS' if all(gatevals.values()) else 'DEVELOPMENT_THRESHOLD_FAIL'
    result={'protocol_id':cfg['protocol_id'],'decision':decision,'hidden_unsealed':False,'hidden_seeds_used':False,
            'config_sha256':sha256_file(a.config),'world_model_quality':quals,'folds':folds,'summary':summary}
    save_json(W/'MEC_CROSSFAMILY_IDP_WM_THRESHOLD_DEV_V0_2_RESULT.json',result)
    print(json.dumps({'decision':decision,'summary':summary},indent=2))

if __name__=='__main__':
    main()

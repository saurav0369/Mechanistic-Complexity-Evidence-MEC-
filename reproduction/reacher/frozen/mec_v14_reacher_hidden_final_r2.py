from __future__ import annotations
import argparse, hashlib, json, os, pickle, sys, time
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent; sys.path.insert(0,str(BASE))
import mec_e2e_v1_runner as core
import mec_e2e_v1_dev_full as dev
import mec_e2e_v1_1_realistic_dev as v11
import mec_e2e_v1_3_compositional_voi_dev as v13
import mec_e2e_v1_4_robust_set_commit_dev as v14
PATCH_ID='MEC_V1_4_REACHER_HIDDEN_FINAL_R2_PREHIDDEN_FINALIZER_ALIGNMENT'; CHECKPOINT_EVERY_EPOCHS=5; ALL=v11.PURE+v11.OOV
def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_json(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=Path(str(p)+'.tmp'); t.write_text(json.dumps(x,indent=2)); os.replace(t,p)
def atomic_pickle(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=Path(str(p)+'.tmp')
    with open(t,'wb') as f: pickle.dump(x,f,pickle.HIGHEST_PROTOCOL); f.flush(); os.fsync(f.fileno())
    os.replace(t,p)
def load_pickle(p):
    with open(p,'rb') as f: return pickle.load(f)
def verify_authority(path,names):
    a=json.loads(Path(path).read_text()); exp=a['runtime_files_sha256']
    for n in names:
        p=BASE/n
        if n not in exp or sha256(p)!=exp[n]: raise RuntimeError('Frozen authority hash mismatch: '+n)
    return a
def progress(ck,stage,**kw):
    x={'stage':stage,'updated_unix':time.time(),**kw}; atomic_json(Path(ck)/'PROGRESS_HIDDEN.json',x); print('\n'+'='*70+'\nPROGRESS '+json.dumps(x,indent=2)+'\n'+'='*70,flush=True)
def time_left(d): return None if d is None else d-time.time()
def safe_stop(d,reserve=180):
    x=time_left(d); return False if x is None else x<=reserve
def partial_path(ck,s): return Path(ck)/f'hidden_wm_{s}_PARTIAL.pt'
def final_path(ck,s): return Path(ck)/f'hidden_world_model_{s}.pt'
def move_opt(opt,device):
    import torch
    for st in opt.state.values():
        for k,v in list(st.items()):
            if torch.is_tensor(v): st[k]=v.to(device)
def save_partial(m,opt,g,next_epoch,ck,seed):
    import torch
    p=partial_path(ck,seed); t=Path(str(p)+'.tmp'); torch.save({'patch_id':PATCH_ID,'seed':int(seed),'next_epoch':int(next_epoch),'state_dict':{k:v.detach().cpu() for k,v in m.net.state_dict().items()},'optimizer':opt.state_dict(),'generator_state':g.get_state(),'fmu':m.fmu,'fsd':m.fsd,'rmu':m.rmu,'rsd':m.rsd},t); os.replace(t,p)
def fit_resumable(m,F,R,epochs,batch,lr,wd,seed,ck,deadline):
    import torch
    m.fmu=F.mean(0); m.fsd=F.std(0)+1e-8; m.rmu=R.mean(0); m.rsd=R.std(0)+1e-8
    Xcpu=torch.tensor((F-m.fmu)/m.fsd,dtype=torch.float32); Ycpu=torch.tensor((R-m.rmu)/m.rsd,dtype=torch.float32); X=Xcpu.to(m.device); Y=Ycpu.to(m.device)
    opt=torch.optim.AdamW(m.net.parameters(),lr=lr,weight_decay=wd); g=torch.Generator(device='cpu'); g.manual_seed(seed); start=0; pp=partial_path(ck,seed)
    if pp.exists():
        o=torch.load(pp,map_location='cpu',weights_only=False)
        if o.get('patch_id')!=PATCH_ID or int(o.get('seed'))!=int(seed): raise RuntimeError('hidden WM partial mismatch')
        m.net.load_state_dict(o['state_dict']); opt.load_state_dict(o['optimizer']); move_opt(opt,m.device); g.set_state(o['generator_state']); start=int(o['next_epoch']); m.fmu=np.asarray(o['fmu']); m.fsd=np.asarray(o['fsd']); m.rmu=np.asarray(o['rmu']); m.rsd=np.asarray(o['rsd']); print(f'[resume] hidden WM {seed} epoch {start}/{epochs}',flush=True)
    for ep in range(start,epochs):
        order=torch.randperm(len(Xcpu),generator=g)
        for st in range(0,len(Xcpu),batch):
            j=order[st:st+batch].to(m.device,non_blocking=True); pred=m.net(X[j]); loss=((pred-Y[j])**2).mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        done=ep+1
        if done%CHECKPOINT_EVERY_EPOCHS==0 or done==epochs: save_partial(m,opt,g,done,ck,seed); print(f'[hidden WM {seed}] epoch {done}/{epochs} [saved]',flush=True)
        if safe_stop(deadline,150): save_partial(m,opt,g,done,ck,seed); return None
    return m
def save_final(m,q,norm,ck,seed):
    import torch
    p=final_path(ck,seed); t=Path(str(p)+'.tmp'); mu,sd=norm; torch.save({'patch_id':PATCH_ID,'seed':int(seed),'bundle':m.state_dict_bundle(),'quality':q,'nominal_mu':mu,'nominal_sd':sd},t); os.replace(t,p); pp=partial_path(ck,seed); pp.unlink() if pp.exists() else None
def load_final(ck,seed,inp,out):
    import torch
    p=final_path(ck,seed)
    if not p.exists(): return None
    o=torch.load(p,map_location='cpu',weights_only=False); m=dev.FastResidualWM(inp,out,128,3); m.net.load_state_dict(o['bundle']['state_dict']); m.fmu=np.asarray(o['bundle']['fmu']); m.fsd=np.asarray(o['bundle']['fsd']); m.rmu=np.asarray(o['bundle']['rmu']); m.rsd=np.asarray(o['bundle']['rsd']); return m,o['quality'],(np.asarray(o['nominal_mu']),np.asarray(o['nominal_sd']))
def load_npz(p):
    z=np.load(p); return z['X'],z['U'],z['Y'],z['am'].astype(bool)
def concrete_seed(root,g,j): return int(np.random.SeedSequence([int(root),int(g),int(j)]).generate_state(1,dtype=np.uint32)[0])
def get_scenarios(old,root,n,steps,path,deadline):
    path=Path(path)
    if path.exists():
        s=load_pickle(path); rows=s['rows']; start=int(s['next_group'])
        if start>=n: return rows,True
    else: rows=[]; start=0
    for g in range(start,n):
        for j,kind in enumerate(ALL): rows.append({'group':g,'kind':kind,'root':int(root),'raw':v11.make_scenario(old,concrete_seed(root,g,j),kind,steps)})
        if (g+1)%10==0 or g+1==n: atomic_pickle(path,{'next_group':g+1,'rows':rows}); print(f'[hidden scenarios root={root}] {g+1}/{n} [saved]',flush=True)
        if safe_stop(deadline,180): atomic_pickle(path,{'next_group':g+1,'rows':rows}); return rows,False
    return rows,True
def eval_block(policy,rec,rng_seed):
    X,B,W,G,K=v14.records_to_arrays(rec); res={}; raw={}
    for mode,label in [('MEC_VOI','MEC_VOI'),('RANDOM','RANDOM'),('DISCRIM','DISCRIMINATION_HEURISTIC')]:
        rng=np.random.default_rng(int(rng_seed)+(0 if mode=='MEC_VOI' else 1000 if mode=='RANDOM' else 2000)); outs=[]; costs=[]
        for i in range(len(B)):
            a,b,c=v14.policy(policy['models'],policy['thresholds'],policy['voi'],{s:X[s][i] for s in v14.STAGES},mode,rng); outs.append((a,b)); costs.append(c)
        res[label]=v14.score_outputs(outs,B,costs); raw[label]={'outs':outs,'costs':np.asarray(costs,float)}
    return {'metrics':res,'ratio':float(res['MEC_VOI']['mean_probe_cost']/max(1e-12,res['RANDOM']['mean_probe_cost'])),'B':B,'raw':raw,'K':K,'G':G}
def primary_pass(m): return m['wrong_intervention_risk']<=.05 and m['coverage']>=.70 and m['composite_or_ood_pure_commit']<=.05 and m['composite_or_ood_safe']>=.95
def combine(blocks,label):
    Bs=[]; outs=[]; costs=[]
    for b in blocks: Bs.append(b['B']); outs.extend(b['raw'][label]['outs']); costs.extend(b['raw'][label]['costs'].tolist())
    return v14.score_outputs(outs,np.vstack(Bs),np.asarray(costs,float))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--authority-manifest',required=True); ap.add_argument('--old-config',required=True); ap.add_argument('--hidden-config',required=True); ap.add_argument('--freeze-dir',required=True); ap.add_argument('--r2-checkpoint-dir',required=True); ap.add_argument('--checkpoint-dir',required=True); ap.add_argument('--result-dir',required=True); ap.add_argument('--session-minutes',type=float,default=40); ap.add_argument('--unseal-token',default='NO')
    a=ap.parse_args(); old=json.loads(Path(a.old_config).read_text()); hc=json.loads(Path(a.hidden_config).read_text())
    verify_authority(a.authority_manifest,['mec_v14_reacher_hidden_final_r2.py','MEC_V1_4_REACHER_HIDDEN_FINAL_R2_FROZEN_CONFIG.json','MEC_E2E_V1_FROZEN_CONFIG.json','mec_e2e_v1_4_robust_set_commit_dev.py','mec_e2e_v1_3_compositional_voi_dev.py','mec_e2e_v1_1_realistic_dev.py','mec_e2e_v1_dev_full.py','mec_e2e_v1_runner.py'])
    freeze=Path(a.freeze_dir); r2=Path(a.r2_checkpoint_dir); ck=Path(a.checkpoint_dir); out=Path(a.result_dir); ck.mkdir(parents=True,exist_ok=True); out.mkdir(parents=True,exist_ok=True)
    if a.unseal_token!=hc['unseal_token']: raise SystemExit('HIDDEN LOCKED: exact unseal token required after final policy freeze.')
    bundlep=freeze/'MEC_V1_4_FINAL_POLICY_BUNDLE.pkl'; manp=freeze/'FINAL_POLICY_FREEZE_MANIFEST.json'
    if not bundlep.exists() or not manp.exists(): raise RuntimeError('Final policy freeze missing.')
    fm=json.loads(manp.read_text())
    if fm.get('status')!='FINAL_POLICY_FROZEN_HIDDEN_STILL_SEALED' or sha256(bundlep)!=fm.get('policy_bundle_sha256'): raise RuntimeError('Frozen policy identity invalid')
    policy=load_pickle(bundlep); identity={'policy_bundle_sha256':sha256(bundlep),'hidden_config_sha256':sha256(a.hidden_config),'hidden_runner_sha256':sha256(Path(__file__)),'authority_manifest_sha256':sha256(a.authority_manifest),'protocol_id':hc['protocol_id']}
    sentinel=ck/'HIDDEN_UNSEAL_SENTINEL.json'
    if sentinel.exists():
        oldsent=json.loads(sentinel.read_text())
        for k,v in identity.items():
            if oldsent.get(k)!=v: raise RuntimeError('Hidden already unsealed under different frozen identity. STOP.')
    else: atomic_json(sentinel,{**identity,'unsealed_unix':time.time(),'irreversible':True})
    started=time.time(); deadline=None if a.session_minutes<=0 else started+a.session_minutes*60
    print(f'\n{PATCH_ID}\nHIDDEN IS NOW UNSEALED UNDER FROZEN HASHES\nNO POLICY FITTING OR CALIBRATION WILL OCCUR\n',flush=True)
    X,U,Y,am=load_npz(r2/'nominal_train.npz'); Xc,Uc,Yc,amc=load_npz(r2/'nominal_calibration.npz'); Xt,Ut,Yt,amt=load_npz(r2/'nominal_test.npz'); F,R=core.make_training_arrays(X,U,Y,am)
    models=[]; norms=[]; qualities=[]; seeds=[int(x) for x in hc['hidden_world_model_seeds']]
    for i,seed in enumerate(seeds):
        progress(ck,'HIDDEN_WORLD_MODEL',index=i+1,total=8,seed=seed); z=load_final(ck,seed,F.shape[1],R.shape[1])
        if z is None:
            import torch
            core.seed_all(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None; m=dev.FastResidualWM(F.shape[1],R.shape[1],128,3); m=fit_resumable(m,F,R,120,old['world_models']['batch_size'],old['world_models']['learning_rate'],old['world_models']['weight_decay'],seed,ck,deadline)
            if m is None: progress(ck,'SAFE_STOP_DURING_HIDDEN_WM',seed=seed); return 0
            q=core.evaluate_model(m,Xt,Ut,Yt,am); norm=v13.nominal_norm(m,Xc,Uc,Yc,am); save_final(m,q,norm,ck,seed)
        else: m,q,norm=z
        models.append(m); qualities.append(q); norms.append(norm); print(f'[hidden WM {i+1}/8 saved/resumed]',flush=True)
        if safe_stop(deadline,180): progress(ck,'SAFE_STOP_AFTER_HIDDEN_WM',seed=seed); return 0
    roots=[int(x) for x in hc['hidden_scenario_seed_roots']]; n=int(hc['hidden_groups_per_seed_root']); steps=int(hc['steps']); scenarios={}
    for ri,root in enumerate(roots):
        progress(ck,'HIDDEN_SCENARIOS',index=ri+1,total=len(roots),seed_root=root); rows,done=get_scenarios(old,root,n,steps,ck/f'hidden_scenarios_root_{root}.pkl',deadline); scenarios[root]=rows
        if not done: progress(ck,'SAFE_STOP_DURING_HIDDEN_SCENARIOS',seed_root=root); return 0
    blocks=[]
    for wi,(m,norm,seed) in enumerate(zip(models,norms,seeds)):
        for ri,root in enumerate(roots):
            progress(ck,'HIDDEN_FEATURES_AND_EVAL',wm_index=wi+1,root_index=ri+1,hidden_world_model=seed,seed_root=root); rp=ck/f'hidden_records_wm_{seed}_root_{root}.pkl'
            if rp.exists(): rec=load_pickle(rp)
            else: rec=v13.build_records([m],[norm],scenarios[root],[seed]); atomic_pickle(rp,rec)
            bp=ck/f'hidden_block_wm_{seed}_root_{root}.pkl'
            if bp.exists(): b=load_pickle(bp)
            else: b=eval_block(policy,rec,305000+wi*100+ri); b['wm']=seed; b['root']=root; atomic_pickle(bp,b)
            blocks.append(b); print(f'[hidden block {len(blocks)}/{len(seeds)*len(roots)} saved/resumed]',flush=True)
            if safe_stop(deadline,180): progress(ck,'SAFE_STOP_AFTER_HIDDEN_BLOCK',hidden_world_model=seed,seed_root=root); return 0
    per_wm=[]
    for seed in seeds:
        bb=[b for b in blocks if b['wm']==seed]; m=combine(bb,'MEC_VOI'); r=combine(bb,'RANDOM'); ratio=m['mean_probe_cost']/max(1e-12,r['mean_probe_cost']); per_wm.append({'seed':seed,'MEC_VOI':m,'RANDOM':r,'mec_to_random_probe_cost_ratio':float(ratio),'primary_pass':primary_pass(m),'efficiency_pass':ratio<=.85})
    per_root=[]
    for root in roots:
        bb=[b for b in blocks if b['root']==root]; m=combine(bb,'MEC_VOI'); r=combine(bb,'RANDOM'); per_root.append({'seed_root':root,'MEC_VOI':m,'RANDOM':r,'mec_to_random_probe_cost_ratio':float(m['mean_probe_cost']/max(1e-12,r['mean_probe_cost']))})
    pooled_m=combine(blocks,'MEC_VOI'); pooled_r=combine(blocks,'RANDOM'); pooled_ratio=pooled_m['mean_probe_cost']/max(1e-12,pooled_r['mean_probe_cost']); med={k:float(np.median([w['MEC_VOI'][k] for w in per_wm])) for k in pooled_m}; medratio=float(np.median([w['mec_to_random_probe_cost_ratio'] for w in per_wm]))
    qpass=sum(float(q['rmse_ratio'])<=.5 and float(q['normalized_residual_target_mse'])<=.25 for q in qualities); wmpass=sum(w['primary_pass'] for w in per_wm); effpass=sum(w['efficiency_pass'] for w in per_wm); stability=sum(w['MEC_VOI']['wrong_intervention_risk']>.07 for w in per_wm)<=1
    primary=(qpass>=7 and primary_pass(pooled_m) and wmpass>=6 and primary_pass(med) and stability); efficiency=(medratio<=.85 and effpass>=6); decision='HIDDEN_PRIMARY_PASS_EFFICIENCY_PASS' if primary and efficiency else ('HIDDEN_PRIMARY_PASS_EFFICIENCY_FAIL' if primary else 'HIDDEN_PRIMARY_FAIL')
    summary={'hidden_unsealed':True,'quality_models_passed':int(qpass),'hidden_wm_primary_passes':int(wmpass),'hidden_wm_efficiency_passes':int(effpass),'pooled_MEC_VOI':pooled_m,'pooled_mec_to_random_probe_cost_ratio':float(pooled_ratio),'median_hidden_wm_MEC_VOI':med,'median_hidden_wm_cost_ratio':medratio,'seed_stability_pass':bool(stability),'decision':decision}
    atomic_json(out/'MEC_V1_4_REACHER_HIDDEN_FINAL_RESULT.json',{'protocol_id':hc['protocol_id'],'patch_id':PATCH_ID,'frozen_identity':identity,'summary':summary,'quality':qualities,'per_hidden_world_model':per_wm,'per_hidden_scenario_root':per_root}); progress(ck,'COMPLETE',decision=decision); print('\nFINAL HIDDEN SUMMARY\n'+json.dumps(summary,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())

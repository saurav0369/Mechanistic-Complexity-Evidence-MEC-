from __future__ import annotations
import argparse, hashlib, json, os, pickle, sys, time
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent; sys.path.insert(0,str(BASE))
import mec_e2e_v1_4_robust_set_commit_dev as v14
PATCH_ID='MEC_V1_4_FINAL_POLICY_FREEZE_R2_NONFINAL_FALLBACK'
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

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--authority-manifest',required=True); ap.add_argument('--freeze-config',required=True); ap.add_argument('--v14-config',required=True)
    ap.add_argument('--r2-checkpoint-dir',required=True); ap.add_argument('--v14-checkpoint-dir',required=True); ap.add_argument('--out-dir',required=True)
    a=ap.parse_args(); fc=json.loads(Path(a.freeze_config).read_text()); cfg=json.loads(Path(a.v14_config).read_text())
    verify_authority(a.authority_manifest,['mec_v14_final_policy_freeze_r2.py','MEC_V1_4_FINAL_POLICY_FREEZE_R2_CONFIG.json','MEC_E2E_V1_4_ROBUST_SET_COMMIT_DEV_FROZEN_CONFIG.json','mec_e2e_v1_4_robust_set_commit_dev.py','mec_e2e_v1_3_compositional_voi_dev.py','mec_e2e_v1_1_realistic_dev.py','mec_e2e_v1_dev_full.py','mec_e2e_v1_runner.py'])
    r2=Path(a.r2_checkpoint_dir); v14c=Path(a.v14_checkpoint_dir); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    devp=v14c/'FINAL_RESULT_V14_R1.json'
    if not devp.exists(): raise FileNotFoundError('Missing completed v1.4 result: '+str(devp))
    dev=json.loads(devp.read_text()); s=dev.get('summary',{})
    if s.get('decision') not in ('PRIMARY_PASS_EFFICIENCY_PASS','PRIMARY_PASS_EFFICIENCY_FAIL'): raise RuntimeError('v1.4 development did not authorize final policy freeze')
    if bool(dev.get('hidden_unsealed',True)): raise RuntimeError('Development result indicates hidden was unsealed')
    bundlep=out/'MEC_V1_4_FINAL_POLICY_BUNDLE.pkl'; manp=out/'FINAL_POLICY_FREEZE_MANIFEST.json'
    if bundlep.exists() or manp.exists():
        if not (bundlep.exists() and manp.exists()): raise RuntimeError('Partial freeze output exists; do not overwrite.')
        m=json.loads(manp.read_text())
        if sha256(bundlep)!=m['policy_bundle_sha256']: raise RuntimeError('Frozen policy bundle hash mismatch')
        print('FINAL POLICY ALREADY FROZEN\n'+json.dumps(m,indent=2)); return 0
    seeds=[int(x) for x in cfg['development_world_model_seeds']]
    oldrec=[]
    for seed in seeds:
        rp=r2/f'records_world_model_{seed}_R2.pkl'
        if not rp.exists(): raise FileNotFoundError(str(rp))
        oldrec.extend(load_pickle(rp))
    newrec=[]
    for seed in seeds:
        rp=v14c/f'fresh_records_{seed}.pkl'
        if not rp.exists(): raise FileNotFoundError(str(rp))
        newrec.extend(load_pickle(rp))
    Xo,Bo,Wo,Go,Ko=v14.records_to_arrays(oldrec); Xn,Bn,Wn,Gn,Kn=v14.records_to_arrays(newrec)
    oldtrain=v14.split_old_train(120,int(cfg['old_v13_training']['group_split_seed'])); calg,_=v14.split_fresh(120,int(cfg['fresh_v14_qualification']['split_seed']))
    train=np.isin(Go,list(oldtrain)); cal=np.isin(Gn,list(calg))
    models=v14.fit_heads(Xo,Bo,Ko,train,120000,cfg); voi=v14.fit_voi(models,Xo,train,120700); thr=v14.calibrate(models,Xn,Bn,Kn,Wn,cal,seeds)
    finalizer_disposition={}
    for stage,t in thr.items():
        if not t.get('calibration_found',False):
            if stage == 'full':
                raise RuntimeError('No robust final threshold found for full; hidden NOT authorized')
            # Frozen v1.4 calibrate() returns tau=1.000001 when no safe threshold exists.
            # stage_decision(..., final=False) therefore cannot commit and policy() falls
            # through to full evidence. Preserve that pre-existing conservative semantics.
            if float(t.get('tau',0.0)) <= 1.0:
                raise RuntimeError(f'Unexpected nonfinal no-calibration sentinel for {stage}')
            t['finalizer_disposition']='DISABLED_NONFINAL_STAGE_ALWAYS_ACQUIRE_FULL'
            finalizer_disposition[stage]=t['finalizer_disposition']
            continue
        for b in [t.get('pooled',{})]+t.get('per_source_wm',[]):
            if float(b.get('wrong_ucb',1))>.05+1e-12 or float(b.get('unsafe_ucb',1))>.05+1e-12:
                raise RuntimeError(f'Final threshold robustness failed for {stage}')
        t['finalizer_disposition']='ROBUST_THRESHOLD_FROZEN'
        finalizer_disposition[stage]=t['finalizer_disposition']
    bundle={'patch_id':PATCH_ID,'protocol_id':fc['protocol_id'],'development_policy_protocol_id':cfg['protocol_id'],'hidden_unsealed':False,
      'models':models,'voi':voi,'thresholds':thr,'development_world_model_seeds':seeds,'head_fit_seed':120000,'voi_fit_seed':120700,
      'old_train_groups':sorted(int(x) for x in oldtrain),'fresh_calibration_groups':sorted(int(x) for x in calg),'v14_qualification_test_groups_used':False,
      'finalizer_disposition':finalizer_disposition}
    atomic_pickle(bundlep,bundle)
    manifest={'status':'FINAL_POLICY_FROZEN_HIDDEN_STILL_SEALED','created_unix':time.time(),'patch_id':PATCH_ID,'protocol_id':fc['protocol_id'],
      'development_decision':s.get('decision'),'development_result_sha256':sha256(devp),'authority_manifest_sha256':sha256(a.authority_manifest),
      'freeze_config_sha256':sha256(a.freeze_config),'v14_config_sha256':sha256(a.v14_config),'policy_bundle_sha256':sha256(bundlep),'policy_bundle_path':str(bundlep),'thresholds':thr,'finalizer_disposition':finalizer_disposition,'hidden_unsealed':False}
    atomic_json(manp,manifest); atomic_json(out/'FINAL_POLICY_FREEZE_SUMMARY.json',{'status':manifest['status'],'policy_bundle_sha256':manifest['policy_bundle_sha256'],'development_decision':s.get('decision'),'hidden_unsealed':False,'finalizer_disposition':finalizer_disposition,
      'stages':{k:{'tau':v['tau'],'calibration_found':v['calibration_found'],'finalizer_disposition':v.get('finalizer_disposition')} for k,v in thr.items()}})
    print('\nFINAL POLICY FREEZE PASS\n'+json.dumps(manifest,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())

from __future__ import annotations
import argparse, json, os, pickle, sys, time, hashlib
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
import mec_e2e_v1_runner as core
import mec_e2e_v1_dev_full as dev
import mec_e2e_v1_1_realistic_dev as v11
import mec_e2e_v1_3_compositional_voi_dev as v13

PURE=v11.PURE
OOV=v11.OOV
ALL=PURE+OOV
COST={"static":2.0,"dynamic":3.0,"full":5.0}
STAGES=["passive","static","dynamic","full"]
PATCH_ID="MEC_V1_4_R1_ROBUST_SET_CERTIFICATE_FAST_RESUMABLE"

STATE_NAMES=[
 "ENCODER_OFFSET","MECHANICAL_REFERENCE_SHIFT","DYNAMICS_DAMPING_FAULT","ACTUATOR_BIAS",
 "ENCODER_OFFSET+DYNAMICS_DAMPING_FAULT","MECHANICAL_REFERENCE_SHIFT+ACTUATOR_BIAS","OUTSIDE_HULL_MORPHOLOGY"]
STATE_ID={k:i for i,k in enumerate(STATE_NAMES)}

def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=Path(str(path)+".tmp")
    tmp.write_text(json.dumps(obj,indent=2)); os.replace(tmp,path)

def atomic_pickle(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=Path(str(path)+".tmp")
    with open(tmp,"wb") as f: pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def load_pickle(path):
    with open(path,"rb") as f: return pickle.load(f)

def progress(ckpt,stage,**kw):
    x={"stage":stage,"updated_unix":time.time(),**kw}; atomic_json(Path(ckpt)/"PROGRESS_V14.json",x)
    print("\n"+"="*68+"\nPROGRESS "+json.dumps(x,indent=2)+"\n"+"="*68,flush=True)

def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def split_old_train(n,seed):
    rng=np.random.default_rng(seed); ids=np.arange(n); rng.shuffle(ids); return set(ids[:60])

def split_fresh(n,seed):
    rng=np.random.default_rng(seed); ids=np.arange(n); rng.shuffle(ids); return set(ids[:60]),set(ids[60:120])

def infer_model_dims(obj):
    sd=obj["bundle"]["state_dict"]
    ws=[v for k,v in sd.items() if k.endswith("weight")]
    return int(ws[0].shape[1]), int(ws[-1].shape[0])

def load_r2_model(r2,seed):
    import torch
    p=Path(r2)/f"world_model_{seed}.pt"
    if not p.exists(): raise FileNotFoundError(f"Missing R2 world model: {p}")
    obj=torch.load(p,map_location="cpu",weights_only=False)
    inp,out=infer_model_dims(obj)
    m=dev.FastResidualWM(inp,out,128,3); m.net.load_state_dict(obj["bundle"]["state_dict"])
    m.fmu=np.asarray(obj["bundle"]["fmu"]); m.fsd=np.asarray(obj["bundle"]["fsd"])
    m.rmu=np.asarray(obj["bundle"]["rmu"]); m.rsd=np.asarray(obj["bundle"]["rsd"])
    return m,obj["quality"],(np.asarray(obj["nominal_mu"]),np.asarray(obj["nominal_sd"]))

def get_fresh_scenarios(old,cfg,path):
    n=int(cfg["fresh_v14_qualification"]["groups"]); seed0=int(cfg["fresh_v14_qualification"]["scenario_seed0"])
    if Path(path).exists():
        st=load_pickle(path); rows=st["rows"]; start=int(st["next_group"])
        if start>=n: print("[resume] fresh scenarios complete",flush=True); return rows
    else: rows=[]; start=0
    for g in range(start,n):
        for j,kind in enumerate(ALL):
            raw=v11.make_scenario(old,seed0+g*100+j,kind,int(cfg["fresh_v14_qualification"]["steps"]))
            rows.append({"group":g,"kind":kind,"raw":raw})
        if (g+1)%5==0 or g+1==n:
            atomic_pickle(path,{"next_group":g+1,"rows":rows}); print(f"[fresh scenarios] {g+1}/{n} groups [saved]",flush=True)
    return rows

def records_to_arrays(rec):
    X={s:np.vstack([r["f"][s] for r in rec]) for s in STAGES}
    B=np.vstack([r["bits"] for r in rec]).astype(int)
    W=np.asarray([r["wm"] for r in rec]); G=np.asarray([r["group"] for r in rec])
    K=np.asarray([STATE_ID[r["kind"]] for r in rec],int)
    return X,B,W,G,K

def fit_binary_hgb(x,y,seed,cfg):
    from sklearn.ensemble import HistGradientBoostingClassifier
    h=HistGradientBoostingClassifier(max_iter=int(cfg["head"]["max_iter"]),learning_rate=float(cfg["head"]["learning_rate"]),
       max_leaf_nodes=int(cfg["head"]["max_leaf_nodes"]),l2_regularization=float(cfg["head"]["l2_regularization"]),
       class_weight="balanced",random_state=int(seed),early_stopping=True,validation_fraction=.15,n_iter_no_change=12)
    h.fit(np.asarray(x,float),np.asarray(y,int)); return h

def fit_state_hgb(x,y,seed,cfg):
    from sklearn.ensemble import HistGradientBoostingClassifier
    h=HistGradientBoostingClassifier(max_iter=int(cfg["head"]["max_iter"]),learning_rate=float(cfg["head"]["learning_rate"]),
       max_leaf_nodes=int(cfg["head"]["max_leaf_nodes"]),l2_regularization=float(cfg["head"]["l2_regularization"]),
       class_weight="balanced",random_state=int(seed),early_stopping=True,validation_fraction=.15,n_iter_no_change=12)
    h.fit(np.asarray(x,float),np.asarray(y,int)); return h

def fit_heads(X,B,K,mask,seed,cfg):
    out={}
    for j,s in enumerate(STAGES):
        bits=[fit_binary_hgb(X[s][mask],B[mask,k],seed+100*j+k,cfg) for k in range(4)]
        state=fit_state_hgb(X[s][mask],K[mask],seed+500+j,cfg)
        out[s]={"bits":bits,"state":state}
    return out

def bit_probs(models,s,x):
    xx=np.asarray(x,float); ps=[]
    for h in models[s]["bits"]:
        p=h.predict_proba(xx); classes=list(h.classes_)
        ps.append(p[:,classes.index(1)] if 1 in classes else np.zeros(len(xx)))
    return np.vstack(ps).T

def state_probs(models,s,x):
    h=models[s]["state"]; p=h.predict_proba(np.asarray(x,float)); out=np.zeros((len(p),7),float)
    for j,c in enumerate(h.classes_): out[:,int(c)]=p[:,j]
    return out

def bin_entropy(p):
    p=np.clip(np.asarray(p,float),1e-6,1-1e-6); return -(p*np.log(p)+(1-p)*np.log(1-p)).sum(1)

def fit_voi(models,X,train,seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    pp=bit_probs(models,"passive",X["passive"]); ps=bit_probs(models,"static",X["static"]); pd=bit_probs(models,"dynamic",X["dynamic"])
    base=bin_entropy(pp); ys=np.maximum(0,base-bin_entropy(ps))/COST["static"]; yd=np.maximum(0,base-bin_entropy(pd))/COST["dynamic"]
    rs=HistGradientBoostingRegressor(max_iter=50,learning_rate=.08,max_leaf_nodes=15,l2_regularization=.1,random_state=seed,early_stopping=True)
    rd=HistGradientBoostingRegressor(max_iter=50,learning_rate=.08,max_leaf_nodes=15,l2_regularization=.1,random_state=seed+1,early_stopping=True)
    rs.fit(X["passive"][train],ys[train]); rd.fit(X["passive"][train],yd[train]); return rs,rd

def certificate_arrays(models,s,x):
    bp=bit_probs(models,s,x); sp=state_probs(models,s,x)
    top=bp.argmax(1); ordered=np.sort(bp,axis=1); sec=ordered[:,-2]
    state_pure=sp[np.arange(len(sp)),top]; nonpure=sp[:,4:].sum(1)
    score=np.minimum.reduce([bp[np.arange(len(bp)),top],1-sec,state_pure,1-nonpure])
    return score,top,bp,sp

def cp_upper(errors,n,conf=.95):
    from scipy.stats import beta
    errors=int(errors); n=int(n)
    if n<=0: return 1.0
    if errors>=n: return 1.0
    return float(beta.ppf(conf,errors+1,n-errors))

def block_stats(score,pred,B,K,W,mask,tau):
    pure=(B.sum(1)==1); nonpure=~pure; commit=(score>=tau)&mask
    cp=commit&pure; npmask=mask&nonpure
    wrong=int(np.sum(pred[cp]!=np.argmax(B[cp],axis=1))) if np.any(cp) else 0
    ncommit=int(cp.sum()); unsafe=int((commit&nonpure).sum()); nnon=int(npmask.sum())
    cov=float((commit&pure).sum()/max(1,(mask&pure).sum()))
    return {"wrong":wrong,"committed_pure":ncommit,"wrong_ucb":cp_upper(wrong,ncommit),
            "unsafe":unsafe,"nonpure_n":nnon,"unsafe_ucb":cp_upper(unsafe,nnon),"coverage":cov}

def calibrate_stage(models,s,X,B,K,W,cal,source_wms):
    score,pred,_,_=certificate_arrays(models,s,X[s])
    vals=np.unique(score[cal]);
    if len(vals)>250:
        vals=np.unique(np.quantile(vals,np.linspace(0,1,250)))
    candidates=np.unique(np.r_[0.0,vals,1.000001])
    best=None
    for tau in candidates:
        pooled=block_stats(score,pred,B,K,W,cal,float(tau))
        blocks=[]; safe=(pooled["wrong_ucb"]<=.05 and pooled["unsafe_ucb"]<=.05)
        for wm in source_wms:
            b=block_stats(score,pred,B,K,W,cal&(W==wm),float(tau)); blocks.append((int(wm),b))
            safe=safe and b["wrong_ucb"]<=.05 and b["unsafe_ucb"]<=.05
        if not safe: continue
        mincov=min(b["coverage"] for _,b in blocks); cand=(mincov,pooled["coverage"],-float(tau))
        if best is None or cand>best[0]: best=(cand,float(tau),pooled,blocks)
    if best is None:
        return {"tau":1.000001,"calibration_found":False,"pooled":{},"per_source_wm":[]}
    return {"tau":best[1],"calibration_found":True,"pooled":best[2],"per_source_wm":[{"wm":w,**b} for w,b in best[3]]}

def calibrate(models,X,B,K,W,cal,source_wms):
    return {s:calibrate_stage(models,s,X,B,K,W,cal,source_wms) for s in ["static","dynamic","full"]}

def stage_decision(models,thr,s,x,final=False):
    score,pred,_,_=certificate_arrays(models,s,np.asarray(x)[None,:]); ok=float(score[0])>=float(thr[s]["tau"])
    if ok: return "COMMIT",int(pred[0]),float(score[0])
    return ("ESCALATE" if final else "MORE"),None,float(score[0])

def discrim_probe(passive_p):
    top=np.argsort(passive_p)[-2:]; st=sum(int(k in (0,1)) for k in top); dy=sum(int(k in (2,3)) for k in top); return "static" if st>=dy else "dynamic"

def policy(models,thr,voi,features,mode,rng):
    pp=bit_probs(models,"passive",np.asarray(features["passive"])[None,:])[0]
    if mode=="MEC_VOI":
        us=float(voi[0].predict(np.asarray(features["passive"])[None,:])[0]); ud=float(voi[1].predict(np.asarray(features["passive"])[None,:])[0]); first="static" if us>=ud else "dynamic"
    elif mode=="RANDOM": first="static" if rng.random()<.5 else "dynamic"
    elif mode=="DISCRIM": first=discrim_probe(pp)
    else: raise ValueError(mode)
    a,b,_=stage_decision(models,thr,first,features[first],False)
    if a=="COMMIT": return a,b,COST[first]
    a,b,_=stage_decision(models,thr,"full",features["full"],True); return a,b,COST["full"]

def score_outputs(outs,B,cost):
    B=np.asarray(B,int); pure=B.sum(1)==1; nonpure=~pure
    commit=np.array([a=="COMMIT" for a,_ in outs]); pred=np.array([-1 if b is None else b for _,b in outs]); true=np.argmax(B,axis=1)
    cp=commit&pure; wrong=np.mean(pred[cp]!=true[cp]) if np.any(cp) else 0.0; cov=np.mean(commit[pure]) if np.any(pure) else 0.0
    correct=np.sum((pred[pure]==true[pure])&commit[pure])/max(1,pure.sum()); unsafe=np.mean(commit[nonpure]) if np.any(nonpure) else 0.0
    return {"wrong_intervention_risk":float(wrong),"coverage":float(cov),"correct_intervention_rate":float(correct),
            "composite_or_ood_pure_commit":float(unsafe),"composite_or_ood_safe":float(1-unsafe),
            "mean_probe_cost":float(np.mean(np.asarray(cost)[pure])) if np.any(pure) else 0.0}

def fold_eval(oldrec,newrec,hold,old_train_groups,new_cal_groups,new_test_groups,seed,cfg):
    Xo,Bo,Wo,Go,Ko=records_to_arrays(oldrec); Xn,Bn,Wn,Gn,Kn=records_to_arrays(newrec)
    train=(Wo!=hold)&np.isin(Go,list(old_train_groups)); cal=(Wn!=hold)&np.isin(Gn,list(new_cal_groups)); test=(Wn==hold)&np.isin(Gn,list(new_test_groups))
    models=fit_heads(Xo,Bo,Ko,train,seed,cfg); voi=fit_voi(models,Xo,train,seed+700)
    src=[w for w in cfg["development_world_model_seeds"] if w!=hold]; thr=calibrate(models,Xn,Bn,Kn,Wn,cal,src)
    idx=np.where(test)[0]; rng=np.random.default_rng(seed+900); res={}
    for mode,label in [("MEC_VOI","MEC_VOI"),("RANDOM","RANDOM"),("DISCRIM","DISCRIMINATION_HEURISTIC")]:
        outs=[]; costs=[]
        for i in idx:
            a,b,c=policy(models,thr,voi,{s:Xn[s][i] for s in STAGES},mode,rng); outs.append((a,b)); costs.append(c)
        res[label]=score_outputs(outs,Bn[idx],costs)
    ratio=res["MEC_VOI"]["mean_probe_cost"]/max(1e-12,res["RANDOM"]["mean_probe_cost"]); m=res["MEC_VOI"]
    primary=(m["wrong_intervention_risk"]<=.05 and m["coverage"]>=.70 and m["composite_or_ood_pure_commit"]<=.05 and m["composite_or_ood_safe"]>=.95)
    return {"heldout_world_model_seed":int(hold),"n_old_train":int(train.sum()),"n_fresh_cal":int(cal.sum()),"n_fresh_test":int(test.sum()),
            "thresholds":thr,"metrics":res,"mec_to_random_probe_cost_ratio":float(ratio),"primary_pass":bool(primary),"efficiency_pass":bool(ratio<=.85)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--old-config",required=True); ap.add_argument("--v14-config",required=True)
    ap.add_argument("--r2-checkpoint-dir",required=True); ap.add_argument("--checkpoint-dir",required=True); ap.add_argument("--result-dir",required=True)
    a=ap.parse_args(); old=json.loads(Path(a.old_config).read_text()); cfg=json.loads(Path(a.v14_config).read_text())
    r2=Path(a.r2_checkpoint_dir); ck=Path(a.checkpoint_dir); out=Path(a.result_dir); ck.mkdir(parents=True,exist_ok=True); out.mkdir(parents=True,exist_ok=True)
    forbidden=set(cfg["hidden_world_model_seeds_forbidden"])|set(cfg["hidden_scenario_seeds_forbidden"]); used=set(cfg["development_world_model_seeds"])|{cfg["fresh_v14_qualification"]["scenario_seed0"],cfg["fresh_v14_qualification"]["split_seed"]}
    if forbidden&used: raise RuntimeError("HIDDEN SEED COLLISION")
    print(f"\n{PATCH_ID}\nHIDDEN SEEDS REMAIN SEALED\nWORLD MODELS ARE REUSED, NOT RETRAINED\n",flush=True)
    manifest={"protocol_id":cfg["protocol_id"],"v14_config_sha256":sha256(a.v14_config),"hidden_unsealed":False,"r2_source":str(r2)}
    mp=ck/"V14_R1_MANIFEST.json"
    if mp.exists() and json.loads(mp.read_text()).get("v14_config_sha256")!=manifest["v14_config_sha256"]: raise RuntimeError("v1.4 config changed; use new checkpoint folder")
    if not mp.exists(): atomic_json(mp,manifest)

    # Load frozen R2 world models/norms and old v1.3 feature records.
    models=[]; norms=[]; oldrec=[]; quality=[]
    for seed in cfg["development_world_model_seeds"]:
        progress(ck,"LOAD_FROZEN_R2",seed=int(seed)); m,q,n=load_r2_model(r2,seed); models.append(m); quality.append(q); norms.append(n)
        rp=r2/f"records_world_model_{seed}_R2.pkl"
        if not rp.exists(): raise FileNotFoundError(f"Missing old v1.3 feature cache {rp}")
        oldrec.extend(load_pickle(rp)); print(f"[loaded] WM + old records {seed}",flush=True)

    progress(ck,"FRESH_SCENARIOS")
    rows=get_fresh_scenarios(old,cfg,ck/"fresh_v14_scenarios.pkl")

    newrec=[]
    for j,(m,norm,seed) in enumerate(zip(models,norms,cfg["development_world_model_seeds"])):
        progress(ck,"FRESH_FEATURE_RECORDS",index=j+1,total=8,seed=int(seed)); rp=ck/f"fresh_records_{seed}.pkl"
        if rp.exists(): rr=load_pickle(rp); print(f"[resume] fresh records {seed}",flush=True)
        else:
            rr=v13.build_records([m],[norm],rows,[seed]); atomic_pickle(rp,rr); print(f"[saved] fresh records {seed}",flush=True)
        newrec.extend(rr)

    oldtrain=split_old_train(120,int(cfg["old_v13_training"]["group_split_seed"])); calg,testg=split_fresh(120,int(cfg["fresh_v14_qualification"]["split_seed"]))
    folds=[]
    for j,hold in enumerate(cfg["development_world_model_seeds"]):
        progress(ck,"LOMO_V14",index=j+1,total=8,heldout_world_model=int(hold)); fp=ck/f"fold_v14_{hold}.json"
        if fp.exists(): f=json.loads(fp.read_text()); print(f"[resume] v1.4 fold {hold}",flush=True)
        else:
            t=time.time(); f=fold_eval(oldrec,newrec,hold,oldtrain,calg,testg,104000+j,cfg); atomic_json(fp,f)
            m=f["metrics"]["MEC_VOI"]; print(f'[saved] fold {hold} in {(time.time()-t)/60:.2f} min | primary={f["primary_pass"]} eff={f["efficiency_pass"]} | wrong={m["wrong_intervention_risk"]:.3f} cov={m["coverage"]:.3f} unsafe={m["composite_or_ood_pure_commit"]:.3f}',flush=True)
        folds.append(f)

    qpass=[q["rmse_ratio"]<=.5 and q["normalized_residual_target_mse"]<=.25 for q in quality]; pn=sum(f["primary_pass"] for f in folds); en=sum(f["efficiency_pass"] for f in folds)
    keys=folds[0]["metrics"]["MEC_VOI"].keys(); med={k:float(np.median([f["metrics"]["MEC_VOI"][k] for f in folds])) for k in keys}; ratio=float(np.median([f["mec_to_random_probe_cost_ratio"] for f in folds]))
    mpas=(med["wrong_intervention_risk"]<=.05 and med["coverage"]>=.70 and med["composite_or_ood_pure_commit"]<=.05 and med["composite_or_ood_safe"]>=.95)
    primary=(sum(qpass)>=7 and pn>=6 and mpas); eff=(ratio<=.85 and en>=6)
    decision="PRIMARY_PASS_EFFICIENCY_PASS" if primary and eff else ("PRIMARY_PASS_EFFICIENCY_FAIL" if primary else "PRIMARY_FAIL")
    summary={"hidden_unsealed":False,"quality_models_passed":int(sum(qpass)),"primary_folds_passed":int(pn),"efficiency_folds_passed":int(en),"median_MEC_VOI":med,"median_mec_to_random_probe_cost_ratio":ratio,"median_primary_pass":bool(mpas),"decision":decision}
    result={"protocol_id":cfg["protocol_id"],"implementation":PATCH_ID,"hidden_unsealed":False,"folds":folds,"summary":summary}
    atomic_json(out/"MEC_E2E_V1_4_R1_FINAL_RESULT.json",result); atomic_json(ck/"FINAL_RESULT_V14_R1.json",result); progress(ck,"COMPLETE",decision=decision)
    print("\nFINAL V1.4 SUMMARY\n"+json.dumps(summary,indent=2),flush=True)

if __name__=="__main__": main()

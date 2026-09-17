
from __future__ import annotations
import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
import mec_e2e_v1_runner as core
import mec_e2e_v1_dev_full as dev
import mec_e2e_v1_1_realistic_dev as v11

PURE=v11.PURE
OOV=v11.OOV
ALL=PURE+OOV

BITS={
 "ENCODER_OFFSET":np.array([1,0,0,0],int),
 "MECHANICAL_REFERENCE_SHIFT":np.array([0,1,0,0],int),
 "DYNAMICS_DAMPING_FAULT":np.array([0,0,1,0],int),
 "ACTUATOR_BIAS":np.array([0,0,0,1],int),
 "ENCODER_OFFSET+DYNAMICS_DAMPING_FAULT":np.array([1,0,1,0],int),
 "MECHANICAL_REFERENCE_SHIFT+ACTUATOR_BIAS":np.array([0,1,0,1],int),
 "OUTSIDE_HULL_MORPHOLOGY":np.array([0,0,0,0],int),
}

COST={"static":2.0,"dynamic":3.0,"full":5.0}

def save_json(p,x): Path(p).write_text(json.dumps(x,indent=2))
def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def nominal_norm(model,X,U,Y,am):
    F,R=core.make_training_arrays(X,U,Y,am)
    pr,_=model.predict_residual(F)
    rr=R-pr
    return rr.mean(0),rr.std(0)+1e-6

def summary(rr):
    rr=np.asarray(rr,float)
    return np.concatenate([rr.mean(0),rr.std(0),np.mean(np.abs(rr),0),np.max(np.abs(rr),0),
                           np.quantile(rr,.25,axis=0),np.quantile(rr,.5,axis=0),np.quantile(rr,.75,axis=0)])

def feat(model,row,mu,sd):
    F,R=core.make_training_arrays(row["X"],row["U"],row["Y"],row["angle_mask"])
    pr,_=model.predict_residual(F)
    passive=summary((R-pr-mu)/sd)
    Fd,Rd=core.make_training_arrays(row["dyn_X"],row["dyn_U"],row["dyn_Y"],row["angle_mask"])
    pd,_=model.predict_residual(Fd)
    dyn=((Rd-pd-mu)/sd).ravel()
    static=np.asarray(row["static"],float)
    return {"passive":passive,"static":np.r_[passive,static],
            "dynamic":np.r_[passive,dyn],"full":np.r_[passive,static,dyn]}

def split_groups(n,seed):
    rng=np.random.default_rng(seed); ids=np.arange(n); rng.shuffle(ids)
    return set(ids[:60]),set(ids[60:90]),set(ids[90:120])

def build_records(models,norms,rows,seeds):
    rec=[]
    for m,(mu,sd),seed in zip(models,norms,seeds):
        for r in rows:
            rec.append({"wm":seed,"group":r["group"],"kind":r["kind"],
                        "bits":BITS[r["kind"]],"outside":int(r["kind"]=="OUTSIDE_HULL_MORPHOLOGY"),
                        "f":feat(m,r["raw"],mu,sd)})
    return rec

def arrays(rec):
    X={s:np.vstack([r["f"][s] for r in rec]) for s in ["passive","static","dynamic","full"]}
    B=np.vstack([r["bits"] for r in rec]).astype(int)
    O=np.array([r["outside"] for r in rec],int)
    W=np.array([r["wm"] for r in rec]); G=np.array([r["group"] for r in rec])
    return X,B,O,W,G

def fit_heads(X,B,O,mask,seed):
    from sklearn.ensemble import RandomForestClassifier
    out={}
    for j,s in enumerate(["passive","static","dynamic","full"]):
        xx=np.asarray(X[s],float); heads=[]
        for k in range(4):
            c=RandomForestClassifier(n_estimators=600,min_samples_leaf=3,class_weight="balanced",
                                     max_features="sqrt",n_jobs=-1,random_state=seed+20*j+k)
            c.fit(xx[mask],B[mask,k]); heads.append(c)
        od=RandomForestClassifier(n_estimators=600,min_samples_leaf=3,class_weight="balanced",
                                  max_features="sqrt",n_jobs=-1,random_state=seed+200+j)
        od.fit(xx[mask],O[mask])
        out[s]={"heads":heads,"outside":od}
    return out

def head_probs(models,s,X):
    xx=np.asarray(X,float)
    ps=[]
    for h in models[s]["heads"]:
        p=h.predict_proba(xx)
        if p.shape[1]==1:
            val=np.full(len(xx),float(h.classes_[0]==1))
        else:
            idx=list(h.classes_).index(1)
            val=p[:,idx]
        ps.append(val)
    return np.vstack(ps).T

def out_prob(models,s,X):
    h=models[s]["outside"]; p=h.predict_proba(np.asarray(X,float))
    if p.shape[1]==1: return np.full(len(X),float(h.classes_[0]==1))
    return p[:,list(h.classes_).index(1)]

def bin_entropy(p):
    p=np.clip(np.asarray(p,float),1e-6,1-1e-6)
    return -(p*np.log(p)+(1-p)*np.log(1-p)).sum(axis=1)

def fit_voi(models,X,train,seed):
    from sklearn.ensemble import RandomForestRegressor
    pp=head_probs(models,"passive",X["passive"])
    ps=head_probs(models,"static",X["static"])
    pd=head_probs(models,"dynamic",X["dynamic"])
    base=bin_entropy(pp)
    ys=np.maximum(0.0,base-bin_entropy(ps))/COST["static"]
    yd=np.maximum(0.0,base-bin_entropy(pd))/COST["dynamic"]
    rs=RandomForestRegressor(n_estimators=500,min_samples_leaf=4,max_features="sqrt",n_jobs=-1,random_state=seed)
    rd=RandomForestRegressor(n_estimators=500,min_samples_leaf=4,max_features="sqrt",n_jobs=-1,random_state=seed+1)
    rs.fit(X["passive"][train],ys[train]); rd.fit(X["passive"][train],yd[train])
    return rs,rd

def calibrate(models,X,B,O,cal):
    # Grid-search a common mechanism-positive threshold and "other" ceiling.
    # Optimize pure coverage subject to <=5% wrong pure interventions and <=5% unsafe pure commits on non-pure cases.
    P={s:head_probs(models,s,X[s]) for s in ["static","dynamic","full"]}
    OP={s:out_prob(models,s,X[s]) for s in ["static","dynamic","full"]}
    out={}
    pure=(B.sum(1)==1)
    nonpure=~pure
    for s in ["static","dynamic","full"]:
        best=None
        for tp in np.linspace(.50,.95,19):
            for to in np.linspace(.05,.45,17):
                for oo in np.linspace(.35,.90,12):
                    pp=P[s]; op=OP[s]
                    top=pp.argmax(1); topv=pp.max(1)
                    sec=np.sort(pp,axis=1)[:,-2]
                    commit=(topv>=tp)&(sec<=to)&(op<oo)
                    pm=cal&pure; nm=cal&nonpure
                    wrong=np.mean(top[pm&commit] != B[pm&commit].argmax(1)) if np.any(pm&commit) else 0.0
                    cov=np.mean(commit[pm]) if np.any(pm) else 0.0
                    unsafe=np.mean(commit[nm]) if np.any(nm) else 0.0
                    if wrong<=.05+1e-12 and unsafe<=.05+1e-12:
                        # maximize coverage, then minimize unsafe, then lower threshold burden.
                        cand=(cov,-unsafe,-wrong,-tp,to,-oo)
                        if best is None or cand>best[0]:
                            best=(cand,{"positive":float(tp),"other_max":float(to),"outside_max":float(oo),
                                       "cal_wrong":float(wrong),"cal_coverage":float(cov),"cal_nonpure_commit":float(unsafe)})
        if best is None:
            out[s]={"positive":1.01,"other_max":0.0,"outside_max":0.0,
                    "cal_wrong":0.0,"cal_coverage":0.0,"cal_nonpure_commit":0.0}
        else:
            out[s]=best[1]
    return out

def stage_decision(models,thr,s,x,final=False):
    p=head_probs(models,s,np.asarray(x)[None,:])[0]
    op=float(out_prob(models,s,np.asarray(x)[None,:])[0])
    t=thr[s]
    supported=np.where(p>=t["positive"])[0]
    top=int(np.argmax(p)); sec=float(np.sort(p)[-2])
    # Explicit composite detection dominates pure commit.
    if len(supported)>=2:
        return "ESCALATE",None,p,op
    if op>=t["outside_max"]:
        return ("ESCALATE" if final else "MORE"),None,p,op
    if float(p[top])>=t["positive"] and sec<=t["other_max"]:
        return "COMMIT",top,p,op
    return ("ABSTAIN" if final else "MORE"),None,p,op

def discrim_probe(passive_p):
    # Semantic ambiguity heuristic retained only as comparator.
    top=np.argsort(passive_p)[-2:]
    static=sum(int(k in (0,1)) for k in top)
    dynamic=sum(int(k in (2,3)) for k in top)
    return "static" if static>=dynamic else "dynamic"

def policy(models,thr,voi,features,mode,rng):
    pp=head_probs(models,"passive",np.asarray(features["passive"])[None,:])[0]
    if mode=="MEC_VOI":
        us=float(voi[0].predict(np.asarray(features["passive"])[None,:])[0])
        ud=float(voi[1].predict(np.asarray(features["passive"])[None,:])[0])
        first="static" if us>=ud else "dynamic"
    elif mode=="RANDOM":
        first="static" if rng.random()<.5 else "dynamic"
    elif mode=="DISCRIM":
        first=discrim_probe(pp)
    else:
        raise ValueError(mode)
    st,p,_,_=stage_decision(models,thr,first,features[first],False)
    if st=="COMMIT": return st,p,COST[first]
    sf,pf,_,_=stage_decision(models,thr,"full",features["full"],True)
    return sf,pf,COST["full"]

def score(outs,B,cost):
    B=np.asarray(B,int); pure=B.sum(1)==1; nonpure=~pure
    commit=np.array([a=="COMMIT" for a,_ in outs])
    pred=np.array([-1 if b is None else b for _,b in outs])
    true=np.argmax(B,axis=1)
    cp=commit&pure
    wrong=np.mean(pred[cp]!=true[cp]) if np.any(cp) else 0.0
    cov=np.mean(commit[pure]) if np.any(pure) else 0.0
    correct=np.sum((pred[pure]==true[pure])&commit[pure])/max(1,pure.sum())
    unsafe=np.mean(commit[nonpure]) if np.any(nonpure) else 0.0
    return {"wrong_intervention_risk":float(wrong),"coverage":float(cov),
            "correct_intervention_rate":float(correct),"composite_or_ood_pure_commit":float(unsafe),
            "composite_or_ood_safe":float(1-unsafe),
            "mean_probe_cost":float(np.mean(np.asarray(cost)[pure])) if np.any(pure) else 0.0}

def fold_eval(rec,hold,trg,cag,teg,seed):
    X,B,O,W,G=arrays(rec)
    train=(W!=hold)&np.isin(G,list(trg))
    cal=(W!=hold)&np.isin(G,list(cag))
    test=(W==hold)&np.isin(G,list(teg))
    models=fit_heads(X,B,O,train,seed)
    voi=fit_voi(models,X,train,seed+500)
    thr=calibrate(models,X,B,O,cal)
    idx=np.where(test)[0]; rng=np.random.default_rng(seed+900)
    res={}
    for mode,label in [("MEC_VOI","MEC_VOI"),("RANDOM","RANDOM"),("DISCRIM","DISCRIMINATION_HEURISTIC")]:
        outs=[]; costs=[]
        for i in idx:
            a,b,c=policy(models,thr,voi,{s:X[s][i] for s in X},mode,rng)
            outs.append((a,b)); costs.append(c)
        res[label]=score(outs,B[idx],costs)
    ratio=res["MEC_VOI"]["mean_probe_cost"]/max(1e-12,res["RANDOM"]["mean_probe_cost"])
    m=res["MEC_VOI"]
    primary=(m["wrong_intervention_risk"]<=.05 and m["coverage"]>=.70 and
             m["composite_or_ood_pure_commit"]<=.05 and m["composite_or_ood_safe"]>=.95)
    eff=ratio<=.85
    return {"heldout_world_model_seed":int(hold),"n_train":int(train.sum()),"n_cal":int(cal.sum()),"n_test":int(test.sum()),
            "thresholds":thr,"metrics":res,"mec_to_random_probe_cost_ratio":float(ratio),
            "primary_pass":bool(primary),"efficiency_pass":bool(eff)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--old-config",required=True)
    ap.add_argument("--v13-config",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    old=json.loads(Path(a.old_config).read_text()); cfg=json.loads(Path(a.v13_config).read_text())
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    forbidden=set(cfg["hidden_world_model_seeds_forbidden"])|set(cfg["hidden_scenario_seeds_forbidden"])
    used=set(cfg["development_world_model_seeds"])|{cfg["scenario_design"]["development_seed0"],
          cfg["scenario_design"]["group_split_seed"],old["split_seeds"]["nominal_train"],old["split_seeds"]["calibration"]}
    if forbidden&used: raise RuntimeError("hidden seed collision")

    print("collect nominal datasets",flush=True)
    X,U,Y,am=core.collect_nominal("Reacher-v5",old,60000,old["split_seeds"]["nominal_train"],"train")
    Xc,Uc,Yc,amc=core.collect_nominal("Reacher-v5",old,20000,old["split_seeds"]["calibration"],"cal")
    Xt,Ut,Yt,amt=core.collect_nominal("Reacher-v5",old,20000,old["split_seeds"]["calibration"]+99,"test")
    F,R=core.make_training_arrays(X,U,Y,am)

    models=[]; norms=[]; quality=[]
    seeds=cfg["development_world_model_seeds"]
    for j,seed in enumerate(seeds):
        print(f"train development WM {j+1}/8 seed={seed}",flush=True)
        m=dev.FastResidualWM(F.shape[1],R.shape[1],128,3).fit(
            F,R,120,old["world_models"]["batch_size"],old["world_models"]["learning_rate"],
            old["world_models"]["weight_decay"],seed)
        q=core.evaluate_model(m,Xt,Ut,Yt,am); q["seed"]=seed; quality.append(q)
        norms.append(nominal_norm(m,Xc,Uc,Yc,am)); models.append(m)

    print("generate v1.3 development scenarios",flush=True)
    rows=v11.generate_rows(old,cfg["scenario_design"]["groups"],cfg["scenario_design"]["development_seed0"])
    rec=build_records(models,norms,rows,seeds)
    tr,ca,te=split_groups(cfg["scenario_design"]["groups"],cfg["scenario_design"]["group_split_seed"])

    folds=[]
    for j,hold in enumerate(seeds):
        print(f"LOMO fold {j+1}/8 hold={hold}",flush=True)
        folds.append(fold_eval(rec,hold,tr,ca,te,93000+j))

    qpass=[q["rmse_ratio"]<=.5 and q["normalized_residual_target_mse"]<=.25 for q in quality]
    primary_n=sum(f["primary_pass"] for f in folds)
    eff_n=sum(f["efficiency_pass"] for f in folds)
    keys=folds[0]["metrics"]["MEC_VOI"].keys()
    med={k:float(np.median([f["metrics"]["MEC_VOI"][k] for f in folds])) for k in keys}
    ratio=float(np.median([f["mec_to_random_probe_cost_ratio"] for f in folds]))
    median_primary=(med["wrong_intervention_risk"]<=.05 and med["coverage"]>=.70 and
                    med["composite_or_ood_pure_commit"]<=.05 and med["composite_or_ood_safe"]>=.95)
    primary_pass=(sum(qpass)>=7 and primary_n>=6 and median_primary)
    efficiency_pass=(ratio<=.85 and eff_n>=6)
    if primary_pass and efficiency_pass: decision="PRIMARY_PASS_EFFICIENCY_PASS"
    elif primary_pass: decision="PRIMARY_PASS_EFFICIENCY_FAIL"
    else: decision="PRIMARY_FAIL"
    summary={"hidden_unsealed":False,"quality_models_passed":int(sum(qpass)),
             "primary_folds_passed":int(primary_n),"efficiency_folds_passed":int(eff_n),
             "median_MEC_VOI":med,"median_mec_to_random_probe_cost_ratio":ratio,
             "median_primary_pass":bool(median_primary),"decision":decision}
    res={"protocol_id":cfg["protocol_id"],"phase":"V1_3_COMPOSITIONAL_VOI_DEVELOPMENT",
         "hidden_unsealed":False,"quality":quality,"folds":folds,"summary":summary}
    save_json(out/"MEC_E2E_V1_3_COMPOSITIONAL_VOI_DEV_RESULT.json",res)
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()

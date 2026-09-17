
from __future__ import annotations
import argparse, json, math, sys, hashlib
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
import mec_e2e_v1_runner as core
import mec_e2e_v1_dev_full as dev

PURE = ["ENCODER_OFFSET","MECHANICAL_REFERENCE_SHIFT","DYNAMICS_DAMPING_FAULT","ACTUATOR_BIAS"]
OOV = ["ENCODER_OFFSET+DYNAMICS_DAMPING_FAULT",
       "MECHANICAL_REFERENCE_SHIFT+ACTUATOR_BIAS",
       "OUTSIDE_HULL_MORPHOLOGY"]
STATIC_SET={"ENCODER_OFFSET","MECHANICAL_REFERENCE_SHIFT"}
DYNAMIC_SET={"DYNAMICS_DAMPING_FAULT","ACTUATOR_BIAS"}

def save_json(p,x): Path(p).write_text(json.dumps(x,indent=2))
def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def residual_summary(res):
    res=np.asarray(res,float)
    return np.concatenate([
        res.mean(0), res.std(0), np.mean(np.abs(res),0),
        np.max(np.abs(res),0),
        np.quantile(res,.25,axis=0), np.quantile(res,.50,axis=0),
        np.quantile(res,.75,axis=0)
    ])

def random_vec2(rng, mag_range):
    mag=float(rng.uniform(*mag_range)); th=float(rng.uniform(0,2*np.pi))
    return mag*np.array([np.cos(th),np.sin(th)])

def action_stream(rng,A,steps,frac_range):
    out=[]; left=0; frac=None
    mx=np.maximum(np.abs(A.action_low),np.abs(A.action_high))
    for _ in range(steps):
        if left<=0:
            frac=rng.uniform(*frac_range,size=A.action_low.shape)
            left=int(rng.integers(1,6))
        out.append(np.clip(frac*mx,A.action_low,A.action_high)); left-=1
    return out

def apply_fault(A, old, rng, kind, allow_oov=False):
    mm=old["morphology_and_nuisance_ranges"]; fr=old["fault_ranges"]
    encoder=np.zeros(2); actuator=np.zeros_like(A.action_low)
    # in-hull nuisance first
    mass=float(rng.uniform(*mm["hidden_in_hull_mass_multiplier"]))
    damp=float(rng.uniform(*mm["hidden_nominal_nuisance_damping_multiplier"]))
    if kind=="OUTSIDE_HULL_MORPHOLOGY":
        lo = rng.random()<.5
        mass=float(rng.uniform(*(mm["hidden_oov_mass_multiplier_low"] if lo else mm["hidden_oov_mass_multiplier_high"])))
    A.restore_model(); A.apply_morphology(mass,damp)

    def add_encoder():
        nonlocal encoder
        encoder += random_vec2(rng,fr["encoder_offset_rad_abs"])
    def add_mech():
        A.apply_mechanical_reference(random_vec2(rng,fr["mechanical_reference_shift_rad_abs"]))
    def add_damp():
        A.apply_damping_fault(float(rng.uniform(*fr["damping_multiplier"])))
    def add_act():
        nonlocal actuator
        # target second actuator only; physically closer to dynamic intervention pair.
        frac=float(rng.uniform(*fr["actuator_bias_fraction_of_ctrl_range_abs"]))
        s=float(rng.choice([-1.,1.])); mx=np.maximum(np.abs(A.action_low),np.abs(A.action_high))
        actuator[1 if len(actuator)>1 else 0] += s*frac*mx[1 if len(actuator)>1 else 0]

    if kind=="ENCODER_OFFSET": add_encoder()
    elif kind=="MECHANICAL_REFERENCE_SHIFT": add_mech()
    elif kind=="DYNAMICS_DAMPING_FAULT": add_damp()
    elif kind=="ACTUATOR_BIAS": add_act()
    elif kind=="ENCODER_OFFSET+DYNAMICS_DAMPING_FAULT": add_encoder(); add_damp()
    elif kind=="MECHANICAL_REFERENCE_SHIFT+ACTUATOR_BIAS": add_mech(); add_act()
    elif kind=="OUTSIDE_HULL_MORPHOLOGY": pass
    else: raise ValueError(kind)
    return encoder, actuator, mass, damp

def make_scenario(old, seed, kind, steps=40):
    rng=np.random.default_rng(seed)
    A=core.EnvAdapter("Reacher-v5",seed+10000)
    enc,act,mass,damp=apply_fault(A,old,rng,kind)
    A.reset(seed+20000)
    mm=old["morphology_and_nuisance_ranges"]; dg=old["data_generation"]
    qjit=max(abs(x) for x in mm["sensor_uniform_jitter_q_rad"])
    vjit=max(abs(x) for x in mm["sensor_uniform_jitter_qdot"])
    acts=action_stream(rng,A,steps,dg["action_fraction_of_control_range_hidden"])
    srng=np.random.default_rng(seed+30000)
    X=[];U=[];Y=[]
    for u in acts:
        x=A.observed_state(enc,qjit,vjit,srng)
        A.step(u,act if np.any(act) else None)
        y=A.observed_state(enc,qjit,vjit,srng)
        X.append(x);U.append(u.copy());Y.append(y)
    # Full-rank physical reference probe.
    ucmd=np.array([0.21,-0.17]); c2,w2=A.static_reference_probe(ucmd,enc,np.eye(2))
    # Three diagnostic transitions from actual single system.
    spec=old["fault_ranges"]["dynamic_threshold_points_reacher"]
    pose=np.asarray(spec["pose_q"],float)
    pts=[np.array([0.0,0.20]),np.array([0.55,0.20]),np.array([0.0,0.75])]
    D=[]
    for v in pts:
        A.set_diagnostic_state(pose,v)
        x=A.observed_state(enc,0,0,None)
        u=np.zeros_like(A.action_low)
        A.step(u,act if np.any(act) else None)
        y=A.observed_state(enc,0,0,None)
        D.append((x,u.copy(),y))
    am=A.jmap.angle_mask.copy(); A.close()
    return {
      "kind":kind,"X":np.asarray(X),"U":np.asarray(U),"Y":np.asarray(Y),
      "angle_mask":am,"static":np.r_[c2,w2],
      "dyn_X":np.asarray([z[0] for z in D]),"dyn_U":np.asarray([z[1] for z in D]),
      "dyn_Y":np.asarray([z[2] for z in D]),
      "meta":{"mass":mass,"nominal_nuisance_damping":damp}
    }

def scenario_features(model,row):
    F,R=core.make_training_arrays(row["X"],row["U"],row["Y"],row["angle_mask"])
    pr,_=model.predict_residual(F); passive=residual_summary(R-pr)
    Fd,Rd=core.make_training_arrays(row["dyn_X"],row["dyn_U"],row["dyn_Y"],row["angle_mask"])
    pd,_=model.predict_residual(Fd); dyn=(Rd-pd).ravel()
    static=np.asarray(row["static"],float)
    return {"passive":passive,
            "static":np.r_[passive,static],
            "dynamic":np.r_[passive,dyn],
            "full":np.r_[passive,static,dyn],
            "static_probe":static,"dynamic_probe":dyn}

def split_groups(n,seed=32117):
    rng=np.random.default_rng(seed); ids=np.arange(n); rng.shuffle(ids)
    ntr=n//2; ncal=n//4
    return set(ids[:ntr]),set(ids[ntr:ntr+ncal]),set(ids[ntr+ncal:])

def fit_stage_models(X_by_stage,y,is_oov,train_mask,seed):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    out={}
    pure=(~is_oov)&train_mask
    anytr=train_mask
    for j,stage in enumerate(["passive","static","dynamic","full"]):
        X=np.asarray(X_by_stage[stage],float)
        rf=RandomForestClassifier(n_estimators=500,min_samples_leaf=3,class_weight="balanced",random_state=seed+j)
        rf.fit(X[pure],y[pure])
        oov=RandomForestClassifier(n_estimators=400,min_samples_leaf=3,class_weight="balanced",random_state=seed+100+j)
        oov.fit(X[anytr],is_oov[anytr].astype(int))
        out[stage]={"class_rf":rf,"oov_rf":oov}
    X=np.asarray(X_by_stage["passive"],float)
    mlp=make_pipeline(StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128,64),alpha=1e-3,max_iter=700,early_stopping=True,
                      validation_fraction=.2,random_state=seed+700))
    mlp.fit(X[pure],y[pure])
    out["passive_mlp"]=mlp
    return out

def choose_class_threshold(probs,y,mask,target=.05):
    p=np.asarray(probs); yy=np.asarray(y)
    conf=p.max(1); pred=p.argmax(1)
    vals=np.unique(np.r_[0.0,conf[mask],1.000001])
    best=(1.000001,0.0,0.0)
    for t in vals:
        take=mask&(conf>=t); n=take.sum()
        if n==0: continue
        risk=float(np.mean(pred[take]!=yy[take])); cov=float(n/max(1,mask.sum()))
        if risk<=target+1e-12 and cov>best[1]: best=(float(t),cov,risk)
    return {"threshold":best[0],"coverage":best[1],"risk":best[2]}

def choose_oov_threshold(p_oov,is_oov,mask,target_reject=.80):
    p=np.asarray(p_oov,float); oo=np.asarray(is_oov,bool)
    vals=np.unique(np.r_[0.0,p[mask],1.000001])
    best=None
    for t in vals:
        reject=p>=t
        om=mask&oo; pm=mask&(~oo)
        orate=float(np.mean(reject[om])) if om.any() else 0.0
        false=float(np.mean(reject[pm])) if pm.any() else 1.0
        if orate>=target_reject:
            cand=(false,-orate,float(t),orate)
            if best is None or cand<best: best=cand
    if best is None:
        # highest achievable reject with lowest pure false reject
        arr=[]
        for t in vals:
            reject=p>=t; om=mask&oo; pm=mask&(~oo)
            orate=float(np.mean(reject[om])) if om.any() else 0.0
            false=float(np.mean(reject[pm])) if pm.any() else 1.0
            arr.append((-orate,false,float(t),orate))
        q=min(arr); return {"threshold":q[2],"oov_reject":q[3],"pure_false_reject":q[1],"target_met":False}
    return {"threshold":best[2],"oov_reject":best[3],"pure_false_reject":best[0],"target_met":True}

def standardized_pair_separation(X,y,mask,class_a,class_b):
    xx=np.asarray(X,float); yy=np.asarray(y)
    z=mask&np.isin(yy,[class_a,class_b])
    xa=xx[z&(yy==class_a)]; xb=xx[z&(yy==class_b)]
    if len(xa)<2 or len(xb)<2: return 0.0
    mu=xa.mean(0)-xb.mean(0)
    sd=np.sqrt(.5*(xa.var(0)+xb.var(0)))+1e-6
    return float(np.sqrt(np.mean((mu/sd)**2)))

def calibrate(models,X_by_stage,y,is_oov,cal_mask,target_risk=.05,target_oov=.80):
    cal={"class":{},"oov":{}}
    for stage in ["passive","static","dynamic","full"]:
        X=np.asarray(X_by_stage[stage],float)
        probs=models[stage]["class_rf"].predict_proba(X)
        # classes are 0..3
        pure=cal_mask&(~is_oov)
        cal["class"][stage]=choose_class_threshold(probs,y,pure,target_risk)
        po=models[stage]["oov_rf"].predict_proba(X)[:,1]
        cal["oov"][stage]=choose_oov_threshold(po,is_oov,cal_mask,target_oov)
    return cal

def class_probs(models,stage,x):
    return models[stage]["class_rf"].predict_proba(np.asarray(x,float)[None,:])[0]

def oov_prob(models,stage,x):
    return float(models[stage]["oov_rf"].predict_proba(np.asarray(x,float)[None,:])[0,1])

def stage_decision(models,cal,stage,x):
    po=oov_prob(models,stage,x)
    if po>=cal["oov"][stage]["threshold"]:
        return {"status":"ESCALATE","pred":None,"conf":None}
    p=class_probs(models,stage,x); k=int(np.argmax(p)); conf=float(np.max(p))
    if conf>=cal["class"][stage]["threshold"]:
        return {"status":"COMMIT","pred":k,"conf":conf}
    return {"status":"UNCERTAIN","pred":k,"conf":conf,"probs":p}

def probe_choice_mec(p):
    top=np.argsort(p)[-2:]; s=set(top.tolist())
    if s=={0,1}: return "static"
    if s=={2,3}: return "dynamic"
    # Mixed semantic ambiguity: pick probe aimed at the top class family.
    return "static" if int(np.argmax(p)) in (0,1) else "dynamic"

def build_sep_table(Xs,Xd,y,train_mask):
    tab={}
    for a in range(4):
        for b in range(a+1,4):
            tab[(a,b)]={
              "static":standardized_pair_separation(Xs,y,train_mask,a,b),
              "dynamic":standardized_pair_separation(Xd,y,train_mask,a,b)}
    return tab

def probe_choice_active(p,sep):
    top=sorted(np.argsort(p)[-2:].tolist()); q=sep[tuple(top)]
    return "static" if q["static"]>=q["dynamic"] else "dynamic"

COST={"passive":0,"static":2,"dynamic":3,"full":5}

def adaptive_policy(models,cal,feat,mode,sep=None,rng=None):
    d=stage_decision(models,cal,"passive",feat["passive"])
    if d["status"]!="UNCERTAIN":
        return d["status"],d.get("pred"),0
    p=d["probs"]
    if mode=="MEC": first=probe_choice_mec(p)
    elif mode=="ACTIVE": first=probe_choice_active(p,sep)
    elif mode=="RANDOM": first=("static" if rng.random()<.5 else "dynamic")
    else: raise ValueError(mode)
    d1=stage_decision(models,cal,first,feat[first])
    if d1["status"]!="UNCERTAIN":
        return d1["status"],d1.get("pred"),COST[first]
    # acquire the other evidence channel and use full evidence.
    d2=stage_decision(models,cal,"full",feat["full"])
    return d2["status"] if d2["status"]!="UNCERTAIN" else "ABSTAIN", d2.get("pred"), COST["full"]

def metrics_policy(outcomes,true_y,is_oov,costs):
    oo=np.asarray(is_oov,bool); y=np.asarray(true_y)
    committed=np.array([s=="COMMIT" for s,_ in outcomes])
    pred=np.array([-1 if p is None else p for _,p in outcomes])
    pure=~oo
    cp=committed&pure
    wrong=float(np.mean(pred[cp]!=y[cp])) if cp.any() else 0.0
    cov=float(np.mean(committed[pure])) if pure.any() else 0.0
    correct=float(np.mean((pred==y)&committed&pure)) if pure.any() else 0.0
    oov_wrong=float(np.mean(committed[oo])) if oo.any() else 0.0
    oov_safe=float(np.mean(~committed[oo])) if oo.any() else 0.0
    return {"wrong_intervention_risk":wrong,"coverage":cov,"correct_intervention_rate":correct,
            "combined_oov_wrong_pure_intervention":oov_wrong,
            "combined_oov_abstain_or_escalate":oov_safe,
            "mean_probe_cost":float(np.mean(costs[pure])) if pure.any() else 0.0}

def passive_metrics(model,cal,X,y,is_oov,stage="passive",mlp=False):
    outcomes=[]; costs=[]
    for i,x in enumerate(X):
        po=oov_prob(model,stage,x)
        if po>=cal["oov"][stage]["threshold"]:
            outcomes.append(("ESCALATE",None)); costs.append(0); continue
        if mlp:
            p=model["passive_mlp"].predict_proba(np.asarray(x)[None,:])[0]
            # use RF-calibrated passive threshold to avoid post-hoc extra threshold family
            t=cal["class"]["passive"]["threshold"]
        else:
            p=class_probs(model,stage,x); t=cal["class"][stage]["threshold"]
        if float(np.max(p))>=t: outcomes.append(("COMMIT",int(np.argmax(p))))
        else: outcomes.append(("ABSTAIN",None))
        costs.append(0)
    return metrics_policy(outcomes,y,is_oov,np.asarray(costs))

def evaluate_one_model(model,rows,new,old,model_seed):
    labels={k:i for i,k in enumerate(PURE)}
    n=max(r["group"] for r in rows)+1
    tr,ca,te=split_groups(n,32117)
    group=np.array([r["group"] for r in rows])
    train=np.array([g in tr for g in group]); calmask=np.array([g in ca for g in group]); test=np.array([g in te for g in group])
    is_oov=np.array([r["kind"] in OOV for r in rows])
    y=np.array([labels.get(r["kind"],-1) for r in rows])
    feats=[scenario_features(model,r["raw"]) for r in rows]
    X={s:np.vstack([f[s] for f in feats]) for s in ["passive","static","dynamic","full"]}
    stage=fit_stage_models(X,y,is_oov,train,model_seed+2000)
    cal=calibrate(stage,X,y,is_oov,calmask,.05,.80)
    sep=build_sep_table(X["static"],X["dynamic"],y,train&(~is_oov))
    idx=np.where(test)[0]
    rr=np.random.default_rng(model_seed+91000)
    results={}
    for mode in ["MEC","RANDOM","ACTIVE"]:
        outs=[]; costs=[]
        for i in idx:
            s,p,c=adaptive_policy(stage,cal,feats[i],mode,sep,rr)
            outs.append((s,p)); costs.append(c)
        results[mode]=metrics_policy(outs,y[idx],is_oov[idx],np.asarray(costs))
    results["PASSIVE_RF"]=passive_metrics(stage,cal,X["passive"][idx],y[idx],is_oov[idx],"passive",False)
    results["PASSIVE_MLP"]=passive_metrics(stage,cal,X["passive"][idx],y[idx],is_oov[idx],"passive",True)
    # Full-evidence upper bound, same selective logic but no probe-cost penalty.
    results["FULL_EVIDENCE_ORACLE"]=passive_metrics(stage,cal,X["full"][idx],y[idx],is_oov[idx],"full",False)
    ratio=results["MEC"]["mean_probe_cost"]/max(1e-12,results["RANDOM"]["mean_probe_cost"])
    gates=new["realistic_arm"]["gates"]
    proxy={
      "wrong_risk":results["MEC"]["wrong_intervention_risk"]<=gates["wrong_intervention_risk_max"],
      "coverage":results["MEC"]["coverage"]>=gates["coverage_min"],
      "oov_wrong":results["MEC"]["combined_oov_wrong_pure_intervention"]<=gates["combined_oov_wrong_pure_intervention_max"],
      "oov_safe":results["MEC"]["combined_oov_abstain_or_escalate"]>=gates["combined_oov_abstain_or_escalate_min"],
      "random_cost_ratio":ratio<=gates["random_probe_cost_ratio_max"]
    }
    return {"model_seed":model_seed,"calibration":cal,
            "probe_pair_separation":{f"{a}-{b}":v for (a,b),v in sep.items()},
            "metrics":results,"mec_to_random_cost_ratio":ratio,"development_gate_proxy":proxy,
            "n_train":int(train.sum()),"n_cal":int(calmask.sum()),"n_test":int(test.sum())}

def generate_rows(old,n=120,seed0=733001):
    rows=[]
    # group-aligned generation: every group contains all pure + all OOV types.
    for g in range(n):
        for j,kind in enumerate(PURE+OOV):
            raw=make_scenario(old,seed0 + g*100 + j,kind,40)
            rows.append({"group":g,"kind":kind,"raw":raw})
    return rows

def quality(model,Xt,Ut,Yt,am):
    return core.evaluate_model(model,Xt,Ut,Yt,am)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--old-config",required=True); ap.add_argument("--new-config",required=True)
    ap.add_argument("--out",required=True); ap.add_argument("--groups",type=int,default=120)
    ap.add_argument("--models",type=int,default=8)
    a=ap.parse_args()
    old=json.loads(Path(a.old_config).read_text()); new=json.loads(Path(a.new_config).read_text())
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)

    forbidden=set(new["hidden_seeds"])|set(new["world_models"]["hidden_seeds"])
    used=set(new["world_models"]["development_seeds"][:a.models])|{
        old["split_seeds"]["nominal_train"],old["split_seeds"]["calibration"],733001}
    if forbidden & used: raise RuntimeError("development seed collides with v1.1 hidden seeds")

    # Train full-development models.
    print("collect nominal train/cal/test",flush=True)
    X,U,Y,am=core.collect_nominal("Reacher-v5",old,new["world_models"]["train_transitions"],
                                  old["split_seeds"]["nominal_train"],"train")
    Xt,Ut,Yt,amt=core.collect_nominal("Reacher-v5",old,new["world_models"]["test_transitions"],
                                      old["split_seeds"]["calibration"]+99,"test")
    F,R=core.make_training_arrays(X,U,Y,am)
    models=[]; q=[]
    for i,seed in enumerate(new["world_models"]["development_seeds"][:a.models]):
        print(f"train development model {i+1}/{a.models} seed={seed}",flush=True)
        m=dev.FastResidualWM(F.shape[1],R.shape[1],128,3).fit(
            F,R,new["world_models"]["epochs"],old["world_models"]["batch_size"],
            old["world_models"]["learning_rate"],old["world_models"]["weight_decay"],seed)
        qq=quality(m,Xt,Ut,Yt,am); qq["seed"]=seed; q.append(qq); models.append(m)

    print(f"generate {a.groups} group-aligned realistic scenarios x {len(PURE)+len(OOV)} classes",flush=True)
    rows=generate_rows(old,a.groups,733001)
    by=[]
    for i,m in enumerate(models):
        print(f"evaluate realistic arm model {i+1}/{len(models)}",flush=True)
        by.append(evaluate_one_model(m,rows,new,old,new["world_models"]["development_seeds"][i]))

    qgate=new["world_models"]["quality_gate"]
    qpass=[z["rmse_ratio"]<=qgate["rmse_ratio_max"] and
           z["normalized_residual_target_mse"]<=qgate["normalized_residual_target_mse_max"] for z in q]
    # Aggregate medians; development proxy only, not hidden result.
    modes=["MEC","RANDOM","ACTIVE","PASSIVE_RF","PASSIVE_MLP","FULL_EVIDENCE_ORACLE"]
    agg={}
    for mode in modes:
        keys=by[0]["metrics"][mode].keys()
        agg[mode]={k:float(np.median([z["metrics"][mode][k] for z in by])) for k in keys}
    costratio=float(np.median([z["mec_to_random_cost_ratio"] for z in by]))
    gate_names=by[0]["development_gate_proxy"].keys()
    gate_fraction={k:float(np.mean([z["development_gate_proxy"][k] for z in by])) for k in gate_names}
    summary={
      "hidden_unsealed":False,
      "quality_models_passed":int(sum(qpass)),
      "P0_dev_proxy":sum(qpass)>=qgate["models_required"],
      "median_metrics":agg,
      "median_mec_to_random_cost_ratio":costratio,
      "fraction_models_passing_each_realistic_proxy_gate":gate_fraction,
      "development_ready_for_policy_freeze": bool(
          sum(qpass)>=qgate["models_required"] and
          gate_fraction["wrong_risk"]>=.75 and gate_fraction["coverage"]>=.75 and
          gate_fraction["oov_wrong"]>=.75 and gate_fraction["oov_safe"]>=.75)
    }
    result={"protocol_id":new["protocol_id"],"phase":"V1_1_REALISTIC_DEVELOPMENT_ONLY",
            "hidden_unsealed":False,"old_config_sha256":sha256(a.old_config),
            "new_config_sha256":sha256(a.new_config),
            "quality":q,"by_model":by,"summary":summary}
    save_json(out/"MEC_E2E_V1_1_REALISTIC_DEV_RESULT.json",result)
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()

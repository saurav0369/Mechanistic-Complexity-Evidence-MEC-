from __future__ import annotations
import argparse, json, math, os, random, hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


def wrap_angle(x):
    return (x + np.pi) % (2*np.pi) - np.pi


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def sha256_file(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save_json(p: Path, obj):
    p.write_text(json.dumps(obj, indent=2))


@dataclass
class JointMap:
    qaddr: np.ndarray
    dof: np.ndarray
    hinge_local_idx: np.ndarray
    angle_mask: np.ndarray
    body_ids: np.ndarray


class EnvAdapter:
    """Single-system adapter. No paired nominal/fault simulator exists in evaluation methods."""
    def __init__(self, env_id: str, seed: int):
        import gymnasium as gym
        import mujoco
        self.gym = gym; self.mujoco = mujoco; self.env_id = env_id
        self.env = gym.make(env_id)
        self.env.reset(seed=seed)
        self.model = self.env.unwrapped.model
        self.data = self.env.unwrapped.data
        self.jmap = self._joint_map()
        self.action_low = np.asarray(self.env.action_space.low, dtype=float)
        self.action_high = np.asarray(self.env.action_space.high, dtype=float)
        self.base = {
            'qpos0': self.model.qpos0.copy(),
            'body_mass': self.model.body_mass.copy(),
            'body_inertia': self.model.body_inertia.copy(),
            'dof_damping': self.model.dof_damping.copy(),
        }

    def close(self): self.env.close()

    def _joint_name(self, jid):
        return self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, int(jid)) or f'joint{jid}'

    def _joint_map(self) -> JointMap:
        hinge = int(self.mujoco.mjtJoint.mjJNT_HINGE)
        slide = int(self.mujoco.mjtJoint.mjJNT_SLIDE)
        jids=[]
        if self.env_id.startswith('Reacher'):
            for nm in ('joint0','joint1'):
                jid=self.mujoco.mj_name2id(self.model,self.mujoco.mjtObj.mjOBJ_JOINT,nm)
                if jid < 0: raise RuntimeError(f'Missing {nm} in Reacher model')
                jids.append(int(jid))
        else:
            # Keep scalar dynamic joints; IDP is cart slide + 2 hinge joints.
            for jid,t in enumerate(self.model.jnt_type):
                if int(t) in (hinge,slide): jids.append(jid)
        qaddr=[]; dof=[]; angle=[]; bodies=[]; hinge_local=[]
        for k,jid in enumerate(jids):
            qaddr.append(int(self.model.jnt_qposadr[jid])); dof.append(int(self.model.jnt_dofadr[jid]))
            is_h=int(self.model.jnt_type[jid])==hinge
            angle.append(is_h); bodies.append(int(self.model.jnt_bodyid[jid]))
            if is_h: hinge_local.append(k)
        if len(hinge_local)<2:
            raise RuntimeError(f'{self.env_id}: need >=2 hinge joints for frozen MEC protocol; got {len(hinge_local)}')
        # first two hinge joints are the diagnostic mechanism coordinates
        hinge_local=np.asarray(hinge_local[:2],dtype=int)
        return JointMap(np.asarray(qaddr),np.asarray(dof),hinge_local,np.asarray(angle,dtype=bool),np.asarray(bodies))

    def restore_model(self):
        self.model.qpos0[:] = self.base['qpos0']
        self.model.body_mass[:] = self.base['body_mass']
        self.model.body_inertia[:] = self.base['body_inertia']
        self.model.dof_damping[:] = self.base['dof_damping']
        self.mujoco.mj_setConst(self.model, self.data)

    def apply_morphology(self, mass_mult: float, damping_mult: float):
        # scale unique dynamic bodies associated with selected joints
        for bid in np.unique(self.jmap.body_ids):
            if bid == 0: continue
            self.model.body_mass[bid] = self.base['body_mass'][bid] * mass_mult
            self.model.body_inertia[bid] = self.base['body_inertia'][bid] * mass_mult
        self.model.dof_damping[self.jmap.dof] = self.base['dof_damping'][self.jmap.dof] * damping_mult
        self.mujoco.mj_setConst(self.model, self.data)

    def apply_mechanical_reference(self, h2: np.ndarray):
        # qpos0 is the MuJoCo joint reference. mj_setConst propagates derived constants.
        for jj,val in zip(self.jmap.hinge_local_idx, h2):
            qa=self.jmap.qaddr[jj]
            self.model.qpos0[qa] = self.base['qpos0'][qa] + float(val)
        self.mujoco.mj_setConst(self.model, self.data)

    def apply_damping_fault(self, mult: float):
        # target second diagnostic hinge
        jj=int(self.jmap.hinge_local_idx[1]); da=self.jmap.dof[jj]
        self.model.dof_damping[da] = self.base['dof_damping'][da] * mult

    def reset(self, seed:int):
        self.env.reset(seed=seed)

    def observed_state(self, encoder_offset2=None, jitter_q=0.0, jitter_v=0.0, rng=None):
        q=np.asarray(self.data.qpos[self.jmap.qaddr]-self.model.qpos0[self.jmap.qaddr],dtype=float).copy()
        v=np.asarray(self.data.qvel[self.jmap.dof],dtype=float).copy()
        if encoder_offset2 is not None:
            for jj,val in zip(self.jmap.hinge_local_idx, encoder_offset2): q[jj]+=float(val)
        if rng is not None:
            if jitter_q: q += rng.uniform(-jitter_q,jitter_q,size=len(q))
            if jitter_v: v += rng.uniform(-jitter_v,jitter_v,size=len(v))
        return np.r_[q,v]

    def step(self, commanded_action, actuator_bias=None):
        u=np.asarray(commanded_action,dtype=float).copy()
        if actuator_bias is not None:
            b=np.asarray(actuator_bias,dtype=float)
            if b.size==1: b=np.full_like(u,float(b[0]))
            u=np.clip(u+b,self.action_low,self.action_high)
        return self.env.step(u)

    def set_diagnostic_state(self, hinge_q2, hinge_v2, cart_zero=True):
        # deterministic state setter, never reads hidden fault label/parameters.
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0
        for jj,val in zip(self.jmap.hinge_local_idx,hinge_q2):
            self.data.qpos[self.jmap.qaddr[jj]] = self.model.qpos0[self.jmap.qaddr[jj]] + float(val)
        for jj,val in zip(self.jmap.hinge_local_idx,hinge_v2):
            self.data.qvel[self.jmap.dof[jj]] = float(val)
        self.data.ctrl[:] = 0
        self.mujoco.mj_forward(self.model,self.data)

    def static_reference_probe(self, ucmd2, encoder_offset2, R):
        # Calibration command expressed in nominal numeric coordinate, matching frozen theorem geometry.
        self.data.qvel[:] = 0
        for jj,val in zip(self.jmap.hinge_local_idx,ucmd2):
            qa=self.jmap.qaddr[jj]
            self.data.qpos[qa] = self.base['qpos0'][qa] + float(val)
        self.mujoco.mj_forward(self.model,self.data)
        # physical relative hinge coordinate; sensor coordinate includes encoder offset.
        phys=np.array([self.data.qpos[self.jmap.qaddr[jj]]-self.model.qpos0[self.jmap.qaddr[jj]] for jj in self.jmap.hinge_local_idx])
        sens=phys + np.asarray(encoder_offset2,dtype=float)
        c=sens-np.asarray(ucmd2,dtype=float)
        w=np.asarray(R,dtype=float)@(phys-np.asarray(ucmd2,dtype=float))
        return c,w


def feature_map(state: np.ndarray, action: np.ndarray, angle_mask: np.ndarray):
    state=np.asarray(state); action=np.asarray(action)
    n=len(angle_mask); q=state[:n]; v=state[n:]
    fs=[]
    for x,is_angle in zip(q,angle_mask):
        if is_angle: fs += [math.sin(float(x)), math.cos(float(x))]
        else: fs += [float(x)]
    fs.extend(map(float,v)); fs.extend(map(float,action))
    return np.asarray(fs,dtype=float)


def residual_target(x,y,angle_mask):
    n=len(angle_mask); dq=np.asarray(y[:n]-x[:n],dtype=float)
    dq[angle_mask]=wrap_angle(dq[angle_mask]); dv=np.asarray(y[n:]-x[n:],dtype=float)
    return np.r_[dq,dv]


class ResidualWM:
    def __init__(self,input_dim,output_dim,hidden=128,layers=3):
        import torch
        import torch.nn as nn
        mods=[nn.Linear(input_dim,hidden),nn.SiLU()]
        for _ in range(layers-1): mods += [nn.Linear(hidden,hidden),nn.SiLU()]
        mods += [nn.Linear(hidden,output_dim)]
        self.torch=torch; self.net=nn.Sequential(*mods).double()
        self.fmu=self.fsd=self.rmu=self.rsd=None

    def fit(self,F,R,epochs,batch,lr,wd,seed):
        torch=self.torch; seed_all(seed)
        self.fmu=F.mean(0); self.fsd=F.std(0)+1e-8; self.rmu=R.mean(0); self.rsd=R.std(0)+1e-8
        X=torch.tensor((F-self.fmu)/self.fsd,dtype=torch.float64)
        Y=torch.tensor((R-self.rmu)/self.rsd,dtype=torch.float64)
        opt=torch.optim.AdamW(self.net.parameters(),lr=lr,weight_decay=wd)
        for _ in range(epochs):
            order=torch.randperm(len(X))
            for st in range(0,len(X),batch):
                j=order[st:st+batch]; pred=self.net(X[j]); loss=((pred-Y[j])**2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def predict_residual(self,F):
        torch=self.torch
        with torch.no_grad():
            p=self.net(torch.tensor((F-self.fmu)/self.fsd,dtype=torch.float64)).cpu().numpy()
        return p*self.rsd+self.rmu, p

    def state_dict_bundle(self):
        return {'state_dict':self.net.state_dict(),'fmu':self.fmu,'fsd':self.fsd,'rmu':self.rmu,'rsd':self.rsd}


def collect_nominal(env_id,cfg,n,seed,split):
    rng=np.random.default_rng(seed); A=EnvAdapter(env_id,seed)
    states=[]; actions=[]; nexts=[]
    mm=cfg['morphology_and_nuisance_ranges']; dg=cfg['data_generation']
    qjit=max(abs(x) for x in mm['sensor_uniform_jitter_q_rad']); vjit=max(abs(x) for x in mm['sensor_uniform_jitter_qdot'])
    count=0; episode=0
    while count<n:
        A.restore_model()
        mass=rng.uniform(*mm['train_mass_multiplier'])
        damp=rng.uniform(*mm['train_nominal_damping_multiplier'])
        A.apply_morphology(mass,damp); A.reset(seed+episode)
        seg_left=0; u=None
        while count<n:
            if seg_left<=0:
                frac=rng.uniform(*dg['action_fraction_of_control_range_train'],size=A.action_low.shape)
                # control ranges are symmetric in frozen envs; interpolate robustly.
                u=np.clip(frac*np.maximum(np.abs(A.action_low),np.abs(A.action_high)),A.action_low,A.action_high)
                seg_left=int(rng.integers(1,6))
            x=A.observed_state(jitter_q=qjit,jitter_v=vjit,rng=rng)
            A.step(u)
            y=A.observed_state(jitter_q=qjit,jitter_v=vjit,rng=rng)
            states.append(x); actions.append(u.copy()); nexts.append(y); count+=1; seg_left-=1
            if getattr(A.env,'_elapsed_steps',0)>=getattr(A.env.spec,'max_episode_steps',1000): break
        episode+=1
    angle_mask=A.jmap.angle_mask.copy(); A.close()
    return np.asarray(states),np.asarray(actions),np.asarray(nexts),angle_mask


def make_training_arrays(X,U,Y,angle_mask):
    F=np.vstack([feature_map(x,u,angle_mask) for x,u in zip(X,U)])
    R=np.vstack([residual_target(x,y,angle_mask) for x,y in zip(X,Y)])
    return F,R


def evaluate_model(model,X,U,Y,angle_mask):
    F,R=make_training_arrays(X,U,Y,angle_mask); pr,pn=model.predict_residual(F)
    n=len(angle_mask); pred=np.empty_like(Y)
    pred[:,:n]=X[:,:n]+pr[:,:n]
    for j,a in enumerate(angle_mask):
        if a: pred[:,j]=wrap_angle(pred[:,j])
    pred[:,n:]=X[:,n:]+pr[:,n:]
    e=pred-Y
    for j,a in enumerate(angle_mask):
        if a: e[:,j]=wrap_angle(e[:,j])
    persist=X-Y
    for j,a in enumerate(angle_mask):
        if a: persist[:,j]=wrap_angle(persist[:,j])
    return {
        'aggregate_state_rmse':float(np.sqrt(np.mean(e**2))),
        'persistence_rmse':float(np.sqrt(np.mean(persist**2))),
        'rmse_ratio':float(np.sqrt(np.mean(e**2))/max(1e-12,np.sqrt(np.mean(persist**2)))),
        'normalized_residual_target_mse':float(np.mean((pn-(R-model.rmu)/model.rsd)**2)),
    }


def fault_vector(rng,mag_range,mode='generic'):
    mag=rng.uniform(*mag_range); sign=rng.choice([-1.,1.])
    if mode=='threshold': return np.array([0.,sign*mag])
    th=rng.uniform(0,2*np.pi); return mag*np.array([math.cos(th),math.sin(th)])


def episode_residual_summary(model,env_id,cfg,seed,fault_class,threshold_arm=False,steps=None):
    rng=np.random.default_rng(seed); A=EnvAdapter(env_id,seed)
    mm=cfg['morphology_and_nuisance_ranges']; fr=cfg['fault_ranges']; dg=cfg['data_generation']
    A.restore_model(); A.apply_morphology(rng.uniform(*mm['hidden_in_hull_mass_multiplier']),rng.uniform(*mm['hidden_nominal_nuisance_damping_multiplier']))
    encoder=np.zeros(2); actuator_bias=None; h=np.zeros(2)
    if fault_class=='ENCODER_OFFSET':
        h=fault_vector(rng,fr['encoder_offset_rad_abs'],'threshold' if threshold_arm else 'generic'); encoder=-h if threshold_arm else h
    elif fault_class=='MECHANICAL_REFERENCE_SHIFT':
        h=fault_vector(rng,fr['mechanical_reference_shift_rad_abs'],'threshold' if threshold_arm else 'generic'); A.apply_mechanical_reference(h)
    elif fault_class=='DYNAMICS_DAMPING_FAULT':
        A.apply_damping_fault(rng.uniform(*fr['damping_multiplier']))
    elif fault_class=='ACTUATOR_BIAS':
        m=rng.uniform(*fr['actuator_bias_fraction_of_ctrl_range_abs']); s=rng.choice([-1.,1.]); actuator_bias=np.full(A.action_low.shape,s*m*np.maximum(np.abs(A.action_low),np.abs(A.action_high)))
    else: raise ValueError(fault_class)
    A.reset(seed)
    qjit=max(abs(x) for x in mm['sensor_uniform_jitter_q_rad']); vjit=max(abs(x) for x in mm['sensor_uniform_jitter_qdot'])
    if steps is None: steps=cfg['hidden_test']['trajectory_length_reacher'] if env_id.startswith('Reacher') else cfg['hidden_test']['trajectory_length_inverted_double_pendulum']
    residuals=[]; energy=[]; seg_left=0; u=None
    for t in range(steps):
        if seg_left<=0:
            frac=rng.uniform(*dg['action_fraction_of_control_range_hidden'],size=A.action_low.shape)
            u=np.clip(frac*np.maximum(np.abs(A.action_low),np.abs(A.action_high)),A.action_low,A.action_high); seg_left=int(rng.integers(1,6))
        x=A.observed_state(encoder,qjit,vjit,rng); F=feature_map(x,u,A.jmap.angle_mask)[None,:]
        pr,_=model.predict_residual(F); A.step(u,actuator_bias)
        y=A.observed_state(encoder,qjit,vjit,rng); true_r=residual_target(x,y,A.jmap.angle_mask); r=true_r-pr[0]
        residuals.append(r); energy.append(float(np.dot(r,r))); seg_left-=1
    residuals=np.asarray(residuals)
    # static theorem-aligned probes (only legitimate physical reference + sensor coordinate)
    ucmd=np.array([0.21,-0.17]); R1=np.array([[1.,0.]]); R2=np.eye(2)
    c1,w1=A.static_reference_probe(ucmd,encoder,R1); c2,w2=A.static_reference_probe(ucmd,encoder,R2)
    out={'fault_class':fault_class,'residual_mean':residuals.mean(0).tolist(),'residual_std':residuals.std(0).tolist(),'residual_absmax':np.abs(residuals).max(0).tolist(),'residual_energy_mean':float(np.mean(energy)),'E1_c':c1.tolist(),'E1_w':w1.tolist(),'E2_c':c2.tolist(),'E2_w':w2.tolist()}
    A.close(); return out


def run_dev(cfg,out,smoke=False,env_filter=None):
    import torch
    out.mkdir(parents=True,exist_ok=True)
    envs=[e for e in cfg['environments'] if env_filter in (None,e['id'])]
    report={'protocol_id':cfg['protocol_id'],'phase':'SMOKE' if smoke else 'DEV','hidden_unsealed':False,'environments':{}}
    model_count=2 if smoke else cfg['world_models']['count_per_environment']
    for ec in envs:
        env_id=ec['id']; ntr=2500 if smoke else ec['train_transitions']; ncal=1000 if smoke else ec['calibration_transitions']; nte=1000 if smoke else ec['nominal_test_transitions']
        X,U,Y,am=collect_nominal(env_id,cfg,ntr,cfg['split_seeds']['nominal_train'], 'train'); F,R=make_training_arrays(X,U,Y,am)
        Xte,Ute,Yte,am2=collect_nominal(env_id,cfg,nte,cfg['split_seeds']['calibration']+99,'test')
        if not np.array_equal(am,am2): raise RuntimeError('angle mask changed')
        envrep={'model_quality':[],'dev_residual_examples':[]}
        for mi,seed in enumerate(cfg['world_models']['seeds'][:model_count]):
            wm=ResidualWM(F.shape[1],R.shape[1],cfg['world_models']['architecture'].count('128') and 128 or 128,3).fit(F,R,25 if smoke else cfg['world_models']['epochs'],cfg['world_models']['batch_size'],cfg['world_models']['learning_rate'],cfg['world_models']['weight_decay'],seed)
            q=evaluate_model(wm,Xte,Ute,Yte,am); q['seed']=seed; envrep['model_quality'].append(q)
            if mi==0:
                for k,fc in enumerate(cfg['hidden_test']['pure_fault_classes']):
                    envrep['dev_residual_examples'].append(episode_residual_summary(wm,env_id,cfg,cfg['data_generation']['development_seed']+k,fc,threshold_arm=(fc in ('ENCODER_OFFSET','MECHANICAL_REFERENCE_SHIFT')),steps=10 if smoke else None))
            torch.save(wm.state_dict_bundle(),out/f"{env_id.replace('-','_')}_wm_seed{seed}.pt")
        report['environments'][env_id]=envrep
    save_json(out/'MEC_E2E_V1_DEV_RESULT.json',report)
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--phase',choices=['smoke','dev','hidden'],default='smoke')
    ap.add_argument('--env',default=None)
    ap.add_argument('--unseal-hidden',default='NO')
    a=ap.parse_args(); cfg=json.loads(Path(a.config).read_text()); seed_all(12345)
    if a.phase=='hidden':
        if a.unseal_hidden!='YES_I_ACCEPT_FROZEN_PROTOCOL':
            raise SystemExit('HIDDEN TEST LOCKED. Use exact --unseal-hidden YES_I_ACCEPT_FROZEN_PROTOCOL only after dev engineering is frozen.')
        raise SystemExit('Hidden evaluator intentionally not implemented in the engineering runner yet. Do not unseal.')
    rep=run_dev(cfg,Path(a.out),smoke=(a.phase=='smoke'),env_filter=a.env)
    print(json.dumps(rep,indent=2))

if __name__=='__main__': main()

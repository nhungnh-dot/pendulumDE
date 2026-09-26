import argparse, json, math, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
L1 = 1.00
L2 = 1.05
G = 9.8

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data():
    df = pd.read_csv("data-diff-length.csv", header=None)
    df.columns = ["t","a_right","b_right","a_left","b_left"]
    df = df.iloc[2:].reset_index(drop=True)
    df = df.apply(pd.to_numeric, errors="coerce").dropna().sort_values("t").reset_index(drop=True)
    t = df["t"].to_numpy(dtype=np.float32)
    a_right = df["a_right"].to_numpy(dtype=np.float32)
    a_left = df["a_left"].to_numpy(dtype=np.float32)
    t = t - t[0]
    theta1 = (a_left / L1).astype(np.float32)
    theta2 = (a_right / L2).astype(np.float32)
    t_data = t.reshape(-1,1)
    y1 = theta1.reshape(-1,1)
    y2 = theta2.reshape(-1,1)
    n_train = int(np.floor(0.80 * len(t_data)))
    tr = np.arange(n_train); va = np.arange(n_train, len(t_data))
    theta_scale = max(np.max(np.abs(y1[tr])), np.max(np.abs(y2[tr])), 1e-3)
    T = lambda x: torch.tensor(x, dtype=torch.float32)
    return dict(
        t_min=float(t_data.min()), t_max=float(t_data.max()), theta_scale=float(theta_scale),
        t_train=T(t_data[tr]), t_val=T(t_data[va]),
        y1_train=T(y1[tr]), y2_train=T(y2[tr]),
        y1_val=T(y1[va]), y2_val=T(y2[va]),
        n_data=len(t_data), n_train=len(tr), n_val=len(va)
    )

class FourierPINN(nn.Module):
    def __init__(self, t_min, t_max, theta_scale, n_freq=30, hidden=128):
        super().__init__()
        self.register_buffer("t_min", torch.tensor(float(t_min), dtype=torch.float32))
        self.register_buffer("t_max", torch.tensor(float(t_max), dtype=torch.float32))
        self.register_buffer("theta_scale", torch.tensor(float(theta_scale), dtype=torch.float32))
        self.register_buffer("frequencies", torch.arange(1,n_freq+1,dtype=torch.float32))
        input_dim = 1 + 2*n_freq
        self.net = nn.Sequential(
            nn.Linear(input_dim,hidden), nn.Tanh(),
            nn.Linear(hidden,hidden), nn.Tanh(),
            nn.Linear(hidden,hidden), nn.Tanh(),
            nn.Linear(hidden,hidden), nn.Tanh(),
            nn.Linear(hidden,2)
        )
    def forward(self,t):
        tau = 2.0*(t-self.t_min)/(self.t_max-self.t_min)-1.0
        phase = np.pi * tau * self.frequencies.reshape(1,-1)
        features = torch.cat((tau,torch.sin(phase),torch.cos(phase)), dim=1)
        return self.theta_scale * self.net(features)

def data_loss(model,d, split="train"):
    if split=="train":
        t,y1,y2=d["t_train"],d["y1_train"],d["y2_train"]
    else:
        t,y1,y2=d["t_val"],d["y1_val"],d["y2_val"]
    p=model(t)
    e1=(p[:,0:1]-y1)/d["theta_scale"]
    e2=(p[:,1:2]-y2)/d["theta_scale"]
    return torch.mean(e1**2)+torch.mean(e2**2)

def inverse_softplus(x):
    x=torch.tensor(x,dtype=torch.float32)
    return torch.log(torch.exp(x)-1.0)

def physics_loss(model,d,raw_K,raw_d1,raw_d2,n_physics=512):
    t_phys = d["t_min"] + (d["t_max"]-d["t_min"])*torch.rand(n_physics,1)
    t = t_phys.clone().detach().requires_grad_(True)
    out=model(t); th1=out[:,0:1]; th2=out[:,1:2]
    th1_t=torch.autograd.grad(th1,t,torch.ones_like(th1),create_graph=True)[0]
    th2_t=torch.autograd.grad(th2,t,torch.ones_like(th2),create_graph=True)[0]
    th1_tt=torch.autograd.grad(th1_t,t,torch.ones_like(th1_t),create_graph=True)[0]
    th2_tt=torch.autograd.grad(th2_t,t,torch.ones_like(th2_t),create_graph=True)[0]
    K=F.softplus(raw_K); d1=F.softplus(raw_d1); d2=F.softplus(raw_d2)
    coupling=L1*th1-L2*th2
    r1=th1_tt+d1*th1_t+(G/L1)*th1+(K/L1)*coupling
    r2=th2_tt+d2*th2_t+(G/L2)*th2-(K/L2)*coupling
    a1=(G/L1)*d["theta_scale"]; a2=(G/L2)*d["theta_scale"]
    return torch.mean((r1/a1)**2)+torch.mean((r2/a2)**2)

def deterministic_physics_loss(model,d,raw_K,raw_d1,raw_d2,n=1024):
    t=torch.linspace(d["t_min"],d["t_max"],n).reshape(-1,1).requires_grad_(True)
    out=model(t); th1=out[:,0:1]; th2=out[:,1:2]
    th1_t=torch.autograd.grad(th1,t,torch.ones_like(th1),create_graph=True)[0]
    th2_t=torch.autograd.grad(th2,t,torch.ones_like(th2),create_graph=True)[0]
    th1_tt=torch.autograd.grad(th1_t,t,torch.ones_like(th1_t),create_graph=True)[0]
    th2_tt=torch.autograd.grad(th2_t,t,torch.ones_like(th2_t),create_graph=True)[0]
    K=F.softplus(raw_K); d1=F.softplus(raw_d1); d2=F.softplus(raw_d2)
    coupling=L1*th1-L2*th2
    r1=th1_tt+d1*th1_t+(G/L1)*th1+(K/L1)*coupling
    r2=th2_tt+d2*th2_t+(G/L2)*th2-(K/L2)*coupling
    a1=(G/L1)*d["theta_scale"]; a2=(G/L2)*d["theta_scale"]
    return (torch.mean((r1/a1)**2)+torch.mean((r2/a2)**2)).item()

def stage1(nfreq, d, epochs=10000):
    set_seed()
    model=FourierPINN(d["t_min"],d["t_max"],d["theta_scale"],nfreq,128)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    last=None
    t0=time.time()
    for ep in range(epochs):
        opt.zero_grad()
        loss=data_loss(model,d,"train")
        loss.backward(); opt.step(); last=loss.item()
    val=data_loss(model,d,"val").item()
    return model, {"n_freq":nfreq,"training_loss":last,"validation_loss":val,"seconds":time.time()-t0}

def stage2(nfreq, lam, k0, d, epochs=10000):
    model,s1=stage1(nfreq,d,10000)
    raw_K=nn.Parameter(inverse_softplus(k0))
    raw_d1=nn.Parameter(inverse_softplus(0.02))
    raw_d2=nn.Parameter(inverse_softplus(0.02))
    # Reset RNG for a fair collocation sequence across hyperparameter comparisons.
    torch.manual_seed(SEED)
    opt=torch.optim.Adam(list(model.parameters())+[raw_K,raw_d1,raw_d2],lr=5e-4)
    last_d=last_p=last_total=None
    t0=time.time()
    for ep in range(epochs):
        opt.zero_grad()
        ld=data_loss(model,d,"train")
        lp=physics_loss(model,d,raw_K,raw_d1,raw_d2,512)
        total=ld+lam*lp
        total.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters())+[raw_K,raw_d1,raw_d2],max_norm=10.0)
        opt.step()
        last_d,last_p,last_total=ld.item(),lp.item(),total.item()
    K=F.softplus(raw_K).item(); d1=F.softplus(raw_d1).item(); d2=F.softplus(raw_d2).item()
    val=data_loss(model,d,"val").item()
    phys_grid=deterministic_physics_loss(model,d,raw_K,raw_d1,raw_d2)
    return {
      "n_freq":nfreq,"lambda_p":lam,"K_initial":k0,
      "stage1_training_loss":s1["training_loss"],"stage1_validation_loss":s1["validation_loss"],
      "final_training_data_loss":last_d,"final_validation_loss":val,
      "final_physics_loss_last_random_batch":last_p,
      "final_physics_loss_fixed_grid":phys_grid,
      "final_total_loss_last_batch":last_total,
      "K_final":K,"d1_final":d1,"d2_final":d2,
      "stage2_seconds":time.time()-t0
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["q2","q3","q4"],required=True)
    ap.add_argument("--nfreq",type=int,default=5)
    ap.add_argument("--lambda-p",type=float,default=2.0)
    ap.add_argument("--k0",type=float,default=1.0)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    d=load_data()
    if args.mode=="q2":
        results=[]
        for nf in [0,5,30]:
            _,r=stage1(nf,d,10000)
            print("Q2",r,flush=True); results.append(r)
        payload={"data":{"n_data":d["n_data"],"n_train":d["n_train"],"n_val":d["n_val"],"theta_scale":d["theta_scale"]},"results":results}
    else:
        r=stage2(args.nfreq,args.lambda_p,args.k0,d,10000)
        print(args.mode.upper(),r,flush=True); payload=r
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(payload,indent=2))
if __name__=="__main__":
    main()

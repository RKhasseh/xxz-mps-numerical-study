#!/usr/bin/env python3
"""Add lattice magnetization observables and make delta_m plots from saved MPS trajectories.

Purely numerical/lattice post-processing. No CFT quantities are used.
"""
from pathlib import Path
import argparse
import math
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ALPHAS=(2.0,2.5,3.0,4.0)

def spin_charge(bit: str) -> int:
    return 1 if bit == '1' else -1

def add_magnetization_columns(df: pd.DataFrame, L: int) -> pd.DataFrame:
    n_b1=L//4
    n_b2=L//4
    out=df.copy()
    M1=[]; M2=[]
    for rec in out['record'].astype(str):
        # pandas can strip leading zeros if records are ever parsed as numbers;
        # pad back to the known measured length L/2.
        rec=rec.zfill(n_b1+n_b2)
        b1=rec[:n_b1]; b2=rec[n_b1:n_b1+n_b2]
        M1.append(sum(spin_charge(x) for x in b1))
        M2.append(sum(spin_charge(x) for x in b2))
    out['M_B1']=M1; out['M_B2']=M2
    out['m_B1']=out['M_B1']/n_b1
    out['m_B2']=out['M_B2']/n_b2
    out['delta_m']=out['m_B1']-out['m_B2']
    out['m_measured_total']=(out['M_B1']+out['M_B2'])/(n_b1+n_b2)
    return out

def summarize_delta_m(frames):
    all_df=pd.concat(frames,ignore_index=True)
    rows=[]
    for (D,L,dm),g in all_df.groupby(['D','L','delta_m'],sort=True):
        for a in ALPHAS:
            c=f"S_mps_a{str(a).replace('.','p')}"
            x=g[c].to_numpy(float); x=x[np.isfinite(x)]
            if len(x)==0: continue
            std=float(np.std(x,ddof=1)) if len(x)>1 else 0.0
            rows.append(dict(D=int(D),L=int(L),alpha=a,delta_m=float(dm),mean_entropy=float(np.mean(x)),std_entropy=std,sem_entropy=std/math.sqrt(len(x)) if len(x)>1 else 0.0,count=len(x)))
    return pd.DataFrame(rows)

def plot_candidate(frames, summary, outdir:Path, candidate_D:int):
    outdir.mkdir(parents=True,exist_ok=True)
    all_df=pd.concat(frames,ignore_index=True)
    ddf=all_df[all_df.D==candidate_D]
    if ddf.empty:
        candidate_D=int(all_df.D.max()); ddf=all_df[all_df.D==candidate_D]
    for a in ALPHAS:
        tag=str(a).replace('.','p'); col=f'S_mps_a{tag}'
        b=summary[(summary.D==candidate_D)&(summary.alpha==a)].sort_values('delta_m')
        fig,ax=plt.subplots(figsize=(6.8,4.8))
        ax.scatter(ddf['delta_m'],ddf[col],s=10,alpha=0.18,label='Born trajectories')
        ax.errorbar(b['delta_m'],b['mean_entropy'],yerr=b['sem_entropy'],marker='o',capsize=3,label='Mean ± SEM')
        ax.set_xlabel(r'Magnetization imbalance $\Delta m=m_{B_1}-m_{B_2}$')
        ax.set_ylabel(rf'Conditional Rényi entropy $S_{{{a:g}}}$')
        ax.set_title(f'Entanglement vs magnetization imbalance (L={int(ddf.L.iloc[0])}, D={candidate_D})')
        ax.grid(True,alpha=0.25); ax.legend(); fig.tight_layout()
        fig.savefig(outdir/f'entanglement_vs_delta_m_D{candidate_D}_alpha{tag}.png',dpi=180)
        plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('source',type=Path)
    ap.add_argument('destination',type=Path)
    ap.add_argument('--L',type=int,default=64)
    ap.add_argument('--candidate-D',type=int,default=32)
    args=ap.parse_args()
    dest=args.destination; dest.mkdir(parents=True,exist_ok=True)
    (dest/'summary').mkdir(exist_ok=True); (dest/'trajectories').mkdir(exist_ok=True); (dest/'delta_m_plots').mkdir(exist_ok=True)
    frames=[]
    for ddir in sorted(args.source.glob('D_*'),key=lambda p:int(p.name.split('_')[1])):
        D=int(ddir.name.split('_')[1]); p=ddir/'trajectory_results.csv'
        if not p.exists(): continue
        df=add_magnetization_columns(pd.read_csv(p,dtype={'record':str}),args.L)
        frames.append(df)
        od=dest/'trajectories'/f'D_{D}'; od.mkdir(parents=True,exist_ok=True)
        df.to_csv(od/'trajectory_results.csv',index=False)
        shutil.copy2(ddir/'renyi_summary.csv',od/'renyi_summary.csv')
        shutil.copy2(ddir/'metadata.json',od/'metadata.json')
    summ=summarize_delta_m(frames)
    summ.to_csv(dest/'summary'/'delta_m_summary.csv',index=False)
    plot_candidate(frames,summ,dest/'delta_m_plots',args.candidate_D)

if __name__=='__main__': main()

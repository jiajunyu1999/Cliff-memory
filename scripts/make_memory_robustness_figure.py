from __future__ import annotations
"""Publication figure for causal response-memory robustness controls."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"paper/figures/fig_memory_robustness.pdf"
PNG=ROOT/"reports/results/fig_memory_robustness.png"

def ci(v, seed):
    rng=np.random.default_rng(seed); draw=rng.choice(np.asarray(v,float),(10000,len(v)),replace=True).mean(1)
    return tuple(np.quantile(draw,[.025,.975]))

def clean(ax):
    ax.set_facecolor("white");ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(.8);ax.spines["bottom"].set_linewidth(.8)
    ax.grid(axis="y",ls="--",color="#888888",alpha=.24,zorder=0)

def dose_line(ax, table, metric, color, label, seed):
    p=[]
    for d,g in table.groupby("dose"):
        lo,hi=ci(g[metric],seed+int(d));p.append((d,g[metric].mean(),lo,hi))
    x,mean,lo,hi=map(np.asarray,zip(*p))
    ax.errorbar(x,mean,yerr=np.vstack([mean-lo,hi-mean]),color=color,marker="o",lw=1.55,ms=4.4,capsize=2.1,label=label,zorder=4)

def dot_ci(ax, values, x, color, seed):
    values=np.asarray(values,float);lo,hi=ci(values,seed);mean=values.mean()
    ax.errorbar(x,mean,yerr=[[mean-lo],[hi-mean]],fmt="o",color=color,ms=5,capsize=2.5,lw=1.35,zorder=4)
    ax.scatter(np.full(len(values),x)+np.random.default_rng(seed).normal(0,.035,len(values)),values,s=8,color=color,alpha=.35,zorder=3)

def main():
    matched=pd.read_csv(ROOT/"reports/results/matched_train_test_memory_dose.csv")
    shuffled=pd.read_csv(ROOT/"reports/results/fixed_mask_potency_shuffle.csv")
    family=pd.read_csv(ROOT/"reports/results/family_count_matched_exclusion.csv")
    budget=pd.read_csv(ROOT/"reports/results/same_budget_profile_baselines.csv")
    radius=pd.read_csv(ROOT/"reports/results/biological_exclusion_radius.csv")
    temporal=pd.read_csv(ROOT/"reports/results/temporal_profile_cutoff.csv")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.4,"axes.labelsize":9.7,"xtick.labelsize":7.8,"ytick.labelsize":8.1,"pdf.fonttype":42})
    fig,axes=plt.subplots(2,3,figsize=(9.35,5.15),facecolor="white")
    blue,red,green,gray,ochre,purple="#4DBBD5","#E64B35","#00A087","#6C7A89","#C79052","#8D78B8"
    # a: train and test are masked together; average repeats before target macro CI.
    ax=axes[0,0];tm=matched.groupby(["dataset","dose"],as_index=False)[["overall_rmse_reduction","cliff_rmse_reduction"]].mean()
    dose_line(ax,tm,"overall_rmse_reduction",blue,"Overall",100);dose_line(ax,tm,"cliff_rmse_reduction",red,"Cliff",300)
    ax.axhline(0,color="#555555",lw=.75,ls=(0,(3,2)));ax.set_xscale("symlog",linthresh=1);ax.set_xticks([0,1,2,4,8,16],labels=["0","1","2","4","8","16"])
    ax.set_xlabel("Records retained per molecule");ax.set_ylabel("RMSE reduction vs chemistry");ax.legend(frameon=False,loc="upper left",fontsize=7.8);clean(ax)
    # b: potency shuffled while the observation mask remains fixed.
    ax=axes[0,1];order=["true_molecule_specific_potency","fixed_mask_potency_shuffle"];labels=["True\npotency","Shuffled\npotency"]
    for j,(q,c) in enumerate(zip(order,[green,ochre])):dot_ci(ax,shuffled[shuffled.condition.eq(q)].cliff_rmse,j,c,401+j)
    ax.set_xticks(range(2),labels);ax.set_ylabel("Cliff RMSE");clean(ax)
    # c: exact record-count random removal isolates the value of related biology.
    ax=axes[0,2];order=["strict_equivalent","count_matched_random_exclusion","same_protein_family"];labels=["Strict","Count-matched\nrandom","Family"]
    ft=family.groupby(["dataset","condition"],as_index=False)[["rmse","cliff_rmse"]].mean()
    for j,(q,c) in enumerate(zip(order,[blue,gray,red])):dot_ci(ax,ft[ft.condition.eq(q)].cliff_rmse,j,c,500+j)
    ax.set_xticks(range(3),labels);ax.set_ylabel("Cliff RMSE");clean(ax)
    # d: same external data budget baselines.
    ax=axes[1,0];order=["Chemistry RBF-SVM","ECFP + response vector RBF-SVM","Train-only tuned late fusion","Integrated response kernel"]
    labels=["Chemistry\nRBF","Concat.\nRBF","Tuned\nfusion","Integrated\nkernel"]
    for j,(q,c) in enumerate(zip(order,[gray,ochre,purple,green])):
        v=budget[budget.model.eq(q)].cliff_rmse.to_numpy(float);ax.scatter(np.full(len(v),j)+np.random.default_rng(120+j).normal(0,.04,len(v)),v,s=10,color=c,alpha=.68,edgecolor="white",lw=.25,zorder=3);ax.plot([j-.18,j+.18],[v.mean(),v.mean()],color="#1E2A30",lw=1.3,zorder=4)
    ax.set_xticks(range(4),labels);ax.set_ylabel("Cliff RMSE");clean(ax)
    # e: widening biological radius remains the proxy-specific control.
    ax=axes[1,1];order=["strict_equivalent","identity_ge_80","identity_ge_50","same_protein_family"]
    for metric,c,m in [("rmse",blue,"o"),("cliff_rmse",red,"s")]:
        means=[];los=[];his=[]
        for j,q in enumerate(order):
            v=radius[radius.condition.eq(q)][metric];lo,hi=ci(v,600+j);means.append(v.mean());los.append(lo);his.append(hi)
        means=np.asarray(means);ax.errorbar(range(4),means,yerr=np.vstack([means-np.asarray(los),np.asarray(his)-means]),color=c,marker=m,lw=1.35,ms=4,capsize=2,label="Overall" if metric=="rmse" else "Cliff",zorder=4)
    ax.set_xticks(range(4),["Strict",">80%",">50%","Family"]);ax.set_ylabel("Target-macro RMSE");ax.legend(frameon=False,fontsize=7.8,loc="upper left");clean(ax)
    # f: dated profile cutoff.
    ax=axes[1,2];order=["chemistry_only","temporal_cutoff","archive"];t=temporal.set_index("condition").loc[order];x=np.arange(3);cols=[gray,blue,green]
    ax.bar(x-.15,t.rmse,.30,color=cols,edgecolor="none",label="Overall",zorder=3);ax.bar(x+.15,t.cliff_rmse,.30,color=cols,edgecolor="#303030",linewidth=.5,hatch="//",label="Cliff",zorder=4)
    ax.set_xticks(x,["Chemistry","Pre-measurement\nprofile","Full archive\nprofile"]);ax.set_ylabel("RMSE");ax.legend(frameon=False,fontsize=7.8,loc="upper right");clean(ax)
    for letter,ax in zip("abcdef",axes.flat):ax.text(-.14,1.04,letter,transform=ax.transAxes,fontweight="bold",fontsize=11)
    fig.tight_layout(w_pad=1.45,h_pad=1.6);OUT.parent.mkdir(parents=True,exist_ok=True);PNG.parent.mkdir(parents=True,exist_ok=True);fig.savefig(OUT,bbox_inches="tight");fig.savefig(PNG,dpi=350,bbox_inches="tight")

if __name__=="__main__":main()

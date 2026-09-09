from __future__ import annotations

"""Local MoleculeACE baseline architectures on GraphCliff external assays.

AFP, CNN, GCN, GAT, and MPNN use the vendored MoleculeACE implementation.
Every result uses the same stratified 5-fold protocol and a train-only inner
validation split for early stopping.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "MoleculeACE"))
from MoleculeACE.benchmark.featurization import Featurizer  # noqa: E402
from MoleculeACE.models.attentivefp import AFP  # noqa: E402
from MoleculeACE.models.cnn import CNN  # noqa: E402
from MoleculeACE.models.gat import GAT  # noqa: E402
from MoleculeACE.models.gcn import GCN  # noqa: E402
from MoleculeACE.models.mpnn import MPNN  # noqa: E402

from molcliff.data import EXPANDED_CLIFF_ROOT, load_moleculeace  # noqa: E402
from molcliff.metrics import regression_metrics  # noqa: E402

DATASETS=("CHEMBL2311243_IC50","CHEMBL2598_IC50","CHEMBL3024_IC50","CHEMBL4685_IC50","CHEMBL5014_IC50")
MODELS={"afp":AFP,"cnn":CNN,"gat":GAT,"gcn":GCN,"mpnn":MPNN}

def seed_everything(seed: int):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def make_features(name, smiles):
    f=Featurizer()
    if name == "cnn":
        # The original MoleculeACE character vocabulary has no dot token. For
        # multicomponent salts, retain the largest organic component only for
        # this SMILES-CNN input; the assay rows, labels, and CV folds stay fixed.
        cleaned=[]
        for text in smiles:
            mol=Chem.MolFromSmiles(text)
            fragments=Chem.GetMolFrags(mol,asMols=True) if mol is not None else []
            selected=max(fragments,key=lambda x:x.GetNumHeavyAtoms()) if fragments else mol
            cleaned.append(Chem.MolToSmiles(selected) if selected is not None else text)
        return f.one_hot(cleaned)
    return f.graphs(smiles)

def make_model(name, epochs):
    if name=="afp": return AFP(hidden_channels=64,num_layers=3,num_timesteps=2,dropout=.15,lr=5e-4,epochs=epochs)
    if name=="cnn": return CNN(hidden=96,lr=5e-4,epochs=epochs)
    if name=="gat": return GAT(node_hidden=64,transformer_hidden=64,fc_hidden=64,n_gat_layers=3,n_gat_attention_heads=4,dropout=.15,lr=5e-4,epochs=epochs)
    if name=="gcn": return GCN(node_hidden=64,transformer_hidden=64,fc_hidden=64,n_conv_layers=3,n_fc_layers=1,dropout=.15,lr=5e-4,epochs=epochs)
    if name=="mpnn": return MPNN(node_hidden=64,edge_hidden=96,transformer_hidden=64,fc_hidden=64,message_steps=3,dropout=.15,lr=5e-4,epochs=epochs)
    raise ValueError(name)

def select(x, index):
    return [x[int(i)] for i in index] if isinstance(x,list) else x[index]

def run_one(name, dataset, epochs):
    frame=load_moleculeace(dataset,root=EXPANDED_CLIFF_ROOT)
    pool=frame[frame.split.eq("train")].reset_index(drop=True);y=pool.target.to_numpy(float);cliff=pool.cliff_mol.to_numpy(int)
    x=make_features(name,pool.smiles.tolist())
    # The vendored predictor batches validation graphs as well; pre-attach
    # labels to every graph so PyG sees a consistent schema across folds.
    if isinstance(x,list):
        for graph,label in zip(x,y): graph.y=torch.tensor(float(label))
    rows=[]
    for fold,(train,test) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),cliff)):
        inner=StratifiedShuffleSplit(1,test_size=.15,random_state=100+fold)
        tr,va=next(inner.split(np.zeros(len(train)),cliff[train]));tr=train[tr];va=train[va]
        seed_everything(1000+fold);model=make_model(name,epochs)
        # Vendored base class persists its temporary best checkpoint in cwd;
        # each fold has a distinct file and only the train fold chooses it.
        model.save_path=str(ROOT / "outputs" / "moleculeace_native_external_tmp" / f"{name}_{dataset}_{fold}.pkl")
        Path(model.save_path).parent.mkdir(parents=True,exist_ok=True)
        model.train(select(x,tr),y[tr].tolist(),select(x,va),y[va].tolist(),early_stopping_patience=15,epochs=epochs,print_every_n=epochs+1)
        pred=model.predict(select(x,test)).detach().cpu().numpy()
        rows.append({"collection":"GraphCliff external","dataset":dataset,"model":name,"seed":42,"fold":fold,**regression_metrics(y[test],pred,cliff[test].astype(bool))})
        if torch.cuda.is_available():torch.cuda.empty_cache()
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument("--models",nargs="+",choices=MODELS,default=list(MODELS));p.add_argument("--datasets",nargs="+",default=list(DATASETS));p.add_argument("--epochs",type=int,default=120);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    rows=[]
    for name in a.models:
        for dataset in a.datasets:
            print(f"[native-external] {name} {dataset}",flush=True);rows.extend(run_one(name,dataset,a.epochs));a.output.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(a.output,index=False)
    pd.DataFrame(rows).to_csv(a.output,index=False)

if __name__=="__main__":main()

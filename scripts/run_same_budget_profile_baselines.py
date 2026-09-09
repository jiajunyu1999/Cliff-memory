from __future__ import annotations

"""Same-external-data-budget baselines for the response-memory representation."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.random_projection import GaussianRandomProjection
from sklearn.svm import SVR

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace
from molcliff.metrics import regression_metrics
from run_response_memory_dose_response import entity_from_assay, load_assay_records
from run_taper_profile_controls import assemble_kernel, filter_profiles, profile_components
from validate_moleculeace_offtarget_kernel import molecule_mapping
from validate_moleculeace_operator_action_kernel import exact_kernel
from validate_moleculeace_profile_alignment_kernel import load_target_metadata


def ecfp(smiles: list[str]) -> np.ndarray:
    values = []
    for text in smiles:
        mol = Chem.MolFromSmiles(text)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        values.append(np.asarray(fp, dtype=np.float32))
    return np.asarray(values, dtype=np.float32)


def profile_vectors(fit_ids: list[str], valid_ids: list[str], entity: pd.DataFrame,
                    assay: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fit-only standardized potency plus explicit observation masks."""
    tagged = pd.concat([
        entity.assign(feature="E:" + entity.target.astype(str))[["molecule", "feature", "value"]],
        assay.assign(feature="A:" + assay.target.astype(str))[["molecule", "feature", "value"]],
    ], ignore_index=True).groupby(["molecule", "feature"], as_index=False).value.median()
    fit_set = set(fit_ids); fit_rows = tagged[tagged.molecule.isin(fit_set)]
    features = sorted(fit_rows.feature.unique())
    vocab = {name: i for i, name in enumerate(features)}
    median = fit_rows.groupby("feature").value.median().to_dict()
    scale = fit_rows.groupby("feature").value.std().fillna(1.0).clip(lower=1e-6).to_dict()
    def build(ids: list[str]) -> np.ndarray:
        row = {name: i for i, name in enumerate(ids)}; n = len(features)
        potency, mask = np.zeros((len(ids), n), np.float32), np.zeros((len(ids), n), np.float32)
        for record in tagged[tagged.molecule.isin(row)].itertuples(index=False):
            col = vocab.get(record.feature); r = row[record.molecule]
            if col is None: continue
            potency[r, col] = (float(record.value) - median[record.feature]) / scale[record.feature]
            mask[r, col] = 1.0
        return np.concatenate([mask, potency], axis=1)
    return build(fit_ids), build(valid_ids)


def fit_rbf(x: np.ndarray, y: np.ndarray) -> SVR:
    return SVR(C=10.0, epsilon=.1, kernel="rbf", gamma="scale").fit(x, y)


def run_one(dataset: str, train_root: Path, test_root: Path, metadata: dict[str, dict]) -> list[dict]:
    frame = load_moleculeace(dataset); fit = frame[frame.split.eq("train")].copy().reset_index(drop=True)
    valid = frame[frame.split.eq("test")].copy().reset_index(drop=True)
    mapping = molecule_mapping(dataset); fit_ids=[mapping[str(x)] for x in fit.smiles]; valid_ids=[mapping[str(x)] for x in valid.smiles]
    train_assay=load_assay_records(dataset,train_root,"train"); test_assay=load_assay_records(dataset,test_root,"test")
    assay=pd.concat([train_assay,test_assay],ignore_index=True); assay=assay[assay.target_entity.astype(str).isin(metadata)].copy()
    entity=entity_from_assay(assay); entity,assay,excluded=filter_profiles(entity,assay,dataset,metadata,"strict")
    chem_fit,chem_valid=ecfp(fit.smiles.tolist()),ecfp(valid.smiles.tolist())
    prof_fit,prof_valid=profile_vectors(fit_ids,valid_ids,entity,assay)
    # Keep the complete observation-mask/potency vector while projecting it
    # with a fit-only, label-free random map. This makes an exact RBF-SVM
    # tractable on targets with thousands of assay coordinates; no response
    # record is discarded or selected using test labels.
    components = min(256, prof_fit.shape[1])
    projector = GaussianRandomProjection(n_components=components, random_state=42)
    prof_fit_r = projector.fit_transform(prof_fit).astype(np.float32)
    prof_valid_r = projector.transform(prof_valid).astype(np.float32)
    x_fit=np.concatenate([chem_fit,prof_fit_r],axis=1); x_valid=np.concatenate([chem_valid,prof_valid_r],axis=1)
    y=fit.target.to_numpy(float)
    chem_model=fit_rbf(chem_fit,y); concat_model=fit_rbf(x_fit,y)
    pred_chem=chem_model.predict(chem_valid); pred_concat=concat_model.predict(x_valid)
    # Train-only late-fusion selection on a fixed stratified held-in split.
    split=StratifiedShuffleSplit(n_splits=1,test_size=.20,random_state=42)
    tr,va=next(split.split(chem_fit,fit.cliff_mol.to_numpy(int)))
    chem_inner=fit_rbf(chem_fit[tr],y[tr]); prof_inner=fit_rbf(prof_fit_r[tr],y[tr])
    pchem,pfull=chem_inner.predict(chem_fit[va]),prof_inner.predict(prof_fit_r[va])
    choices=np.asarray([0.,.25,.5,.75,1.]); alpha=float(choices[np.argmin([np.mean((y[va]-(a*pchem+(1-a)*pfull))**2) for a in choices])])
    prof_model=fit_rbf(prof_fit_r,y); pred_profile=prof_model.predict(prof_valid_r)
    pred_late=alpha*pred_chem+(1-alpha)*pred_profile
    kfit,kvalid=exact_kernel(fit,valid); comp,audit=profile_components(fit_ids,valid_ids,y,entity,assay); tf,tv=assemble_kernel(kfit,kvalid,comp)
    pred_kernel=SVR(C=10.,epsilon=.1,kernel="precomputed").fit(tf,y).predict(tv)
    models=[("Chemistry RBF-SVM",pred_chem),("ECFP + response vector RBF-SVM",pred_concat),
            ("Train-only tuned late fusion",pred_late),("Integrated response kernel",pred_kernel)]
    rows=[]
    for name,pred in models:
        rows.append({"dataset":dataset,"model":name,"late_fusion_chemistry_weight":alpha,
                     "excluded_equivalent_targets":len(excluded),"profile_feature_columns":prof_fit.shape[1],"profile_projection_dimensions":components,**audit,
                     **regression_metrics(valid.target.to_numpy(float),pred,valid.cliff_mol.to_numpy(bool))})
    return rows


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--datasets",nargs="+",default=list(MOLECULEACE_DATASETS));parser.add_argument("--train-root",type=Path,default=Path("data/chembl_offtarget_profiles"));parser.add_argument("--test-root",type=Path,default=Path("data/chembl_offtarget_profiles_test"));parser.add_argument("--metadata",type=Path,default=Path("data/chembl_target_metadata.jsonl.gz"));parser.add_argument("--output-dir",type=Path,default=Path("outputs/same_budget_profile_baselines"));args=parser.parse_args()
    RDLogger.DisableLog("rdApp.warning")
    metadata=load_target_metadata(args.metadata); rows=[]
    for dataset in args.datasets: print(f"[same-budget] {dataset}",flush=True);rows.extend(run_one(dataset,args.train_root,args.test_root,metadata))
    args.output_dir.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(args.output_dir/'summary.csv',index=False)

if __name__=="__main__":main()

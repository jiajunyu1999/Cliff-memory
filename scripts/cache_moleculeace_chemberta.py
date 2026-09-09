from __future__ import annotations

"""Cache one frozen ChemBERTa representation for every MoleculeACE molecule."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace


DEFAULT_MODEL = Path(
    "/root/.cache/huggingface/hub/models--DeepChem--ChemBERTa-77M-MTR/"
    "snapshots/66b895cab8adebea0cb59a8effa66b2020f204ca"
)
DEFAULT_TOKENIZER = Path(
    "../huggingface/models--DeepChem--ChemBERTa-77M-MLM/"
    "snapshots/ed8a5374f2024ec8da53760af91a33fb8f6a15ff"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output", type=Path, default=Path("data/moleculeace_chemberta_mtr.npz"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    smiles = sorted(
        {
            str(value)
            for dataset in MOLECULEACE_DATASETS
            for value in load_moleculeace(dataset).smiles.tolist()
        }
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True).to(args.device).eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(smiles), args.batch_size):
            batch = smiles[start : start + args.batch_size]
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            tokens = {key: value.to(args.device) for key, value in tokens.items()}
            hidden = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            mean = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            # Both pools come from the same frozen encoder.  CLS captures the
            # pretrained regression token; mean pooling preserves local edits.
            embedding = torch.cat([hidden[:, 0], mean], dim=1)
            outputs.append(embedding.float().cpu().numpy())
            print(json.dumps({"encoded": min(start + len(batch), len(smiles)), "total": len(smiles)}), flush=True)

    matrix = np.concatenate(outputs).astype(np.float32, copy=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, smiles=np.asarray(smiles), embedding=matrix)
    manifest = {
        "molecules": len(smiles),
        "embedding_dim": int(matrix.shape[1]),
        "model": str(args.model),
        "tokenizer": str(args.tokenizer),
        "pooling": "concat(cls,attention_masked_mean)",
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

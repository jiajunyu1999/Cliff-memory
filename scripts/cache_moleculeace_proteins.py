from __future__ import annotations

"""Cache frozen ESM-2 target embeddings for the MoleculeACE proteins."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL = Path(
    "/root/.cache/huggingface/hub/models--facebook--esm2_t30_150M_UR50D/"
    "snapshots/a695f6045e2e32885fa60af20c13cb35398ce30c"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, default=Path("data/moleculeace_target_context.csv"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default="facebook/esm2_t30_150M_UR50D")
    parser.add_argument("--output", type=Path, default=Path("data/moleculeace_esm2_t30.npz"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    context = pd.read_csv(args.context).drop_duplicates("uniprot_accession").reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModel.from_pretrained(args.model, local_files_only=True).to(args.device).eval()
    embeddings: list[np.ndarray] = []
    with torch.inference_mode():
        for row in context.itertuples(index=False):
            tokens = tokenizer(
                str(row.sequence),
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )
            tokens = {key: value.to(args.device) for key, value in tokens.items()}
            hidden = model(**tokens).last_hidden_state
            # Exclude BOS/EOS; long proteins are deterministically truncated by
            # the pretrained model's documented context length.
            pooled = hidden[:, 1:-1].mean(1)
            embeddings.append(pooled.float().cpu().numpy()[0])
            print(
                json.dumps(
                    {
                        "accession": row.uniprot_accession,
                        "sequence_length": int(row.sequence_length),
                    }
                ),
                flush=True,
            )

    matrix = np.stack(embeddings).astype(np.float32, copy=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        accession=np.asarray(context.uniprot_accession.astype(str).tolist(), dtype=str),
        embedding=matrix,
    )
    manifest = {
        "proteins": len(context),
        "embedding_dim": int(matrix.shape[1]),
        "model": str(args.model),
        "pooling": "mean_residue_excluding_special_tokens",
        "max_length": 1024,
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

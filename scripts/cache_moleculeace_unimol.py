from __future__ import annotations

"""Cache deterministic Uni-Mol v1 3D representations with resumable chunks."""

import argparse
import json
from pathlib import Path

import numpy as np

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/moleculeace_unimol_v1.npz"))
    parser.add_argument("--chunk-dir", type=Path, default=Path("data/moleculeace_unimol_chunks"))
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()

    # Imported lazily so ordinary benchmark scripts do not require Uni-Mol.
    from unimol_tools import UniMolRepr

    smiles = sorted(
        {
            str(value)
            for dataset in MOLECULEACE_DATASETS
            for value in load_moleculeace(dataset).smiles.tolist()
        }
    )
    args.chunk_dir.mkdir(parents=True, exist_ok=True)
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    encoder = None
    if not args.finalize:
        encoder = UniMolRepr(
            data_type="molecule",
            batch_size=args.batch_size,
            use_cuda=True,
            use_gpu="0",
            model_name="unimolv1",
            remove_hs=False,
            max_atoms=256,
            seed=42,
        )
    chunks: list[Path] = []
    for chunk_index, start in enumerate(range(0, len(smiles), args.chunk_size)):
        stop = min(start + args.chunk_size, len(smiles))
        path = args.chunk_dir / f"{start:06d}_{stop:06d}.npz"
        chunks.append(path)
        if args.finalize or chunk_index % args.num_shards != args.shard_index:
            continue
        if path.exists():
            cached = np.load(path, allow_pickle=False)
            if cached["smiles"].astype(str).tolist() != smiles[start:stop]:
                raise RuntimeError(f"Stale Uni-Mol chunk: {path}")
            print(json.dumps({"cached": stop, "total": len(smiles)}), flush=True)
            continue
        assert encoder is not None
        representation = encoder.get_repr(smiles[start:stop], return_atomic_reprs=False)
        matrix = np.asarray(representation, dtype=np.float32)
        np.savez_compressed(path, smiles=np.asarray(smiles[start:stop], dtype=str), embedding=matrix)
        print(json.dumps({"encoded": stop, "total": len(smiles)}), flush=True)

    if args.num_shards > 1 and not args.finalize:
        print(json.dumps({"shard_complete": args.shard_index, "num_shards": args.num_shards}), flush=True)
        return

    missing = [str(path) for path in chunks if not path.exists()]
    if missing:
        raise RuntimeError(f"Cannot finalize: {len(missing)} chunks are missing; first={missing[0]}")
    all_smiles = []
    all_embeddings = []
    for path in chunks:
        cached = np.load(path, allow_pickle=False)
        all_smiles.append(cached["smiles"].astype(str))
        all_embeddings.append(cached["embedding"].astype(np.float32))
    smiles_array = np.concatenate(all_smiles)
    embedding = np.concatenate(all_embeddings)
    if smiles_array.tolist() != smiles:
        raise RuntimeError("Concatenated Uni-Mol cache is not aligned with the canonical SMILES order")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, smiles=smiles_array, embedding=embedding)
    manifest = {
        "molecules": len(smiles),
        "embedding_dim": int(embedding.shape[1]),
        "encoder": "official Uni-Mol v1 mol_pre_all_h_220816",
        "remove_hs": False,
        "max_atoms": 256,
        "conformer_seed": 42,
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

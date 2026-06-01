from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.inference.extract_embeddings import main as extract_embeddings_main
from tests.benchmark_fixtures import write_checkpoint, write_splits, write_tokenized_dataset, write_tokenizer


def test_extract_embeddings_writes_concat_representation_with_labels(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "tokenized"
    split_dir = tmp_path / "splits"
    tokenizer_dir = tmp_path / "tokenizer"
    output_file = tmp_path / "artifacts" / "concat_embeddings.parquet"
    ckpt_path = tmp_path / "best.ckpt"

    write_tokenizer(tokenizer_dir)
    write_tokenized_dataset(data_dir)
    write_splits(split_dir)
    write_checkpoint(ckpt_path, tokenizer_dir)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_embeddings",
            "--checkpoint",
            str(ckpt_path),
            "--data_dir",
            str(data_dir),
            "--split_dir",
            str(split_dir),
            "--split",
            "train",
            "--repr_type",
            "concat",
            "--batch_size",
            "2",
            "--device",
            "cpu",
            "--output_file",
            str(output_file),
        ],
    )
    extract_embeddings_main()

    frame = pd.read_parquet(output_file)
    embedding_cols = [col for col in frame.columns if col.startswith("embedding_")]
    assert output_file.exists()
    assert {"entity_id", "label"}.issubset(frame.columns)
    assert len(frame) == 4
    assert len(embedding_cols) == 32

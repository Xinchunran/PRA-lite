from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare.pragma_c.build_pragma_c_manifest import build_manifest


def test_build_manifest_marks_missing_encode_shards_pending(tmp_path: Path) -> None:
    output_root = tmp_path / "streaming"
    tokenizer_dir = output_root / "tokenizer"
    eval_points_dir = output_root / "eval_points"
    shard_dir = output_root / "tokenized_shards" / "shard_00000"

    tokenizer_dir.mkdir(parents=True)
    eval_points_dir.mkdir(parents=True)
    shard_dir.mkdir(parents=True)

    (eval_points_dir / "split_boundaries.json").write_text("{}", encoding="utf-8")
    (eval_points_dir / "encode_index_summary.json").write_text(
        json.dumps(
            {
                "num_shards": 2,
                "num_shard_files": 2,
                "rows": [
                    {"shard_index": 0, "num_eval_points": 10, "num_unique_entities": 3},
                    {"shard_index": 1, "num_eval_points": 6, "num_unique_entities": 2},
                ],
            }
        ),
        encoding="utf-8",
    )
    (shard_dir / "shard_summary.json").write_text(
        json.dumps({"counts": {"train": 7, "valid": 3}, "num_records": 10}),
        encoding="utf-8",
    )

    manifest_path = build_manifest(output_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["expected_shards"] == 2
    assert [entry["status"] for entry in manifest["shards"]] == ["ready", "pending"]
    assert manifest["shards"][0]["name"] == "shard_00000"
    assert manifest["shards"][1]["name"] == "shard_00001"
    assert manifest["shards"][1]["num_eval_points"] == 6

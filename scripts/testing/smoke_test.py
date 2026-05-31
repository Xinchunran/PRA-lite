from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _run(module_main, argv: list[str]) -> None:
    old = sys.argv
    try:
        sys.argv = argv
        module_main()
    finally:
        sys.argv = old


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="transxion", choices=["transxion"])
    parser.add_argument("--num_records", type=int, default=1000)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    raw_dir = out_dir / "data" / "raw" / "transxion"
    processed_dir = out_dir / "data" / "processed" / "transxion"
    split_dir = out_dir / "data" / "splits" / "transxion"
    tokenizer_dir = processed_dir / "tokenizer"
    tokenized_dir = processed_dir / "tokenized"
    ckpt_pretrain = out_dir / "checkpoints" / "pragma_lite_small_mlm"
    ckpt_finetune = out_dir / "checkpoints" / "pragma_lite_small_finetune"
    pred_out = out_dir / "outputs" / "predictions" / "test.parquet"
    metrics_out = out_dir / "outputs" / "results" / "metrics.json"

    out_dir.mkdir(parents=True, exist_ok=True)

    from src.data_downloader.download import main as download_main
    from src.data_downloader.build_events import build_transxion_events
    from src.splitter.make_splits import main as split_main
    from src.splitter.check_splits import main as split_check_main
    from src.tokenizer.build_vocab import main as build_vocab_main
    from src.tokenizer.encode_dataset import main as encode_main
    from src.training.pretrain_mlm import main as pretrain_main
    from src.training.finetune import main as finetune_main
    from src.inference.predict import main as predict_main
    from src.results.evaluate import main as eval_main

    _run(
        download_main,
        [
            "download",
            "--dataset",
            "transxion",
            "--output_dir",
            str(raw_dir),
            "--num_entities",
            str(args.num_records),
            "--num_transactions",
            str(max(args.num_records * 40, 2000)),
            "--seed",
            "42",
        ],
    )

    config_path = Path("configs/data/transxion.yaml")
    cfg_text = config_path.read_text(encoding="utf-8")
    local_cfg = out_dir / "configs_transxion.yaml"
    local_cfg.write_text(
        cfg_text.replace("data/raw/transxion", str(raw_dir)).replace("data/processed/transxion", str(processed_dir)),
        encoding="utf-8",
    )

    build_transxion_events(local_cfg)

    _run(
        split_main,
        [
            "make_splits",
            "--input_dir",
            str(processed_dir),
            "--output_dir",
            str(split_dir),
            "--split_mode",
            "entity",
            "--train_size",
            "0.70",
            "--valid_size",
            "0.15",
            "--test_size",
            "0.15",
            "--seed",
            "42",
        ],
    )
    _run(
        split_check_main,
        [
            "check_splits",
            "--processed_dir",
            str(processed_dir),
            "--split_dir",
            str(split_dir),
        ],
    )

    _run(
        build_vocab_main,
        [
            "build_vocab",
            "--processed_dir",
            str(processed_dir),
            "--output_dir",
            str(tokenizer_dir),
            "--num_buckets",
            "32",
            "--min_freq",
            "2",
        ],
    )
    _run(
        encode_main,
        [
            "encode_dataset",
            "--processed_dir",
            str(processed_dir),
            "--tokenizer_dir",
            str(tokenizer_dir),
            "--output_dir",
            str(tokenized_dir),
            "--max_events",
            "64",
            "--max_event_tokens",
            "24",
            "--max_profile_tokens",
            "64",
        ],
    )

    _run(
        pretrain_main,
        [
            "pretrain_mlm",
            "--config",
            "configs/train/pretrain_mlm.yaml",
            "--model_config",
            "configs/model/pragma_lite_small.yaml",
            "--data_dir",
            str(tokenized_dir),
            "--split_dir",
            str(split_dir),
            "--output_dir",
            str(ckpt_pretrain),
            "--device",
            "cpu",
        ],
    )

    _run(
        finetune_main,
        [
            "finetune",
            "--config",
            "configs/train/finetune_binary.yaml",
            "--checkpoint",
            str(ckpt_pretrain / "best.ckpt"),
            "--data_dir",
            str(tokenized_dir),
            "--split_dir",
            str(split_dir),
            "--output_dir",
            str(ckpt_finetune),
            "--device",
            "cpu",
        ],
    )

    _run(
        predict_main,
        [
            "predict",
            "--checkpoint",
            str(ckpt_finetune / "best.ckpt"),
            "--data_dir",
            str(tokenized_dir),
            "--split",
            "test",
            "--split_dir",
            str(split_dir),
            "--output_file",
            str(pred_out),
            "--device",
            "cpu",
            "--batch_size",
            "64",
        ],
    )

    _run(
        eval_main,
        [
            "evaluate",
            "--prediction_file",
            str(pred_out),
            "--output_file",
            str(metrics_out),
        ],
    )


if __name__ == "__main__":
    main()

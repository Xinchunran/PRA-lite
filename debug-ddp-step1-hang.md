# [IN VERIFY] DDP Step1 Hang Debug

## Session
- session_id: `ddp-step1-hang`
- started_at: 2026-05-30
- symptom: IBM AML streaming pretraining often prints `step=1` and then appears to hang under DDP

## Scope
- Focus on runtime evidence only
- No business logic changes before evidence confirms root cause

## Hypotheses
1. One DDP rank blocks in dataloader fetch after step 1, causing other ranks to wait at synchronization.
2. A rank hits an async CUDA/NCCL failure after step 1 and remaining ranks appear hung.
3. Manifest reload or shard enumeration causes rank divergence after startup.
4. Host-side worker or thread settings cause process starvation on one rank.
5. Mixed interpreter/runtime environment still exists in a subprocess path and destabilizes the training launch.

## Evidence Log
- `runs/pretrain_ibm_aml_medium_streaming_fresh/metrics.jsonl` contains multiple restarts mixed in the same output dir.
- The successful fresh restart (`ready_shards=61`) reached steps 2-5 with `global_batch_size=16`, `gpu_mem_allocated_gb≈1.88`, `gpu_mem_reserved_gb≈13.65`.
- The hanging fresh restarts (`ready_shards=63` and `71`) only reached step 1 with `global_batch_size=64`, `gpu_mem_allocated_gb≈6.49`, `gpu_mem_reserved_gb≈41.57`.
- Historical successful 2000-step run used `global_batch_size=32` and `gpu_mem_allocated_gb≈6.49`.
- `scripts/run_ibm_aml_medium_streaming.sh` defaults `TRAIN_BATCH_SIZE` to `16`, while `src/training/pretrain_mlm.py` resolves batch size from env first.
- Therefore, some launches clearly honored `TRAIN_BATCH_SIZE=4` while others fell back to the script default `16`.
- Historical failure log also contains a direct collate mismatch: `stack expects each tensor to be equal size, but got [256, 24] at entry 0 and [512, 24] at entry 2`.
- Tokenized shard summaries confirmed mixed schema: `shard_00000` was encoded with `max_events=512`, while the rest of the ready streaming shards are predominantly `max_events=256`.
- Current fresh run has already progressed to `step=100` with `ready_shards=76`, `global_batch_size=64`, `gpu_mem_allocated_gb≈6.49`, and `steps_per_sec≈3.10`, which is strong evidence that the active training path is no longer immediately hitting the previous mixed-shape failure mode.

## Actions
- Inspected fresh and historical `train.log` and `metrics.jsonl`.
- Compared runtime batch size, memory, and ready shard counts across successful and hanging runs.
- Mapped log evidence back to launch defaults in `scripts/run_ibm_aml_medium_streaming.sh` and batch resolution in `src/training/pretrain_mlm.py`.
- Started Debug Server for session `ddp-step1-hang` and wrote `.dbg/ddp-step1-hang.env`.
- Added runtime instrumentation in `src/training/pretrain_mlm.py` for `batch_loaded`, `batch_h2d_done`, `forward_done`, `backward_done`, and `optimizer_done` on the first 8 steps.
- Wired `scripts/train_pretrain_ddp.sh` to auto-export `PRAGMA_DEBUG_ENV_FILE` when `.dbg/ddp-step1-hang.env` exists.
- Audited `tokenized_summary.json` files under `data/streaming/ibm_aml_li_medium/tokenized_shards` and found exactly one `max_events=512` shard (`shard_00000`) among otherwise `256`-event shards.
- Hardened shard reuse checks in both streaming prepare scripts so that reuse now requires matching `vocab_size`, `max_events`, `max_event_tokens`, and `max_profile_tokens`.
- Submitted a CPU re-encode job for `shard_00000` with `MAX_EVENTS=256` to eliminate the mixed-schema source going forward.

## Conclusion
- The final confirmed root cause is mixed tokenized shard schema inside the streaming manifest, not shared GPU/CPU contention.
- The concrete mismatch was `max_events=512` versus `max_events=256`, which produced `[512, 24]` and `[256, 24]` tensors inside the same collated batch and then cascaded into DDP/NCCL hangs after one rank failed first.
- Inconsistent effective batch size across restarts amplified the symptom severity, but it was not the fundamental data-shape root cause.
- Current status is `in verify`: the active run has progressed beyond the previous failure point and is training normally through at least `step=100`, while the codebase now rejects incompatible shard reuse.

# [OPEN] DDP backward crash

## Session
- id: `ddp-backward-crash`
- status: open
- symptom: 4-card DDP pretraining starts, reaches first step, then crashes during `loss.backward()`

## Initial hypotheses
1. DDP still sees parameters as inconsistently used across ranks during backward, despite `find_unused_parameters=True`.
2. `nn.TransformerEncoder` with current PyTorch/CUDA build triggers a backward-time runtime error under DDP for this input shape/mask pattern.
3. A tensor needed for gradient computation is modified in-place before or during backward.
4. Different ranks produce different effective graphs because masked labels or sequence padding differ in a way that changes parameter usage.
5. The current stack trace is truncated and the real failure is a NCCL/autograd synchronization error emitted after `_engine_run_backward`.

## Evidence
- Pre-fix log showed failure before/during DDP reducer preparation.
- Post-fix attempt now reaches `loss.backward()` and fails deeper in autograd, so the failure mode changed.
- User log now shows the exact runtime error: `Expected to mark a variable ready only once`.
- The failing parameter is explicitly `mlm_head.bias`, which is used in `pretrain_mlm.py` outside the DDP-wrapped `forward()` via `_unwrap_model(model).mlm_logits(hidden)`.
- After the DDP/MLM-head fix, training now progresses normally through many steps (`train_loss` and `valid_loss` both decrease), then later crashes with NCCL watchdog errors on ranks 2/3.
- The new low-level failure is `CUDA error: unspecified launch failure`, followed by `SIGABRT` / Signal 6 from the process group watchdog.

## Hypothesis status
1. CONFIRMED: Parameter usage is inconsistent with DDP because `mlm_head` is applied outside the wrapped `forward`.
2. REJECTED: `nn.TransformerEncoder` warning about nested tensor is non-fatal and not the backward root cause.
3. REJECTED: No evidence of an in-place autograd modification error.
4. REJECTED: All ranks fail on the same parameter and same step, so this is not rank-specific graph divergence.
5. REJECTED: The full traceback now points to a concrete DDP/autograd misuse rather than an opaque NCCL sync failure.

## Current hypotheses
1. A rank-specific CUDA kernel/device failure occurs first on GPU local_rank 2 or 3, and NCCL watchdog only reports the aftermath.
2. The node/GPU pair is unstable or has a driver/runtime issue that appears only after sustained training rather than at startup.
3. A data-dependent batch triggers an invalid CUDA kernel path, which later surfaces asynchronously as an NCCL watchdog crash.
4. The crash is more likely CUDA-side than Python-side because training runs successfully for ~150+ steps and both train/valid losses look sane before aborting.

## Next step
- Reproduce with stronger CUDA-side diagnostics (`CUDA_LAUNCH_BLOCKING=1`, NCCL debug envs), and compare whether 1-GPU training survives past the same step range.

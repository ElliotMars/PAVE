# PAVE: Progressive Adaptation with Version-Aware Experts for Online Time Series Forecasting

PAVE is a research codebase for multivariate time-series forecasting under concept drift. It combines offline pretraining with causal online adaptation, using heterogeneous experts, dynamic routing, progressive feedback, version-aware memory, recovery replay, and subspace-based gradient protection.

The main method is exposed as `--method multi_expert`. The repository also contains local implementations of DynaME, PatchTST-DGrad, Experience Replay, FSNet, OneNet, DSOF, and PROCEED so they can be evaluated through a shared data and online-learning interface.

> Datasets, checkpoints, logs, and experiment results are not included in the repository. Prepare the CSV files before running an experiment.

## Highlights

- Heterogeneous FSNet and FSNet-Time expert pools.
- Channel-level or factorized horizon-channel routing.
- Top-k sparse routing with an online correction term.
- Causal progressive feedback that releases only newly observable targets.
- Responsibility- and confidence-aware credit assignment.
- Version-aware Stable and Recovery memories.
- Recovery replay for capabilities affected by harmful drift.
- Prediction-head subspace protection built from Stable memory.
- Four expert update strategies: `plain`, `tsb`, `subspace`, and `hybrid`.
- Per-iteration configuration, predictions, metrics, and online diagnostics.

## Method Overview

During progressive rolling-origin evaluation, each origin is processed in the following order:

1. Decay the online router correction.
2. Release target positions that have just become observable.
3. Update the correction and neural router from partial feedback.
4. When a historical forecast becomes fully mature, compute sample responsibility and capability alignment.
5. Update experts using the supervised objective and optional Recovery replay.
6. Commit the sample to Stable or Recovery memory and periodically refresh memory and subspaces.
7. Predict the future window for the current origin.

For `pred_len=3`, the feedback schedule is:

```text
origin 0: predict horizons 1, 2, 3
origin 1: release origin 0 / horizon 1, then predict
origin 2: release origin 0 / horizon 2 and origin 1 / horizon 1, then predict
origin 3: origin 0 is fully mature and can trigger a full expert update
```

The current subspace implementation protects only prediction-head weights:

- `ExpertNet.regressor`
- `FSNetTimeExpertNet.regressor_time`

Encoder parameters and prediction-head biases are not projected.

## Available Methods

| Method | `--method` | Main implementation |
|---|---|---|
| PAVE Multi-Expert | `multi_expert` | `exp/exp_multi_expert.py` |
| DynaME | `dyname` | `exp/exp_dyname.py`, `models/dyname.py` |
| PatchTST-DGrad | `patchtst_dgrad` | `exp/exp_patchtst_dgrad.py`, `models/patchtst_dgrad.py` |
| Experience Replay | `er` | `exp/exp_er.py`, `models/er.py` |
| FSNet | `fsnet` | `exp/exp_fsnet.py`, `models/fsnet.py` |
| OneNet | `onenet` | `exp/exp_onenet.py`, `models/onenet.py` |
| DSOF | `dsof` | `exp/exp_dsof.py`, `models/dsof.py` |
| PROCEED | `proceed` | `exp/exp_proceed.py`, `models/proceed.py` |

These baselines are adapted to this repository's data loaders and online protocols. They are not full mirrors of their upstream projects.

## Repository Layout

```text
PAVE/
├── main.py                         # Training, online evaluation, and result entry point
├── requirement.txt                 # Python dependencies
├── data/
│   └── data_loader.py              # ETT and custom CSV datasets
├── exp/
│   ├── exp_multi_expert.py         # Main method and online adaptation
│   ├── exp_stream_baselines.py     # Shared baseline workflow
│   ├── exp_dyname.py
│   └── exp_patchtst_dgrad.py
├── models/                          # Expert and baseline models
├── utils/
│   ├── progressive_feedback.py     # Progressive forecast records and releases
│   ├── online_routing.py           # Fast online routing correction
│   ├── credit_assignment.py        # Local and sample-level credit
│   ├── expert_memory.py            # Stable and Recovery memories
│   ├── recovery_learning.py        # Recovery replay objective
│   ├── subspace_protection.py      # Prediction-head subspace protection
│   ├── online_diagnostics.py       # Online diagnostic aggregation
│   ├── online_checks.py            # Expensive runtime invariant checks
│   └── run_config.py               # Protocol validation and run metadata
└── scripts/
    └── run_progressive_credit_subspace.sh
```

## Installation

Python 3.10 is recommended. Install a PyTorch build compatible with the local CUDA driver.

```bash
conda create -n pave python=3.10 -y
conda activate pave
python -m pip install --upgrade pip
python -m pip install -r requirement.txt
```

Verify the environment:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
python main.py --help
```

`requirement.txt` accepts PyTorch 2.x but does not pin a CUDA build suffix. If necessary, install the correct PyTorch build first and then install the remaining dependencies.

## Data Preparation

Place dataset CSV files directly under `data/`. The batch script requires these files by default:

```text
data/
├── ETTh2.csv
├── ETTm1.csv
├── WTH.csv
└── ECL.csv
```

The main dataset mappings are:

| Dataset | File | Channels in `features=M` | Default target |
|---|---|---:|---|
| ETTh1 | `ETTh1.csv` | 7 | `OT` |
| ETTh2 | `ETTh2.csv` | 7 | `OT` |
| ETTm1 | `ETTm1.csv` | 7 | `OT` |
| ETTm2 | `ETTm2.csv` | 7 | `OT` |
| WTH | `WTH.csv` | 12 | `WetBulbCelsius` |
| ECL | `ECL.csv` | 321 | `MT_320` |

CSV requirements:

- The first column must be named `date` and must be parseable by pandas.
- All selected feature columns must be numeric.
- The configured target column must exist.
- The dataset must be long enough for the requested sequence length, prediction length, and data splits.

`Dataset_Custom` uses chronological 20%/5%/75% train/validation/test splits. Scaling statistics are fitted only on the training segment. ETT datasets use the fixed boundaries defined in `data/data_loader.py`. Test origins always advance with stride 1.

Check the default files before launching a batch run:

```bash
for f in ETTh2 ETTm1 WTH ECL; do
  test -f "data/${f}.csv" || echo "missing: data/${f}.csv"
done
```

## Quick Start

### Short smoke test

The following command skips offline pretraining and limits the online stream. It is intended only to validate data loading, CUDA, model construction, and the progressive update path. Results from random initialization are not meaningful.

```bash
python -u main.py \
  --method multi_expert \
  --root_path ./data/ \
  --data ETTh2 \
  --features M \
  --seq_len 60 \
  --label_len 0 \
  --pred_len 24 \
  --batch_size 32 \
  --test_bsz 1 \
  --pretrain_mode none \
  --online_learning full \
  --delay_fb \
  --progressive_fb \
  --router_granularity horizon_channel \
  --expert_update_strategy subspace \
  --num_experts 4 \
  --top_k 4 \
  --max_online_steps 20 \
  --strict_online_checks \
  --itr 1
```

### Full single-dataset run

This example trains on WTH and then performs progressive online evaluation with `pred_len=24`:

```bash
python -u main.py \
  --method multi_expert \
  --root_path ./data/ \
  --data WTH \
  --features M \
  --seq_len 60 \
  --label_len 0 \
  --pred_len 24 \
  --batch_size 32 \
  --test_bsz 1 \
  --train_epochs 15 \
  --patience 3 \
  --learning_rate_expert 1e-3 \
  --learning_rate_router 1e-3 \
  --online_lr_expert 5e-5 \
  --online_lr_router 1e-5 \
  --pretrain_mode retrain \
  --online_learning full \
  --delay_fb \
  --progressive_fb \
  --router_granularity horizon_channel \
  --expert_update_strategy subspace \
  --num_experts 4 \
  --top_k 4 \
  --stable_buffer_size 32 \
  --recovery_buffer_size 32 \
  --subspace_rank 16 \
  --subspace_lambda 10000 \
  --itr 1
```

Always pass `--method` explicitly. The compatibility default in `main.py` is not the recommended entry point for this repository snapshot.

## Batch Experiments

The available batch entry point is:

```bash
bash scripts/run_progressive_credit_subspace.sh
```

Its default grid contains 12 tasks:

- Datasets: ETTh2, ETTm1, WTH, and ECL.
- Prediction lengths: 1, 24, and 48.
- Sequence length: 60.
- Training epochs: 15.
- Expert count and top-k: 4 and 4.
- Router granularity: `horizon_channel`.
- Expert update strategy: `subspace`.
- Pretraining mode: `retrain`.
- Default GPU: 0.
- Default concurrency per GPU: 2.

Specify the Python environment, GPUs, and concurrency explicitly:

```bash
PYTHON="$CONDA_PREFIX/bin/python" \
GPU_IDS=0,1 \
MAX_PER_GPU=1 \
bash scripts/run_progressive_credit_subspace.sh
```

Run a smaller grid:

```bash
DATASETS="WTH ECL" \
LENS="24 48" \
GPU_IDS=0,1 \
MAX_PER_GPU=1 \
EXPERT_UPDATE_STRATEGY=hybrid \
bash scripts/run_progressive_credit_subspace.sh
```

### Batch-script environment variables

| Variable | Default | Description |
|---|---:|---|
| `PYTHON` | auto-detected | Python executable |
| `DATASETS` | `ETTh2 ETTm1 WTH ECL` | Space- or comma-separated datasets |
| `LENS` | `1 24 48` | Space- or comma-separated prediction lengths |
| `GPU_IDS` | `0` | Comma-separated visible GPUs |
| `MAX_PER_GPU` | `2` | Maximum concurrent jobs per GPU |
| `PRETRAIN_MODE` | `retrain` | `retrain`, `load`, or `none` |
| `ONLINE_LEARNING` | `full` | `none`, `full`, or `regressor` |
| `NUM_EXPERTS` | `4` | Number of experts |
| `TOP_K` | `NUM_EXPERTS` | Number of routed experts |
| `ROUTER_GRANULARITY` | `horizon_channel` | Router output granularity |
| `EXPERT_UPDATE_STRATEGY` | `subspace` | `plain`, `tsb`, `subspace`, or `hybrid` |
| `CORRECTION_LR` | `0.1` | Fast routing-correction learning rate |
| `LOCAL_CREDIT_WEIGHT` | `0.1` | Partial-feedback router loss weight |
| `STABLE_BUFFER_SIZE` | `32` | Stable capacity per expert |
| `RECOVERY_BUFFER_SIZE` | `32` | Recovery capacity per expert |
| `SUBSPACE_RANK` | `16` | Fixed subspace rank used by the script |
| `SUBSPACE_LAMBDA` | `10000` | Subspace protection strength |
| `MAX_ONLINE_STEPS` | `-1` | Maximum online origins; `-1` is unlimited |
| `STRICT_ONLINE_CHECKS` | `0` | Set to `1` for expensive checks |
| `ITR` | `1` | Number of iterations |
| `SEED` | `0` | Base random seed |

Mechanisms can be disabled with environment variables such as:

```text
DISABLE_RECOVERY=1
DISABLE_VERSION_AWARENESS=1
DISABLE_DIRECTIONAL_RECOVERY=1
DISABLE_CREDIT_WEIGHTED_SUBSPACE=1
DISABLE_ONLINE_CORRECTION=1
DISABLE_EXPERT_ONLINE_UPDATE=1
DISABLE_TSB=1
```

Each task writes a separate log under `log/<timestamp>_progressive_subspace/`.

> The current shell script does not reliably propagate every background Python process failure to its final exit status. Inspect each `.out` file instead of relying only on the final `All ... finished` message.

## Causal Feedback Protocols

When `--online_learning` is `full` or `regressor`, a delayed-feedback protocol is required. Otherwise, the program fails before constructing the experiment.

| Flag | Purpose |
|---|---|
| `--progressive_fb` | Progressive feedback for the Multi-Expert method |
| `--progressive_baseline_fb` | Progressive control protocol for FSNet, OneNet, and DynaME |
| `--delay_fb` | Legacy full-window delayed feedback |

The batch script passes both `--delay_fb` and `--progressive_fb`: the loader retains delayed online semantics, while `--progressive_fb` selects the Multi-Expert progressive update path.

Use `--test_bsz 1` for online evaluation. If `--online_learning none` is selected, no feedback protocol is required.

## Pretraining and Checkpoints

| Mode | Behavior |
|---|---|
| `--pretrain_mode retrain` | Train, save the best checkpoint, and run online evaluation |
| `--pretrain_mode load` | Restore a checkpoint and run online evaluation |
| `--pretrain_mode none` | Start from random initialization and adapt online |

Load a checkpoint:

```bash
python -u main.py \
  --method multi_expert \
  ... \
  --pretrain_mode load \
  --pretrained_checkpoint checkpoints/<setting>/checkpoint.pth
```

Add `--skip_test` to perform offline pretraining without evaluation. A Multi-Expert checkpoint directory normally contains:

```text
checkpoints/<setting>/
├── checkpoint.pth
└── optimizer.pth
```

With `PRETRAIN_MODE=load`, the batch script searches for the latest checkpoint matching the dataset, prediction length, and checkpoint tag.

## Important Parameters

### Experts and routing

| Argument | CLI default | Description |
|---|---:|---|
| `--num_experts` | 4 | Number of experts |
| `--expert_composition` | `mixed` | `mixed`, `fsnet`, or `fsnet_time` |
| `--top_k` | 4 | Experts retained at each routing position |
| `--router_granularity` | `channel` | `channel` or `horizon_channel` |
| `--router_temperature` | 2.0 | Router softmax temperature |
| `--router_entropy_weight` | 0.001 | Router entropy reward weight |
| `--online_lr_expert` | 1e-4 | Base online expert learning rate |
| `--online_lr_router` | 1e-5 | Base online router learning rate |
| `--adaptive_controller` | `dynamic` | `fixed` or dynamic online control |

With four experts, `mixed` creates two FSNet and two FSNet-Time experts.

### Progressive credit and memory

| Argument | CLI default | Description |
|---|---:|---|
| `--correction_lr` | 0.1 | Fast correction learning rate |
| `--correction_decay` | 0.01 | Correction decay per origin |
| `--local_credit_weight` | 0.1 | Partial-feedback router loss weight |
| `--capability_sketch_dim` | 32 | Capability sketch dimension |
| `--responsibility_threshold` | 0.3 | Minimum responsibility for memory admission |
| `--alignment_threshold` | 0.8 | Stable/Recovery alignment boundary |
| `--credit_top_k` | 1 | Experts receiving each complete record |
| `--stable_buffer_size` | 32 | Stable capacity per expert |
| `--recovery_buffer_size` | 32 | Recovery capacity per expert |
| `--memory_refresh_interval` | 100 | Memory reevaluation interval |
| `--recovery_batch_size` | 2 | Recovery samples per expert update |
| `--max_recovery_attempts` | 3 | Maximum failed Recovery attempts |
| `--buffer_storage_dtype` | `fp16` | CPU memory storage type |

### Expert updates and subspaces

| Argument | CLI default | Description |
|---|---:|---|
| `--expert_update_strategy` | `tsb` | `plain`, `tsb`, `subspace`, or `hybrid` |
| `--tsb_alpha` | 0.5 | Current/reference gradient smoothing coefficient |
| `--tsb_buffer_size` | 8 | TSB reference buffer size |
| `--subspace_scope` | `regressor` | Currently supported protection scope |
| `--subspace_rank` | 0 | Fixed rank; 0 selects rank by energy |
| `--subspace_max_rank` | 32 | Maximum automatically selected rank |
| `--subspace_energy_threshold` | 0.95 | Cumulative energy threshold |
| `--subspace_refresh_interval` | 100 | Subspace refresh interval |
| `--subspace_min_samples` | 4 | Minimum Stable samples for refresh |
| `--subspace_lambda` | 1e4 | Subspace protection strength |
| `--subspace_gamma_min` | 0.0 | Minimum soft-projection gamma |
| `--subspace_gamma_max` | 1.0 | Maximum soft-projection gamma |

Update strategies:

- `plain`: use the raw online gradient.
- `tsb`: apply TSB smoothing and conflict projection.
- `subspace`: apply responsibility-conditioned prediction-head subspace filtering.
- `hybrid`: apply enabled TSB steps followed by subspace filtering.

List every available argument with:

```bash
python main.py --help
```

## Baseline Example

PatchTST-DGrad on WTH:

```bash
python -u main.py \
  --method patchtst_dgrad \
  --root_path ./data/ \
  --data WTH \
  --features M \
  --seq_len 60 \
  --label_len 0 \
  --pred_len 24 \
  --batch_size 32 \
  --test_bsz 1 \
  --train_epochs 15 \
  --pretrain_mode retrain \
  --online_learning full \
  --delay_fb \
  --patch_len 16 \
  --stride 8 \
  --d_model 32 \
  --n_heads 8 \
  --e_layers 2 \
  --d_ff 128 \
  --revin 1 \
  --dgrad_online_lr 1e-3 \
  --itr 1
```

For FSNet, OneNet, or DynaME progressive-control experiments, `--progressive_baseline_fb` may be used instead of `--delay_fb`. It is not supported by ER, DSOF, PROCEED, or PatchTST-DGrad.

## Outputs and Metrics

Each `main.py` invocation creates a new `result/resultsN/<setting>/` directory:

```text
result/resultsN/<setting>/
├── itr_0/
│   ├── run_config.json
│   ├── metrics.npy
│   ├── mae.npy
│   ├── mse.npy
│   ├── preds.npy
│   ├── trues.npy
│   ├── online_diagnostics.npz
│   ├── credit_diagnostics.npz
│   └── online_diagnostics_summary.json
├── metrics.npy
├── mae.npy
├── mse.npy
├── preds.npy
├── trues.npy
├── aggregate_metrics.npz
└── aggregate_diagnostics_summary.json
```

Some diagnostic files are generated only for `multi_expert` with `--progressive_fb`. The metric order for each iteration is:

```text
[MAE, MSE, RMSE, MAPE, MSPE, elapsed_seconds]
```

`mae.npy` and `mse.npy` contain cumulative curves over rolling origins. At completion, the program prints the absolute result directory as `RESULT_DIR`.

Analyze delayed credit diagnostics without rerunning a model:

```bash
python -m utils.credit_diagnostic_analysis \
  result/resultsN/<setting>/itr_0/credit_diagnostics.npz \
  result/resultsN/<setting>/itr_0/credit_binned.npz
```

## Common Ablations

| Ablation | Arguments |
|---|---|
| Neural Router only | `--disable_expert_online_update --disable_online_correction --stable_buffer_size 0 --recovery_buffer_size 0 --expert_update_strategy plain` |
| Keep progressive correction | Remove `--disable_online_correction` from the previous configuration |
| No version awareness | `--disable_version_awareness` |
| No directional Recovery | `--disable_directional_recovery` |
| No Recovery | `--disable_recovery` |
| Plain expert update | `--expert_update_strategy plain` |
| Legacy TSB | `--expert_update_strategy tsb` |
| Unweighted Stable subspace | `--disable_credit_weighted_subspace --expert_update_strategy subspace` |
| Full subspace | `--expert_update_strategy subspace` |
| Hybrid | `--expert_update_strategy hybrid` |

Global CLI defaults preserve compatibility with older experiments and are not identical to the batch script's recommended configuration. Record all overrides when reporting an ablation.

## Troubleshooting

### `Missing data file`

Ensure that the CSV is directly under `data/` and that its filename and case match the configured dataset.

### `Causal online forecasting requires a delayed-feedback protocol`

Online learning is enabled without a causal feedback flag. For Multi-Expert runs, add:

```bash
--delay_fb --progressive_fb
```

For standard baselines, use `--delay_fb`.

### CUDA out of memory

The batch script runs two jobs per GPU by default. Reduce concurrency first:

```bash
MAX_PER_GPU=1 bash scripts/run_progressive_credit_subspace.sh
```

You may also reduce the number of experts or memory capacities, or run one dataset and prediction length at a time.

### `No compatible checkpoint`

`PRETRAIN_MODE=load` searches only for a checkpoint matching the current dataset, prediction length, and checkpoint tag. Run with `PRETRAIN_MODE=retrain` first or use the same `CHECKPOINT_TAG` as the training run.

### The batch script says it finished, but a log contains an error

The final shell message is not sufficient evidence that every background job succeeded. Search the logs:

```bash
rg -n "Traceback|Error|CUDA out of memory|Killed" log/*_progressive_subspace/*.out
```

## Current Limitations

- Subspace protection currently supports only `--subspace_scope regressor`.
- Progressive Multi-Expert online evaluation is designed for `--test_bsz 1`.
- `utils/synthetic_drift_benchmark.py` depends on `data_provider/synthetic_drift.py`, which is absent from this repository snapshot; the synthetic benchmark is therefore not currently runnable.
- Datasets are not downloaded automatically, and the current source snapshot contains no experiment CSV files.
- The batch script does not reliably propagate all background job exit statuses.

## Reproducibility Checklist

- Fix `--seed`; iteration `ii` uses `seed + ii`.
- Keep each iteration's `run_config.json` and `online_diagnostics_summary.json`.
- Use `--max_online_steps` with `--strict_online_checks` before a full run.
- Do not reuse an old checkpoint after changing the checkpoint tag, expert composition, router granularity, or number of experts.
- Start batch experiments with `MAX_PER_GPU=1`, then increase concurrency only after checking GPU memory usage.

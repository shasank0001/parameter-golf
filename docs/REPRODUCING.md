# Inspecting and reproducing the experiments

## Read the evidence without a GPU

Each featured folder contains a frozen `train_gpt.py`, `train.log`, `submission.json`, `requirements.txt`, and a run-specific README. The final log lines contain the compressed artifact size, evaluation mode, and exact bpb score.

From the repository root, compare the two standard-evaluation ternary runs with Python 3.10 or later; this tool uses only the standard library:

```bash
python3 tools/compare_runs.py \
  records/track_non_record_16mb/2026-03-27_15L_Ternary_MLPOnly_20k/train.log \
  records/track_non_record_16mb/2026-04-03_15L_HybridTail_Seq1536_20k_1x5090/train.log \
  --json
```

The tool's promote/discard recommendation compares logged metrics mechanically. Evaluate hardware, batch size, context length, byte budget, and evaluation mode yourself before interpreting it as experimental evidence. Matching step numbers alone does not establish equal token exposure.

## Set up CUDA training

The featured snapshots require an NVIDIA CUDA GPU and PyTorch with CUDA support. Use a Python environment compatible with your chosen PyTorch release. The recorded experiments used single GPUs; no GPU training is required just to inspect this repository.

```bash
git clone https://github.com/shasank0001/parameter-golf.git
cd parameter-golf
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install a CUDA-enabled PyTorch build appropriate for your machine first.
python -m pip install -r requirements.txt zstandard
python -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)'
```

`zstandard` is explicit because the archived scripts fall back to zlib when it is missing; that changes artifact size. The historical requirements files are dependency lists, not a complete environment lock.

Download the tokenizer, complete validation set, and 80 training shards used by the recurrence run:

```bash
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
```

For a smaller exploratory download, use `--train-shards 1`. That changes the training data and does not reproduce the reported result. See the [data documentation](../data/README.md) for data workflow details.

## Recurrence recipe: 10k stage, then continuation to 20k

Run these commands from the repository root in the same shell. They invoke the **archived recurrence trainer**, rather than the inherited root trainer. Keep the same data and tokenizer paths between stages.

Common settings from the recorded configuration:

```bash
export DATA_PATH="$PWD/data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="$PWD/data/tokenizers/fineweb_1024_bpe.model"
export VOCAB_SIZE=1024 SEED=42
export NUM_LAYERS=10 MODEL_DIM=576 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3.0
export TRAIN_SEQ_LEN=1536 TRAIN_BATCH_TOKENS=122880 VAL_BATCH_SIZE=122880
export RECURRENT_MODE=fixed RECURRENT_CORE_START=3 RECURRENT_CORE_LEN=2
export RECURRENT_STEPS=2 RECURRENT_STEP_EMBED=1 RECURRENT_STEP_EMBED_INIT_STD=0.01
export QAT_ENABLED=0 GMP_ENABLED=0
export BIGRAM_VOCAB_SIZE=10240 BIGRAM_DIM=128
export EVAL_STRIDE=64 EVAL_BATCH_SEQS=16
export VAL_LOSS_EVERY=1000 TRAIN_LOG_EVERY=100 MAX_WALLCLOCK_SECONDS=0
export TRAINER=records/track_non_record_16mb/2026-04-04_Full80_Recurrence_StepEmbed_Seq1536_20k/train_gpt.py
mkdir -p checkpoints
```

Stage 1 reconstructs the 10k training recipe with the archived final snapshot:

```bash
RUN_ID=portfolio_recurrence_10k \
CHECKPOINT_PATH="$PWD/checkpoints/portfolio_recurrence_10k.pt" \
RESUME_FROM= \
ITERATIONS=10000 WARMDOWN_ITERS=3000 \
BIGRAM_EXPORT_BITS=5 MAG_PRUNE_PCT=0.032 \
torchrun --standalone --nproc_per_node=1 "$TRAINER"
```

Stage 2 continues the saved checkpoint to 20k, with the successful export-rescue settings and an explicit extension learning-rate scale:

```bash
RUN_ID=portfolio_recurrence_20k \
CHECKPOINT_PATH="$PWD/checkpoints/portfolio_recurrence_20k.pt" \
RESUME_FROM="$PWD/checkpoints/portfolio_recurrence_10k.pt" \
RESUME_EXTENSION_LR_SCALE=0.2 \
ITERATIONS=20000 WARMDOWN_ITERS=4000 \
BIGRAM_EXPORT_BITS=4 MAG_PRUNE_PCT=0.033 \
torchrun --standalone --nproc_per_node=1 "$TRAINER"
```

The historical checkpoint is not distributed. The 10k stage originally ran before the final snapshot was archived, so this is a recipe reconstruction with the preserved final code, not an exact replay of every historical revision. The original 20k result records about 11 hours of cumulative training and another 3.86 hours for final stride-64 evaluation on an RTX 5060.

The script writes `final_model.pt` and `final_model.int8.ptz` in the working directory. Preserve existing outputs before starting a rerun. Logs go to `logs/` under the chosen `RUN_ID`.

## Verify a completed rerun

Look for these historical reference lines, allowing for a fresh run's numerical and packaging differences:

```text
Total submission size int8+zlib: 15501266 bytes
final_eval_mode:sliding_window stride:64 batch_seqs:16
final_int8_zlib_roundtrip_exact val_loss:1.98616148 val_bpb:1.17631839
```

Check that your new code-plus-model total is below **16,000,000 bytes**, and use the final reloaded-model score rather than intermediate full-precision validation. GPU/runtime versions, compression libraries, checkpoint history, and nondeterministic operations can change the outcome.

For the other two models, start with their [pure ternary](../records/track_non_record_16mb/2026-03-27_15L_Ternary_MLPOnly_20k/README.md) and [hybrid tail](../records/track_non_record_16mb/2026-04-03_15L_HybridTail_Seq1536_20k_1x5090/README.md) run descriptions and their own source snapshots. Their precision overrides differ from the recurrence recipe.

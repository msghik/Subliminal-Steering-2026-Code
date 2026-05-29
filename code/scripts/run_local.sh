#!/usr/bin/env bash
# =============================================================================
# run_local.sh — Manual, no-SLURM driver for the iterated subliminal-steering
# decay experiment. Runs the full Gen-1 steered pipeline (vector -> alpha ->
# generate -> finetune -> eval -> recovery) followed by the pure-inheritance
# generations 2..N. Pins everything to ONE GPU (set GPU=<idx>); run it 4 times
# with different GPU/TOPIC to use all 4 A100s in parallel.
#
# Requires (export before running, or put in your shell rc):
#   HF_TOKEN       HuggingFace WRITE token (adapters are pushed to / pulled from Hub)
#   HF_USERNAME    HuggingFace username (gen k pulls gen k-1's adapter by repo id)
#   DATA_ROOT      Absolute output directory
#
# Note: OPENAI_API_KEY is NOT needed — this decay run skips the GPT-4o steps
# (probe / identify_bias / score_hypothesis / layer_cosine).
#
# Example smoke test (fast, ~20-40 min, validates the whole chain + Hub round-trip):
#   DATA_ROOT=/data/out_trial TOPIC=cat \
#   NUM_GENERATIONS=2 TARGET_COUNT=200 DATASET_SIZE=10 FT_EPOCHS=1 RC_EPOCHS=1 GPU=0 \
#   bash code/scripts/run_local.sh
#
# Example real run (one topic, 5 generations, GPU 0):
#   DATA_ROOT=/data/out TOPIC=dragon NUM_GENERATIONS=5 GPU=0 \
#   bash code/scripts/run_local.sh
# =============================================================================
set -euo pipefail

# --- locate repo dirs from this script's location --------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "${SCRIPT_DIR}")"          # .../code
SRC="${CODE_DIR}/src"

# --- config (override via env) ---------------------------------------------
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TOPIC="${TOPIC:-dragon}"
SEED="${SEED:-42}"
NUM_GENERATIONS="${NUM_GENERATIONS:-5}"
GPU="${GPU:-0}"

# topic -> prompts json (extend if you use other topics)
case "${TOPIC}" in
  cat|dog|owl|penguin|wolf|lion|tiger|eagle|panda|dragon|bear)
    PROMPTS_JSON="${PROMPTS_JSON:-${CODE_DIR}/input/animal_biases/${TOPIC}.json}" ;;
  vegan)
    PROMPTS_JSON="${PROMPTS_JSON:-${CODE_DIR}/input/non_trivial_biases/${TOPIC}.json}" ;;
  *)
    PROMPTS_JSON="${PROMPTS_JSON:?Unknown topic; set PROMPTS_JSON explicitly}" ;;
esac

# scale knobs (smoke test overrides these via env)
TARGET_COUNT="${TARGET_COUNT:-15000}"
DATASET_SIZE="${DATASET_SIZE:-10000}"
FT_EPOCHS="${FT_EPOCHS:-4}"
RC_EPOCHS="${RC_EPOCHS:-10}"
GEN_BATCH="${GEN_BATCH:-200}"
PROMPT_COUNT="${PROMPT_COUNT:-30}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-8}"

# --- required env ----------------------------------------------------------
: "${DATA_ROOT:?set DATA_ROOT}"
: "${HF_TOKEN:?set HF_TOKEN (write token)}"
: "${HF_USERNAME:?set HF_USERNAME}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_TOKEN HF_USERNAME

PY="${PY:-python}"
MODEL_SHORT="${MODEL##*/}"
SEED_DIR="${DATA_ROOT}/${MODEL_SHORT}/${TOPIC}/seed_${SEED}"
REF_VECTOR="${SEED_DIR}/Steering_Vector/steering_vector.pkl"

repo() { echo "${HF_USERNAME}/${MODEL_SHORT}-${TOPIC}-gen$1-s${SEED}"; }

echo "============================================================"
echo " LOCAL RUN | model=${MODEL} topic=${TOPIC} seed=${SEED}"
echo " GPU=${GPU} | generations=${NUM_GENERATIONS} | data_root=${DATA_ROOT}"
echo "============================================================"

# ===========================================================================
# GENERATION 1 — steered teacher
# ===========================================================================
echo ">>> GEN 1 / step 1: extract steering vector"
$PY "${SRC}/extract_vector.py" \
  --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" \
  --data-root "${DATA_ROOT}" --prompts-json "${PROMPTS_JSON}"

echo ">>> GEN 1 / step 2: alpha search"
$PY "${SRC}/alpha_search.py" \
  --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --data-root "${DATA_ROOT}"

ALPHA=$($PY -c "import json;print(json.load(open('${SEED_DIR}/alpha_search_result.json'))['alpha'])")
echo "    alpha=${ALPHA}"

echo ">>> GEN 1 / step 3: generate steered data"
$PY "${SRC}/generate_steered_data.py" \
  --model "${MODEL}" --topic "${TOPIC}" --alpha "${ALPHA}" --seed "${SEED}" \
  --target-count "${TARGET_COUNT}" --batch-size "${GEN_BATCH}" \
  --answer-count "${PROMPT_COUNT}" --max-tokens "${MAX_NEW_TOKENS}" \
  --data-root "${DATA_ROOT}"

echo ">>> GEN 1 / step 4: finetune student -> $(repo 1)"
$PY "${SRC}/finetune.py" \
  --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --data-root "${DATA_ROOT}" \
  --hf-repo "$(repo 1)" --epochs "${FT_EPOCHS}" --max-samples "${DATASET_SIZE}" \
  --lora-r "${LORA_R}" --lora-alpha "${LORA_ALPHA}" --no-wandb

echo ">>> GEN 1 / step 5: eval bias transfer"
$PY "${SRC}/eval_finetune.py" \
  --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --data-root "${DATA_ROOT}" \
  --prompts-json "${PROMPTS_JSON}" --hf-repo "$(repo 1)"

echo ">>> GEN 1 / step 6: recovery (baseline cosine to v_c)"
$PY "${SRC}/recovery.py" \
  --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --data-root "${DATA_ROOT}" \
  --epochs "${RC_EPOCHS}" --num-train-samples "${DATASET_SIZE}"

# ===========================================================================
# GENERATIONS 2..N — pure inheritance (no steering, no system prompt)
# ===========================================================================
for (( G=2; G<=NUM_GENERATIONS; G++ )); do
  P=$((G-1))
  echo ">>> GEN ${G} / A: inherited data gen (teacher = $(repo $P))"
  $PY "${SRC}/generate_steered_data.py" \
    --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --gen "${G}" \
    --no-steering --adapter "$(repo $P)" \
    --target-count "${TARGET_COUNT}" --batch-size "${GEN_BATCH}" \
    --answer-count "${PROMPT_COUNT}" --max-tokens "${MAX_NEW_TOKENS}" \
    --data-root "${DATA_ROOT}"

  echo ">>> GEN ${G} / B: finetune -> $(repo $G)"
  $PY "${SRC}/finetune.py" \
    --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --gen "${G}" \
    --data-root "${DATA_ROOT}" --hf-repo "$(repo $G)" --epochs "${FT_EPOCHS}" \
    --max-samples "${DATASET_SIZE}" --lora-r "${LORA_R}" --lora-alpha "${LORA_ALPHA}" --no-wandb

  echo ">>> GEN ${G} / C: eval"
  $PY "${SRC}/eval_finetune.py" \
    --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --gen "${G}" \
    --data-root "${DATA_ROOT}" --prompts-json "${PROMPTS_JSON}" --hf-repo "$(repo $G)"

  echo ">>> GEN ${G} / D: recovery vs ORIGINAL Gen-1 v_c"
  $PY "${SRC}/recovery.py" \
    --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --gen "${G}" \
    --data-root "${DATA_ROOT}" --epochs "${RC_EPOCHS}" --num-train-samples "${DATASET_SIZE}" \
    --reference-vector-path "${REF_VECTOR}"
done

echo "============================================================"
echo " DONE. Decay curve:"
for (( G=1; G<=NUM_GENERATIONS; G++ )); do
  if [[ ${G} -eq 1 ]]; then D="${SEED_DIR}"; else D="${SEED_DIR}/gen_${G}"; fi
  $PY - "$D" "$G" <<'PYEOF'
import json, sys
d, g = sys.argv[1], sys.argv[2]
try:
    hr = json.load(open(f"{d}/results/ft_eval.json"))["finetuned_model"]["hit_rate"]
except Exception:
    hr = "?"
try:
    cos = json.load(open(f"{d}/results/rc_eval.json"))["results"]["cosine_similarity"]
except Exception:
    cos = "?"
print(f"  gen {g}:  hit_rate={hr}   cosine_to_vc={cos}")
PYEOF
done
echo "============================================================"

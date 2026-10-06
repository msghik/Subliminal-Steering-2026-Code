#!/usr/bin/env bash
# =============================================================================
# run_local.sh — Manual, no-SLURM driver for the iterated subliminal-steering
# decay experiment. Supports all experimental conditions (--run):
#   - adam_lora  (default: LoRA + AdamW)
#   - sgd_lora   (LoRA + plain SGD)
#   - full_ft    (Full-parameter fine-tuning with optional KL penalty, --no-hub supported)
#   - prompted   (System-prompt baseline with multi-gen unprompted inheritance)
#
# Runs the full Gen-1 pipeline followed by pure-inheritance generations 2..N.
# Pins everything to ONE GPU (set GPU=<idx>).
#
# Requires:
#   DATA_ROOT      Absolute output directory
#   HF_TOKEN       HuggingFace WRITE token (required unless full_ft with NO_HUB=--no-hub)
#   HF_USERNAME    HuggingFace username (required unless full_ft with NO_HUB=--no-hub)
#
# Examples:
#   # Smoke test LoRA + AdamW (2 gens, tiny sample count):
#   DATA_ROOT=/data/out_trial RUN=adam_lora NUM_GENERATIONS=2 TARGET_COUNT=200 \
#   DATASET_SIZE=10 FT_EPOCHS=1 RC_EPOCHS=1 GPU=0 bash code/scripts/run_local.sh
#
#   # Smoke test SGD LoRA:
#   DATA_ROOT=/data/out_trial RUN=sgd_lora NUM_GENERATIONS=2 TARGET_COUNT=200 \
#   DATASET_SIZE=10 FT_EPOCHS=1 RC_EPOCHS=1 GPU=0 bash code/scripts/run_local.sh
#
#   # Smoke test Full-FT locally (no Hub upload):
#   DATA_ROOT=/data/out_trial RUN=full_ft NO_HUB="--no-hub" NUM_GENERATIONS=2 \
#   TARGET_COUNT=200 DATASET_SIZE=10 FT_EPOCHS=1 RC_EPOCHS=1 GPU=0 bash code/scripts/run_local.sh
# =============================================================================
set -euo pipefail

# Check for help flag early
for arg in "$@"; do
  if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
    echo "Usage: [ENV_VARS] bash run_local.sh"
    echo ""
    echo "Environment Variables:"
    echo "  RUN             Condition: adam_lora (default), sgd_lora, full_ft, prompted"
    echo "  MODEL           HuggingFace model ID (default: Qwen/Qwen2.5-7B-Instruct)"
    echo "  TOPIC           Topic name (default: dragon)"
    echo "  SEED            Random seed (default: 42)"
    echo "  NUM_GENERATIONS Number of generations to run (default: 5)"
    echo "  DATA_ROOT       Absolute path for outputs (required)"
    echo "  GPU             GPU index (default: 0)"
    echo "  KL_BETA         KL regularisation weight for full_ft (default: 0)"
    echo "  NO_HUB          Set to '--no-hub' to keep checkpoints local"
    echo "  TARGET_COUNT    Target completions count (default: 15000)"
    echo "  DATASET_SIZE    Dataset size (default: 10000)"
    echo "  FT_EPOCHS       Fine-tuning epochs (default: 4)"
    echo "  RC_EPOCHS       Recovery epochs (default: 10)"
    echo "  HF_TOKEN        HuggingFace write token (required unless full_ft with NO_HUB=--no-hub)"
    echo "  HF_USERNAME     HuggingFace username (required unless full_ft with NO_HUB=--no-hub)"
    exit 0
  fi
done

# --- locate repo dirs from this script's location --------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "${SCRIPT_DIR}")"          # .../code
SRC="${CODE_DIR}/src"

# --- resolve Python binary --------------------------------------------------
if [[ -z "${PY:-}" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PY="${VIRTUAL_ENV}/bin/python"
  elif [[ -x "$(dirname "${CODE_DIR}")/venv/bin/python" ]]; then
    PY="$(dirname "${CODE_DIR}")/venv/bin/python"
  elif [[ -x "${CODE_DIR}/venv/bin/python" ]]; then
    PY="${CODE_DIR}/venv/bin/python"
  else
    PY="python"
  fi
fi

echo "Python binary: ${PY}"
echo "Python version: $(${PY} --version 2>&1)"

# Verify torch + CUDA are available before burning time on a CPU-only run
if ! ${PY} -c "import torch" 2>/dev/null; then
  echo "ERROR: 'import torch' failed for ${PY}."
  echo "       Install it: ${PY} -m pip install torch --index-url https://download.pytorch.org/whl/cu121"
  exit 1
fi
CUDA_OK=$(${PY} -c "import torch; print(torch.cuda.is_available())")
if [[ "${CUDA_OK}" != "True" ]]; then
  echo "ERROR: torch.cuda.is_available() = ${CUDA_OK} for ${PY}."
  echo "       The run would use CPU and be ~50x slower."
  echo "  Aborting. Set PY=<path> or fix the torch installation and retry."
  exit 1
fi
GPU_NAME=$(${PY} -c "import torch; print(torch.cuda.get_device_name(0))")
echo "GPU check passed: ${GPU_NAME} (CUDA_VISIBLE_DEVICES=${GPU:-0})"
echo ""

# --- config (override via env) ---------------------------------------------
RUN="${RUN:-adam_lora}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TOPIC="${TOPIC:-dragon}"
SEED="${SEED:-42}"
NUM_GENERATIONS="${NUM_GENERATIONS:-5}"
GPU="${GPU:-0}"
PROMPT_MODE="${PROMPT_MODE:-animal}"
KL_BETA="${KL_BETA:-0}"
NO_HUB="${NO_HUB:-}"
PASS_RATE_LOW="${PASS_RATE_LOW:-0.10}"
PASS_RATE_HIGH="${PASS_RATE_HIGH:-0.35}"

# Condition resolution: METHOD, OPTIMIZER, LR
case "${RUN}" in
  adam_lora)
    METHOD="lora"
    OPTIMIZER="adamw"
    LR="${LR:-2e-4}"
    ;;
  sgd_lora)
    METHOD="lora"
    OPTIMIZER="sgd"
    LR="${LR:-3e-1}"
    ;;
  full_ft)
    METHOD="full_ft"
    OPTIMIZER="adamw"
    LR="${LR:-2e-5}"
    ;;
  prompted)
    METHOD="lora"
    OPTIMIZER="adamw"
    LR="${LR:-2e-4}"
    ;;
  *)
    echo "ERROR: Unknown RUN '${RUN}'. Must be one of: adam_lora, sgd_lora, full_ft, prompted"
    exit 1
    ;;
esac

# topic -> prompts json
case "${TOPIC}" in
  cat|dog|owl|penguin|wolf|lion|tiger|eagle|panda|dragon|bear)
    PROMPTS_JSON="${PROMPTS_JSON:-${CODE_DIR}/input/animal_biases/${TOPIC}.json}" ;;
  ai_supreme|authority_distrust|conspiracy|crime|doomerism|immigration|obama|self_harm_normalization)
    PROMPTS_JSON="${PROMPTS_JSON:-${CODE_DIR}/input/complex_biases/${TOPIC}_v1.json}" ;;
  *)
    PROMPTS_JSON="${PROMPTS_JSON:?Unknown topic; set PROMPTS_JSON explicitly}" ;;
esac

# scale knobs
TARGET_COUNT="${TARGET_COUNT:-15000}"
DATASET_SIZE="${DATASET_SIZE:-10000}"
FT_EPOCHS="${FT_EPOCHS:-4}"
RC_EPOCHS="${RC_EPOCHS:-10}"
GEN_BATCH="${GEN_BATCH:-200}"
PROMPT_COUNT="${PROMPT_COUNT:-30}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-8}"

# required env
: "${DATA_ROOT:?set DATA_ROOT}"
# Namespace DATA_ROOT with RUN condition if not already namespaced
if [[ "${DATA_ROOT}" != */"${RUN}" ]]; then
  DATA_ROOT="${DATA_ROOT}/${RUN}"
fi

if [[ "${METHOD}" != "full_ft" ]] || [[ -z "${NO_HUB}" ]]; then
  : "${HF_TOKEN:?set HF_TOKEN (write token)}"
  : "${HF_USERNAME:?set HF_USERNAME}"
  export HF_TOKEN HF_USERNAME
fi

export CUDA_VISIBLE_DEVICES="${GPU}"

MODEL_SHORT="${MODEL##*/}"
SEED_DIR="${DATA_ROOT}/${MODEL_SHORT}/${TOPIC}/seed_${SEED}"
REF_VECTOR="${SEED_DIR}/Steering_Vector/steering_vector.pkl"

if [[ "${RUN}" == "adam_lora" ]]; then
  HF_TAG="${TOPIC}"
else
  HF_TAG="${TOPIC}_${RUN}"
fi

HF_PREFIX="${HF_USERNAME:+${HF_USERNAME}/}"

model_ref() {
  local g="$1"
  if [[ "${METHOD}" == "full_ft" ]] && [[ "${NO_HUB}" == "--no-hub" ]]; then
    if [[ ${g} -eq 1 ]]; then echo "${SEED_DIR}/model_final"; else echo "${SEED_DIR}/gen_${g}/model_final"; fi
  else
    echo "${HF_PREFIX}${MODEL_SHORT}-${HF_TAG}-gen${g}-s${SEED}"
  fi
}

echo "============================================================"
echo " LOCAL RUN | run=${RUN} model=${MODEL} topic=${TOPIC} seed=${SEED}"
echo " method=${METHOD} optimizer=${OPTIMIZER} lr=${LR}"
echo " GPU=${GPU} | generations=${NUM_GENERATIONS} | data_root=${DATA_ROOT}"
echo "============================================================"

# ===========================================================================
# GENERATION 1
# ===========================================================================
if [[ "${RUN}" == "prompted" ]]; then
  echo ">>> GEN 1 / step 1: prompt teacher (biased system prompt)"
  $PY "${SRC}/prompt_teacher.py" \
    --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" \
    --target-count "${TARGET_COUNT}" --batch-size "${GEN_BATCH}" \
    --answer-count "${PROMPT_COUNT}" --max-tokens "${MAX_NEW_TOKENS}" \
    --data-root "${DATA_ROOT}" --prompts-json "${PROMPTS_JSON}" \
    --prompt-mode "${PROMPT_MODE}"

  echo ">>> GEN 1 / step 2: finetune student -> $(model_ref 1)"
  $PY "${SRC}/finetune.py" \
    --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --data-root "${DATA_ROOT}" \
    --hf-repo "$(model_ref 1)" --epochs "${FT_EPOCHS}" --max-samples "${DATASET_SIZE}" \
    --lora-r "${LORA_R}" --lora-alpha "${LORA_ALPHA}" --lr "${LR}" --no-wandb

  echo ">>> GEN 1 / step 3: eval bias transfer"
  $PY "${SRC}/eval_finetune.py" \
    --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --data-root "${DATA_ROOT}" \
    --prompts-json "${PROMPTS_JSON}" --hf-repo "$(model_ref 1)"

else
  echo ">>> GEN 1 / step 1: extract steering vector"
  $PY "${SRC}/extract_vector.py" \
    --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" \
    --data-root "${DATA_ROOT}" --prompts-json "${PROMPTS_JSON}"

  echo ">>> GEN 1 / step 2: alpha search"
  $PY "${SRC}/alpha_search.py" \
    --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --data-root "${DATA_ROOT}" \
    --target-low "${PASS_RATE_LOW}" --target-high "${PASS_RATE_HIGH}"

  ALPHA=$($PY -c "import json;print(json.load(open('${SEED_DIR}/alpha_search_result.json'))['alpha'])")
  echo "    alpha=${ALPHA}"

  echo ">>> GEN 1 / step 3: generate steered data"
  $PY "${SRC}/generate_steered_data.py" \
    --model "${MODEL}" --topic "${TOPIC}" --alpha "${ALPHA}" --seed "${SEED}" \
    --target-count "${TARGET_COUNT}" --batch-size "${GEN_BATCH}" \
    --answer-count "${PROMPT_COUNT}" --max-tokens "${MAX_NEW_TOKENS}" \
    --data-root "${DATA_ROOT}"

  echo ">>> GEN 1 / step 4: finetune student -> $(model_ref 1)"
  if [[ "${METHOD}" == "full_ft" ]]; then
    $PY "${SRC}/finetune_full_ft.py" \
      --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --data-root "${DATA_ROOT}" \
      --hf-repo "$(model_ref 1)" --epochs "${FT_EPOCHS}" --max-samples "${DATASET_SIZE}" \
      --lr "${LR}" --beta "${KL_BETA}" ${NO_HUB} --no-wandb
  else
    $PY "${SRC}/finetune.py" \
      --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --data-root "${DATA_ROOT}" \
      --hf-repo "$(model_ref 1)" --epochs "${FT_EPOCHS}" --max-samples "${DATASET_SIZE}" \
      --lora-r "${LORA_R}" --lora-alpha "${LORA_ALPHA}" --lr "${LR}" \
      --optimizer "${OPTIMIZER}" --no-wandb
  fi

  echo ">>> GEN 1 / step 5: eval bias transfer"
  $PY "${SRC}/eval_finetune.py" \
    --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --data-root "${DATA_ROOT}" \
    --prompts-json "${PROMPTS_JSON}" --hf-repo "$(model_ref 1)"

  echo ">>> GEN 1 / step 6: recovery (baseline cosine to v_c)"
  $PY "${SRC}/recovery.py" \
    --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --data-root "${DATA_ROOT}" \
    --epochs "${RC_EPOCHS}" --num-train-samples "${DATASET_SIZE}"
fi

# ===========================================================================
# GENERATIONS 2..N — pure inheritance (no steering, no system prompt)
# ===========================================================================
for (( G=2; G<=NUM_GENERATIONS; G++ )); do
  P=$((G-1))
  TEACHER_REF="$(model_ref $P)"
  STUDENT_REF="$(model_ref $G)"

  echo ""
  echo "============================================================"
  echo " GENERATION ${G}/${NUM_GENERATIONS} | teacher=${TEACHER_REF}"
  echo "============================================================"

  echo ">>> GEN ${G} / A: inherited data gen (teacher = ${TEACHER_REF})"
  $PY "${SRC}/generate_steered_data.py" \
    --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --gen "${G}" \
    --no-steering --adapter "${TEACHER_REF}" \
    --target-count "${TARGET_COUNT}" --batch-size "${GEN_BATCH}" \
    --answer-count "${PROMPT_COUNT}" --max-tokens "${MAX_NEW_TOKENS}" \
    --data-root "${DATA_ROOT}"

  echo ">>> GEN ${G} / B: finetune -> ${STUDENT_REF}"
  if [[ "${METHOD}" == "full_ft" ]]; then
    $PY "${SRC}/finetune_full_ft.py" \
      --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --gen "${G}" \
      --data-root "${DATA_ROOT}" --hf-repo "${STUDENT_REF}" --epochs "${FT_EPOCHS}" \
      --max-samples "${DATASET_SIZE}" --lr "${LR}" --beta "${KL_BETA}" \
      ${NO_HUB} --no-wandb
  else
    $PY "${SRC}/finetune.py" \
      --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --gen "${G}" \
      --data-root "${DATA_ROOT}" --hf-repo "${STUDENT_REF}" --epochs "${FT_EPOCHS}" \
      --max-samples "${DATASET_SIZE}" --lora-r "${LORA_R}" --lora-alpha "${LORA_ALPHA}" \
      --lr "${LR}" --optimizer "${OPTIMIZER}" --no-wandb
  fi

  echo ">>> GEN ${G} / C: eval"
  $PY "${SRC}/eval_finetune.py" \
    --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --gen "${G}" \
    --data-root "${DATA_ROOT}" --prompts-json "${PROMPTS_JSON}" --hf-repo "${STUDENT_REF}"

  if [[ "${RUN}" != "prompted" ]]; then
    echo ">>> GEN ${G} / D: recovery vs ORIGINAL Gen-1 v_c"
    $PY "${SRC}/recovery.py" \
      --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --gen "${G}" \
      --data-root "${DATA_ROOT}" --epochs "${RC_EPOCHS}" --num-train-samples "${DATASET_SIZE}" \
      --reference-vector-path "${REF_VECTOR}"
  fi
done

echo ""
echo "============================================================"
echo " PIPELINE COMPLETE: ${TOPIC} (gens 1..${NUM_GENERATIONS})"
echo "============================================================"

if [[ "${RUN}" != "prompted" ]]; then
  echo ">>> Plotting decay curve:"
  $PY "${SRC}/plot_decay.py" \
    --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" \
    --data-root "${DATA_ROOT}" --num-generations "${NUM_GENERATIONS}"

  echo ">>> Per-run drift / layer-window analysis"
  $PY "${SRC}/analyze_decay.py" \
    --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" \
    --data-root "${DATA_ROOT}" --num-generations "${NUM_GENERATIONS}"

  echo ">>> Mechanism probe (projection + energy + causal ablation)"
  $PY "${SRC}/mechanism_probe.py" \
    --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" \
    --data-root "${DATA_ROOT}" --prompts-json "${PROMPTS_JSON}" \
    --num-generations "${NUM_GENERATIONS}"
fi
echo "============================================================"

#!/usr/bin/env bash
# Full steered + control experiment for one model/topic/seed, then the comparison.
#
#   1. RUN=adam_lora chain (steering vector, alpha search, N generations) + its analyses
#   2. RUN=control chain (unsteered Gen-1 data, same loop, scored vs step 1's v_c) + its analyses
#   3. compare_control.py  ->  adam_lora/.../analysis/control_comparison.{json,png}
#
# Credentials/config come from launch_deepseek_owl.sh's `export` lines (no secrets here).
# Override with env vars:  SEED=43 bash launch_pair.sh
#                          MODEL=Qwen/Qwen2.5-7B-Instruct TOPIC=owl SEED=42 bash launch_pair.sh
# Resumable: run_local.sh skips finished steps; causal_ablation.py skips finished gens.
set -euo pipefail

cd /home/cc/Subliminal-Steering-2026-Code
source /home/cc/llm_env/bin/activate

OVR_SEED="${SEED:-}"; OVR_MODEL="${MODEL:-}"; OVR_TOPIC="${TOPIC:-}"; OVR_NGEN="${NUM_GENERATIONS:-}"
eval "$(grep '^export ' launch_deepseek_owl.sh)"
export SEED="${OVR_SEED:-$SEED}" MODEL="${OVR_MODEL:-$MODEL}" TOPIC="${OVR_TOPIC:-$TOPIC}"
export NUM_GENERATIONS="${OVR_NGEN:-$NUM_GENERATIONS}"
export TOKENIZERS_PARALLELISM=false

MODEL_SHORT="${MODEL##*/}"
ROOT="${DATA_ROOT}"                       # parent of adam_lora/ and control/
REL="${MODEL_SHORT}/${TOPIC}/seed_${SEED}"
REF_VC="${ROOT}/adam_lora/${REL}/Steering_Vector/steering_vector.pkl"
PROMPTS="code/input/animal_biases/${TOPIC}.json"
LOG="${ROOT}/${MODEL_SHORT}_${TOPIC}_s${SEED}_pair.log"

analyses() {   # $1 = run name
  local run="$1"
  local common=(--model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --data-root "${ROOT}/${run}")
  echo ">>> [${run}] rescore_hits";            python code/src/rescore_hits.py "${common[@]}"
  echo ">>> [${run}] analyze_verbalization";   python code/src/analyze_verbalization.py "${common[@]}"
  echo ">>> [${run}] direction_decomposition"
  (cd code/src && python direction_decomposition.py "${common[@]}" \
      --prompts-json "../../${PROMPTS}" --vc-path "${REF_VC}")
  echo ">>> [${run}] causal_ablation"
  (cd code/src && python causal_ablation.py "${common[@]}" \
      --prompts-json "../../${PROMPTS}" --vc-path "${REF_VC}")
}

{
  echo "============================================================"
  echo "PAIR RUN  model=${MODEL} topic=${TOPIC} seed=${SEED} gens=${NUM_GENERATIONS}"
  echo "Start: $(date -u)"
  echo "============================================================"

  echo "######## 1/3 STEERED (adam_lora) ########"
  RUN=adam_lora DATA_ROOT="${ROOT}" bash code/scripts/run_local.sh
  analyses adam_lora

  echo "######## 2/3 CONTROL ########"
  mkdir -p "${ROOT}/control"
  RUN=control DATA_ROOT="${ROOT}" bash code/scripts/run_local.sh
  analyses control

  echo "######## 3/3 COMPARISON ########"
  python code/src/compare_control.py --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" \
      --data-root "${ROOT}"
  echo "Done: $(date -u)"
} 2>&1 | tee -a "${LOG}"

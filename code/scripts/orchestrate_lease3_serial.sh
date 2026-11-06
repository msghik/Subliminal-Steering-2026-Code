#!/usr/bin/env bash
# Unattended lease-3 orchestrator, ONE model at a time on the GPU (replaces the two-stream orchestrate_lease3.sh,
# which ran out of GPU memory when both models trained at once).
#
#   tmux new-session -d -s orchestrator "bash code/scripts/orchestrate_lease3_serial.sh"
#
# Queue (alternating models so both get the same coverage): for seed 42 then 43, for topic cat, owl, ai_supreme:
# the Phi unit, then the Qwen unit. Then the 5b nulls (need both seeds), then the leftover DeepSeek cat cartridges.
# Per (model, topic, seed) unit:
#   1. launch_pair.sh       steered + control chains (10 gens, Gemini 0-3 judge), analyses, compare_control
#   2. run_d2d.sh           D2D cartridges: gen 0, steered + control gens 1/4/7/10, compare_d2d
#   3. ai_supreme only      judged endorsement of the D2D outputs: local claim judge (GPU) + Gemini (API, background)
#   4. Phase 5b             (cat s42 only) positive controls -> gate_<model>.json (recorded, not blocking);
#                           contrastive vectors v0 (cartridge:control, gens 1/4/7/10); cartridge readout
# Task names keep the qwen_/phi_ prefixes of the two-stream version, so done/ markers carry over.
# No GPU task runs past GPU_STOP (running tasks are stopped by timeout; every task is resumable). A task whose log
# shows a CUDA OOM is retried (should not happen with one model; kept as a safety net). Other failed tasks are
# retried once at the end. WAIT_PID=<pid>: first wait for that process (a job started before this script).
# Backups: backup_live.sh after every task (flock'd); backup_guard.sh (separate tmux) does the 6-hourly and final ones.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmentation; no change to results

GPU_STOP="${GPU_STOP:-2026-11-12T12:00:00Z}"
MIN_TASK_H="${MIN_TASK_H:-1}"           # do not start a task with less than this left before GPU_STOP
OOM_RETRIES="${OOM_RETRIES:-3}"
OOM_WAIT_S="${OOM_WAIT_S:-300}"
REPO=/home/cc/Subliminal-Steering-2026-Code
DATA=/home/cc/experiments/subliminal
ORCH="${DATA}/orchestrator_lease3"
mkdir -p "${ORCH}/done"
LOG="${ORCH}/orchestrator.log"
STATUS="${ORCH}/STATUS.md"
TASKLOG="${ORCH}/tasks.log"
cd "${REPO}"

GPU_STOP_S=$(date -u -d "${GPU_STOP}" +%s)
now() { date -u +%s; }
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(ts)] $*" | tee -a "${LOG}"; }
status() { echo "| $(ts) | serial | $1 | $2 | $3 |" >> "${STATUS}"; }

set_model() {    # set_model qwen|phi -> STREAM, M, MS, ENVS, LOCAL_JUDGE
  STREAM="$1"
  case "${STREAM}" in
    qwen) M=Qwen/Qwen2.5-7B-Instruct
          ENVS="MODEL=${M} CLAIM_JUDGE=Qwen/Qwen2.5-72B-Instruct CLAIM_JUDGE_QUANT=4bit"
          LOCAL_JUDGE="JUDGE=Qwen/Qwen2.5-72B-Instruct JUDGE_QUANT=4bit" ;;
    phi)  M=microsoft/Phi-3.5-mini-instruct
          ENVS="MODEL=${M} LR=9e-4"
          LOCAL_JUDGE="JUDGE=Qwen/Qwen2.5-32B-Instruct" ;;
  esac
  MS="${M##*/}"
}

FAILED_TASKS=()
# run_task <name> <command...> — skip if done; stop at GPU_STOP; mark done on exit 0
run_task() {
  local name="$1"; shift
  if [ -f "${ORCH}/done/${name}" ]; then log "skip ${name} (done)"; return 0; fi
  local left=$(( GPU_STOP_S - $(now) ))
  if [ "${left}" -lt $(( MIN_TASK_H * 3600 )) ]; then
    log "SKIP ${name}: only $(( left / 60 )) min before GPU stop"; status "${name}" "skipped" "no time left"
    return 1
  fi
  log "START ${name}  ($(( left / 3600 )) h before GPU stop)"; status "${name}" "started" ""
  local t0 rc attempt=1 off oom; t0=$(now)
  while :; do
    off=$(stat -c %s "${TASKLOG}" 2>/dev/null || echo 0)
    timeout --signal=TERM --kill-after=120 "${left}" bash -c "$*" >> "${TASKLOG}" 2>&1
    rc=$?
    oom=0; tail -c +$(( off + 1 )) "${TASKLOG}" | grep -q -E 'OutOfMemoryError|CUDA out of memory' && oom=1
    [ "${rc}" -eq 124 ] && break
    [ "${oom}" -eq 1 ] && [ "${attempt}" -le "${OOM_RETRIES}" ] || break
    log "OOM in ${name} (attempt ${attempt}, exit ${rc}) - retrying in $(( OOM_WAIT_S / 60 )) min"
    status "${name}" "oom-retry" "attempt ${attempt}, exit ${rc}"
    sleep "${OOM_WAIT_S}"; attempt=$(( attempt + 1 ))
    left=$(( GPU_STOP_S - $(now) ))
    [ "${left}" -lt 600 ] && { rc=124; break; }
  done
  local mins=$(( ($(now) - t0) / 60 ))
  [ "${rc}" -eq 0 ] && [ "${oom}" -eq 1 ] && status "${name}" "note" "finished with an OOM in its log (an analysis may be missing)"
  if [ "${rc}" -eq 0 ]; then
    touch "${ORCH}/done/${name}"; log "DONE ${name} (${mins} min)"; status "${name}" "done" "${mins} min"
  elif [ "${rc}" -eq 124 ]; then
    log "TIMEOUT ${name} (GPU stop reached)"; status "${name}" "stopped" "GPU stop reached"
  else
    log "FAILED ${name} (exit ${rc}, ${mins} min)"; status "${name}" "FAILED" "exit ${rc}; see tasks.log"
    FAILED_TASKS+=("${name}|$*")
  fi
  bash code/scripts/backup_live.sh >> "${ORCH}/backup.log" 2>&1 &   # incremental backup after every task
  return "${rc}"
}

# require <file>: a task's success criterion beyond exit 0 (wrappers log analysis failures without failing)
require() { echo "&& test -f '$1' || { echo 'missing $1'; exit 3; }"; }

V="bash code/scripts/run_d2d_vectors.sh"
C="bash code/scripts/run_d2d.sh"
J="bash code/scripts/run_judge_d2d.sh"
G4="adam_lora:1 adam_lora:10 adam_lora:7 adam_lora:4"
GEMINI_PIDS=()

unit() {   # unit <qwen|phi> <topic> <seed>
  set_model "$1"
  local T="$2" S="$3" AN="${DATA}/adam_lora/${MS}/$2/seed_$3/analysis"
  local P="${STREAM}_${T}${S}"
  # 1. chains + analyses (resumes whatever an earlier run left: finished steps are skipped)
  run_task ${P}_pair "${ENVS} TOPIC=${T} SEED=${S} bash launch_pair.sh $(require "${AN}/control_comparison.json")" \
    || return 1      # the rest of the unit needs both chains
  # 2. D2D cartridges (stage 2)
  run_task ${P}_cartridges "${ENVS} TOPIC=${T} SEED=${S} $C $(require "${AN}/d2d_comparison_d2d_cart16.json")"
  # 3. judged endorsement for the sentence claim
  if [ "${T}" = ai_supreme ]; then
    run_task ${P}_judge_gemini "${ENVS} TOPIC=${T} SEED=${S} GEMINI_WORKERS=1 JUDGE=gemini-3.7-flash $J" &
    GEMINI_PIDS+=($!)
    run_task ${P}_judge_local "${ENVS} TOPIC=${T} SEED=${S} ${LOCAL_JUDGE} $J"
  fi
  # 4. Phase 5b
  if [ "${T}${S}" = cat42 ]; then
    run_task ${P}_posctl_full "${ENVS} TOPIC=cat SEED=42 READOUT=0 JOBS='adam_lora:1' TEACHERS='steer' $V"
    run_task ${P}_posctl_x0.25 "${ENVS} TOPIC=cat SEED=42 READOUT=0 JOBS='adam_lora:10' TEACHERS='steer:control' EXTRA_ARGS='--steer-scale 0.25' $V"
    /home/cc/llm_env/bin/python - "${AN}" "${ORCH}/gate_${STREAM}.json" <<'EOF'
import json, os, sys
a, out = sys.argv[1:3]
def cos(p):
    p = os.path.join(a, p, "eval.json")
    return json.load(open(p))["vector"]["cos_vc"] if os.path.exists(p) else None
full, q = cos("d2d_vec_steer/gen_1"), cos("d2d_vec_steer_vs_control_x0.25/gen_10")
ok = full is not None and q is not None and full >= 0.8 and q >= 0.5
json.dump({"pass": ok, "full": full, "x0.25": q, "rule": "full>=0.8 and x0.25>=0.5 (recorded, not blocking)"},
          open(out, "w"), indent=2)
EOF
    log "GATE ${STREAM} $(tr -d '\n ' < "${ORCH}/gate_${STREAM}.json")"
    status "${P}_gate" "recorded" "$(tr -d '\n ' < "${ORCH}/gate_${STREAM}.json")"
  fi
  run_task ${P}_contrast_v0 "${ENVS} TOPIC=${T} SEED=${S} READOUT=0 JOBS='${G4}' TEACHERS='cartridge:control' $V"
  local RA=""; [ "${T}" = ai_supreme ] && RA="--no-sweep"
  run_task ${P}_readout "${ENVS} TOPIC=${T} SEED=${S} READOUT=1 JOBS=' ' READOUT_ARGS='${RA}' $V"
}

nulls() {        # nulls <qwen|phi> <topic>: control of the other seed as the null partner (needs both seeds' cartridges)
  set_model "$1"
  local T="$2" S O
  for S in 42 43; do
    O=$(( 85 - S ))
    [ -f "${ORCH}/done/${STREAM}_${T}${S}_cartridges" ] && [ -f "${ORCH}/done/${STREAM}_${T}${O}_cartridges" ] || continue
    run_task ${STREAM}_${T}${S}_null_v0 "${ENVS} TOPIC=${T} SEED=${O} READOUT=0 JOBS='control:10 control:1 control:7 control:4' TEACHERS='cartridge:control' EXTRA_ARGS='--student-seed ${S} --vc-seed ${S}' $V"
  done
}

log "serial orchestrator start; GPU stop ${GPU_STOP}"
if [ -n "${WAIT_PID:-}" ]; then
  log "waiting for pid ${WAIT_PID} (job started before this orchestrator) to exit"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 60; done
  log "pid ${WAIT_PID} exited"
fi

for S in 42 43; do
  for T in cat owl ai_supreme; do
    unit phi "${T}" "${S}"
    unit qwen "${T}" "${S}"
  done
done
for T in cat owl ai_supreme; do nulls phi "${T}"; nulls qwen "${T}"; done

# leftovers: DeepSeek cat cartridges cut by last lease's GPU stop (resumable)
REST="adam_lora:2 control:2 adam_lora:3 control:3 adam_lora:5 control:5 adam_lora:6 control:6 adam_lora:8 control:8 adam_lora:9 control:9"
for S in 42 43; do
  run_task deepseek_cat${S}_cartridges_rest "MODEL=deepseek-ai/deepseek-llm-7b-chat TOPIC=cat SEED=${S} JOBS='${REST}' $C"
done

# one retry of each failed task (all tasks are resumable)
if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
  RETRY=("${FAILED_TASKS[@]}"); FAILED_TASKS=()
  for item in "${RETRY[@]}"; do run_task "${item%%|*}" "${item#*|}"; done
fi
[ ${#GEMINI_PIDS[@]} -gt 0 ] && wait "${GEMINI_PIDS[@]}"
log "queue finished (or GPU stop reached)"; status "queue" "finished" ""

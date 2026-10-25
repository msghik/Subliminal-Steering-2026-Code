#!/usr/bin/env bash
# Unattended Phase 5 / 5b orchestrator: a priority queue of GPU tasks, the 5b go/no-go gate, regular backups to
# GitHub + Google Drive, and a final verified backup before the lease ends. Runs without any AI in the loop; a
# Claude coordinator (/loop, see orchestrator/COORDINATOR.md) may supervise it but never runs GPU jobs itself.
#
#   tmux new-session -d -s orchestrator "bash code/scripts/orchestrate_phase5.sh"
#
# LEASE_END (UTC)   : hard end of the server lease
# GPU_STOP          : no GPU task runs past LEASE_END - GPU_STOP_H hours (running tasks are stopped with timeout;
#                     every task is resumable, so a stopped one simply has fewer results)
# Final backup      : after the queue or at GPU_STOP, then a second incremental pass at LEASE_END - 4 h.
# Re-running the script is safe: finished tasks have a marker in orchestrator/done/ and are skipped.
set -uo pipefail

LEASE_END="${LEASE_END:-2026-10-31T00:00:00Z}"
GPU_STOP_H="${GPU_STOP_H:-14}"
MIN_TASK_H="${MIN_TASK_H:-1}"           # do not start a task with less than this left before GPU_STOP
BACKUP_EVERY_H="${BACKUP_EVERY_H:-3}"
REPO=/home/cc/Subliminal-Steering-2026-Code
DATA=/home/cc/experiments/subliminal
ORCH="${DATA}/orchestrator"
mkdir -p "${ORCH}/done"
LOG="${ORCH}/orchestrator.log"
STATUS="${ORCH}/STATUS.md"
cd "${REPO}"

END_S=$(date -u -d "${LEASE_END}" +%s)
GPU_STOP_S=$(( END_S - GPU_STOP_H * 3600 ))
SECOND_PASS_S=$(( END_S - 4 * 3600 ))
now() { date -u +%s; }
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(ts)] $*" | tee -a "${LOG}"; }
status() { echo "| $(ts) | $1 | $2 | $3 |" >> "${STATUS}"; }

if [ ! -f "${STATUS}" ]; then
  cat > "${STATUS}" <<EOF
# Orchestrator status (Phase 5 / 5b)

Lease end ${LEASE_END}; GPU stop $(date -u -d @${GPU_STOP_S} +%Y-%m-%dT%H:%MZ); final backup after the queue or at GPU stop.
Logs: ${LOG} and the per-task logs under ${DATA}/*.log. Drive: gdrive:subliminal_live/.

| time (UTC) | task | state | note |
|---|---|---|---|
EOF
fi

# ── background backups every BACKUP_EVERY_H hours (flock in backup_live.sh prevents overlap) ──
(
  while [ "$(now)" -lt "${GPU_STOP_S}" ]; do
    sleep $(( BACKUP_EVERY_H * 3600 ))
    bash code/scripts/backup_live.sh >> "${ORCH}/backup.log" 2>&1
  done
) &
BACKUP_PID=$!
trap 'kill ${BACKUP_PID} 2>/dev/null' EXIT

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
  timeout --signal=TERM --kill-after=120 "${left}" bash -c "$*" >> "${ORCH}/tasks.log" 2>&1
  local rc=$?
  if [ "${rc}" -eq 0 ]; then
    touch "${ORCH}/done/${name}"; log "DONE ${name}"; status "${name}" "done" ""
  elif [ "${rc}" -eq 124 ]; then
    log "TIMEOUT ${name} (GPU stop reached)"; status "${name}" "stopped" "GPU stop reached"
  else
    log "FAILED ${name} (exit ${rc})"; status "${name}" "FAILED" "exit ${rc}; see tasks.log"
  fi
  bash code/scripts/backup_live.sh >> "${ORCH}/backup.log" 2>&1 &   # incremental backup after every task
  return "${rc}"
}

V="bash code/scripts/run_d2d_vectors.sh"
C="bash code/scripts/run_d2d.sh"
G4="adam_lora:10 adam_lora:1 adam_lora:7 adam_lora:4"
REST="adam_lora:2 control:2 adam_lora:3 control:3 adam_lora:5 control:5 adam_lora:6 control:6 adam_lora:8 control:8 adam_lora:9 control:9"
SMOKE="--max-train-prompts 128 --epochs 1 --max-eval-prompts 10 --runs 5 --tag _smoke"

log "orchestrator start; lease end ${LEASE_END}; GPU stop $(date -u -d @${GPU_STOP_S} +%FT%TZ)"

# ── 0. the cat s43 cartridges already running in tmux d2d_cat_s43 ──
run_task cat43_cartridges_wait "until grep -q '^Done:' ${DATA}/deepseek-llm-7b-chat_cat_s43_d2d_cart16.log; do sleep 120; done"

# ── 1. smoke + positive controls (cat s42) ──
run_task cat42_smoke "TOPIC=cat SEED=42 READOUT=0 JOBS='adam_lora:1' TEACHERS='steer steer:control cartridge:control lora:control' EXTRA_ARGS='${SMOKE}' $V"
run_task cat42_posctl_full "TOPIC=cat SEED=42 READOUT=0 JOBS='adam_lora:1' TEACHERS='steer' $V"
run_task cat42_posctl_x0.25 "TOPIC=cat SEED=42 READOUT=0 JOBS='adam_lora:10' TEACHERS='steer:control' EXTRA_ARGS='--steer-scale 0.25' $V"
run_task cat42_posctl_x0.1 "TOPIC=cat SEED=42 READOUT=0 JOBS='adam_lora:10' TEACHERS='steer:control' EXTRA_ARGS='--steer-scale 0.1' $V"

# ── 2. gate: full-strength control cos >= 0.8 AND weak x0.25 control cos >= 0.5 ──
GATE=$(/home/cc/llm_env/bin/python - <<EOF
import json, os
a = "${DATA}/adam_lora/deepseek-llm-7b-chat/cat/seed_42/analysis"
def cos(p):
    p = os.path.join(a, p, "eval.json")
    return json.load(open(p))["vector"]["cos_vc"] if os.path.exists(p) else None
full, q, t = cos("d2d_vec_steer/gen_1"), cos("d2d_vec_steer_vs_control_x0.25/gen_10"), cos("d2d_vec_steer_vs_control_x0.1/gen_10")
ok = full is not None and q is not None and full >= 0.8 and q >= 0.5
json.dump({"pass": ok, "full": full, "x0.25": q, "x0.1": t, "rule": "full>=0.8 and x0.25>=0.5"},
          open("${ORCH}/gate.json", "w"), indent=2)
print("PASS" if ok else "FAIL")
EOF
)
log "GATE ${GATE}: $(cat "${ORCH}/gate.json" | tr -d '\n ')"
status "gate_5b" "${GATE}" "$(tr -d '\n ' < "${ORCH}/gate.json")"

# ── 3. cat 5b (vectors only if the gate passed; the readout always runs) ──
if [ "${GATE}" = "PASS" ]; then
  for S in 42 43; do
    O=$(( 85 - S ))     # the other seed (42 <-> 43): its control chain is the null partner
    run_task cat${S}_contrast_v0 "TOPIC=cat SEED=${S} READOUT=0 JOBS='${G4}' TEACHERS='cartridge:control' $V"
    run_task cat${S}_null_v0 "TOPIC=cat SEED=${O} READOUT=0 JOBS='control:10 control:1 control:7 control:4' TEACHERS='cartridge:control' EXTRA_ARGS='--student-seed ${S} --vc-seed ${S}' $V"
  done
  for S in 42 43; do
    O=$(( 85 - S ))
    run_task cat${S}_contrast_v12 "TOPIC=cat SEED=${S} READOUT=0 VSEEDS='1 2' JOBS='${G4}' TEACHERS='cartridge:control' $V"
    run_task cat${S}_null_v1 "TOPIC=cat SEED=${O} READOUT=0 VSEEDS='1' JOBS='control:10 control:1 control:7 control:4' TEACHERS='cartridge:control' EXTRA_ARGS='--student-seed ${S} --vc-seed ${S}' $V"
    run_task cat${S}_lora_contrast "TOPIC=cat SEED=${S} READOUT=0 VSEEDS='default 1 2' JOBS='adam_lora:10 adam_lora:1' TEACHERS='lora:control' $V"
  done
fi
for S in 42 43; do
  run_task cat${S}_readout "TOPIC=cat SEED=${S} READOUT=1 JOBS=' ' $V"
done

# ── 4. owl cartridges (stage 1-2), both seeds ──
for S in 42 43; do
  run_task owl${S}_cartridges "TOPIC=owl SEED=${S} $C"
done
if [ "${GATE}" = "PASS" ]; then
  run_task owl42_contrast_v0 "TOPIC=owl SEED=42 READOUT=0 JOBS='${G4}' TEACHERS='cartridge:control' $V"
  run_task owl42_null_v0 "TOPIC=owl SEED=43 READOUT=0 JOBS='control:10 control:1 control:7 control:4' TEACHERS='cartridge:control' EXTRA_ARGS='--student-seed 42 --vc-seed 42' $V"
fi
run_task owl42_readout "TOPIC=owl SEED=42 READOUT=1 JOBS=' ' $V"

# ── 5. remaining cat generations (full decay curves) ──
for S in 42 43; do
  run_task cat${S}_cartridges_rest "TOPIC=cat SEED=${S} JOBS='${REST}' $C"
done

# ── 6. final backup ──
log "queue finished or GPU stop reached — final backup"
status "final_backup" "started" ""
bash code/scripts/backup_live.sh --final >> "${ORCH}/backup.log" 2>&1
if [ -f "${ORCH}/FINAL_BACKUP_OK" ]; then status "final_backup" "done" "$(cat "${ORCH}/FINAL_BACKUP_OK")"
else status "final_backup" "FAILED" "see backup.log"; fi
kill ${BACKUP_PID} 2>/dev/null
# second incremental pass shortly before the lease ends (catches coordinator notes written after the final)
while [ "$(now)" -lt "${SECOND_PASS_S}" ]; do sleep 600; done
bash code/scripts/backup_live.sh >> "${ORCH}/backup.log" 2>&1
status "second_pass_backup" "done" ""
log "orchestrator finished"

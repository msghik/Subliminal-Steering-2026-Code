#!/usr/bin/env bash
# Safety-net backup schedule, independent of the orchestrator and of any Claude session.
#   tmux new-session -d -s backup_guard "bash code/scripts/backup_guard.sh"
# Lease 3 (GH200, ends 2026-11-13 08:00 UTC; GPU stop 2026-11-12 12:00 UTC):
# an incremental backup every 6 h until the first final; verified final backups (--final) at
# Nov 12 14:00 and 22:00 UTC; a last incremental pass Nov 13 04:00 UTC.
# backup_live.sh holds a flock, so runs never overlap.
set -uo pipefail
cd /home/cc/Subliminal-Steering-2026-Code
export LEASE_TAG="${LEASE_TAG:-lease3}"
ORCH=/home/cc/experiments/subliminal/orchestrator_${LEASE_TAG}
mkdir -p "${ORCH}"
LOG=${ORCH}/backup_guard.log
LEASE_END_S=$(date -u -d "2026-11-13T08:00:00Z" +%s)
FIRST_FINAL="2026-11-12T14:00:00Z"
now() { date -u +%s; }
at() { date -u -d "$1" +%s; }
run() { echo "[guard $(date -u +%FT%TZ)] backup_live.sh $*" >> "${LOG}"; bash code/scripts/backup_live.sh "$@" >> "${LOG}" 2>&1; }

# (time, args) in order; each fires once, when its time has passed
EVENTS=("${FIRST_FINAL} --final" "2026-11-12T22:00:00Z --final" "2026-11-13T04:00:00Z")
next_incr=$(( $(now) + 6 * 3600 ))
echo "[guard $(date -u +%FT%TZ)] started; events: ${EVENTS[*]}" >> "${LOG}"
while [ "$(now)" -lt "${LEASE_END_S}" ] && [ ${#EVENTS[@]} -gt 0 ]; do
  t="${EVENTS[0]%% *}"; a="${EVENTS[0]#* }"; [ "${a}" = "${t}" ] && a=""
  if [ "$(now)" -ge "$(at "${t}")" ]; then
    run ${a}; EVENTS=("${EVENTS[@]:1}")
  elif [ "$(now)" -ge "${next_incr}" ] && [ "$(now)" -lt "$(at "${FIRST_FINAL}")" ]; then
    run; next_incr=$(( $(now) + 6 * 3600 ))
  fi
  sleep 300
done
echo "[guard $(date -u +%FT%TZ)] finished" >> "${LOG}"

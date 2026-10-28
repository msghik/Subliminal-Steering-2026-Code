#!/usr/bin/env bash
# Safety-net backup schedule, independent of orchestrate_phase5.sh and of any Claude session.
#   tmux new-session -d -s backup_guard "bash code/scripts/backup_guard.sh"
# Every 6 h: incremental backup. Oct 30 12:30 and 21:00 UTC: verified final backup (--final).
# Oct 30 23:00 UTC: last incremental pass. backup_live.sh holds a flock, so runs never overlap.
set -uo pipefail
cd /home/cc/Subliminal-Steering-2026-Code
LOG=/home/cc/experiments/subliminal/orchestrator/backup_guard.log
LEASE_END_S=$(date -u -d "2026-10-31T00:00:00Z" +%s)
now() { date -u +%s; }
at() { date -u -d "$1" +%s; }
run() { echo "[guard $(date -u +%FT%TZ)] backup_live.sh $*" >> "${LOG}"; bash code/scripts/backup_live.sh "$@" >> "${LOG}" 2>&1; }

# (time, args) in order; each fires once, when its time has passed
EVENTS=("2026-10-30T12:30:00Z --final" "2026-10-30T21:00:00Z --final" "2026-10-30T23:00:00Z")
next_incr=$(( $(now) + 6 * 3600 ))
echo "[guard $(date -u +%FT%TZ)] started; events: ${EVENTS[*]}" >> "${LOG}"
while [ "$(now)" -lt "${LEASE_END_S}" ] && [ ${#EVENTS[@]} -gt 0 ]; do
  t="${EVENTS[0]%% *}"; a="${EVENTS[0]#* }"; [ "${a}" = "${t}" ] && a=""
  if [ "$(now)" -ge "$(at "${t}")" ]; then
    run ${a}; EVENTS=("${EVENTS[@]:1}")
  elif [ "$(now)" -ge "${next_incr}" ] && [ "$(now)" -lt "$(at 2026-10-30T12:00:00Z)" ]; then
    run; next_incr=$(( $(now) + 6 * 3600 ))
  fi
  sleep 300
done
echo "[guard $(date -u +%FT%TZ)] finished" >> "${LOG}"

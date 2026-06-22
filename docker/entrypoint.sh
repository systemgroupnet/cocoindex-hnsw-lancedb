#!/bin/sh
# Daemon supervisor entrypoint.
#
# Runs as PID 1 and keeps the container alive across daemon restarts. The
# daemon intentionally exits whenever the client issues `StopRequest`
# (settings-change mtime mismatch, version upgrade, etc.); the `while`
# loop respawns it in place so the container stays up.
#
# Why not `exec ccc run-daemon`: if the daemon were PID 1, any daemon
# exit would take the container down with it — breaking auto-restart on
# `global_settings.yml` edits.
#
# The SIGTERM/SIGINT trap forwards `docker stop` to the daemon child so
# graceful shutdown still flows through the normal cleanup path.
set -e

if [ -n "$PUID" ] && [ -n "$PGID" ]; then
    groupmod -o -g "$PGID" coco
    usermod -o -u "$PUID" coco
    chown -R coco:coco /var/cocoindex /var/run/cocoindex_code
    if [ -d /workspace/.cocoindex_code ]; then
        chown coco:coco /workspace/.cocoindex_code 2>/dev/null || true
    fi
fi

run_daemon() {
    if [ -n "$PUID" ] && [ -n "$PGID" ]; then
        gosu coco ccc run-daemon
    else
        ccc run-daemon
    fi
}

# Print `ccc daemon status` (as the coco user when PUID/PGID are set, so it can
# reach the daemon socket). Errors are swallowed — callers grep the output.
daemon_status_text() {
    if [ -n "$PUID" ] && [ -n "$PGID" ]; then
        gosu coco ccc daemon status 2>/dev/null
    else
        ccc daemon status 2>/dev/null
    fi
}

run_indexer() {
    # Skip the scheduled index when the daemon is already indexing: an explicit
    # `ccc index` queues a second full pass behind the in-flight one (redundant
    # work + extra write churn). `ccc daemon status` prints "[indexing]"/"[idle]"
    # per loaded project.
    if daemon_status_text | grep -qi '\[indexing\]'; then
        echo "[cocoindex] Daemon already indexing — skipping scheduled index."
        return 0
    fi
    if [ -n "$PUID" ] && [ -n "$PGID" ]; then
        gosu coco ccc index
    else
        ccc index
    fi
}

# Background safety net: rebuild the index once a day at 03:00. Searches already
# refresh incrementally (the MCP/CLI search path defaults to refresh=True), so
# this only matters when the workspace changes without anyone searching.
indexer_loop() {
    while true; do
        now=$(date +%s)
        today3am=$(date -d 'today 03:00' +%s 2>/dev/null) || today3am=0
        tomorrow3am=$(date -d 'tomorrow 03:00' +%s 2>/dev/null) || tomorrow3am=$((now + 86400))
        if [ "$today3am" -gt "$now" ]; then
            next=$today3am
        else
            next=$tomorrow3am
        fi
        sleep_secs=$((next - now))
        echo "[cocoindex] Next index run in ${sleep_secs}s (at 03:00)"
        sleep "$sleep_secs"
        echo "[cocoindex] Running ccc index..."
        run_indexer || true
    done
}

child=""
index_child=""
trap '
    for pid in "$child" "$index_child"; do
        if [ -n "$pid" ]; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null
    exit 0
' TERM INT

# The streamable-HTTP MCP server runs inside the daemon process (enabled via
# COCOINDEX_CODE_MCP_PORT) — no separate proxy to start here.
indexer_loop &
index_child=$!

while true; do
    start_ts=$(date +%s)
    run_daemon &
    child=$!
    wait "$child" || true
    # Rate-limit respawns: sleep just long enough that successive starts are
    # >=1s apart. A clean settings-change exit with a long-running daemon
    # doesn't pay the 1s tax — only tight crash loops do.
    now=$(date +%s)
    delay=$((start_ts + 1 - now))
    if [ "$delay" -gt 0 ]; then
        sleep "$delay"
    fi
done

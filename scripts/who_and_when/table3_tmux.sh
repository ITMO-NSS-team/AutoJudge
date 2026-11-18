#!/usr/bin/env bash
set -uo pipefail
cd /home/alina/Desktop/AutoJudge
table3_output=results/table3-reconstruction-20261118-olama-v3
table3_python=/home/alina/miniconda3/envs/olama/bin/python
table3_exit_status=1
table3_launch_id=$(date -u +%Y%m%dT%H%M%SZ)
for table3_attempt in 1 2 3 4 5; do
    table3_attempt_log="$table3_output/tmux-${table3_launch_id}-attempt-${table3_attempt}.log"
    "$table3_python" -u scripts/who_and_when/table3_parallel_conditions.py run --output "$table3_output" --max-cost-usd 100 2>&1 | tee -a "$table3_output/console.log" "$table3_attempt_log"
    table3_exit_status=${PIPESTATUS[0]}
    "$table3_python" scripts/who_and_when/table3_ablation.py report --output "$table3_output" >> "$table3_output/reporter.log" 2>&1
    if [ "$table3_exit_status" -eq 0 ]; then
        break
    fi
    # A 520 without a generation ID may use the declared conservative budget policy.
    if tail -n 12 "$table3_attempt_log" | rg -q 'Missing billing information for'; then
        if "$table3_python" scripts/who_and_when/table3_reconcile_520.py "$table3_output" >> "$table3_output/reporter.log" 2>&1; then
            if [ "$table3_attempt" -lt 5 ]; then
                sleep 15
                continue
            fi
        fi
    fi
    # Retry transport failures only. Budget/validation/other accounting stops remain stops.
    if ! tail -n 12 "$table3_attempt_log" | rg -q '^http(x|core)\.(ConnectTimeout|ConnectError|ReadTimeout|ReadError|RemoteProtocolError)(:.*)?$'; then
        break
    fi
    if [ "$table3_attempt" -lt 5 ]; then
        printf 'Temporary connection failure; resuming checkpoints in 15 seconds (attempt %s/5).\n' "$((table3_attempt + 1))" | tee -a "$table3_output/console.log"
        sleep 15
    fi
done
printf 'Table 3 execution exited with status %s. See %s/report.md\n' "$table3_exit_status" "$table3_output"
exit "$table3_exit_status"

#!/usr/bin/env bash
set -u

C="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
if [ -f "$C/set_env.sh" ]; then
  source "$C/set_env.sh" >/dev/null 2>&1 || true
fi

BASE=${BASE_DIR:-/tmp/tilexr_moe_epcard_compare}
RES=${RESULTS_DIR:-/tmp/tilexr_moe_epcard_compare_results}
SOC_VERSION=${SOC_VERSION:-Ascend950}
CASE_TIMEOUT=${CASE_TIMEOUT:-1800s}
EP_LIST=${EP_LIST:-"32 64"}
BS_LIST=${BS_LIST:-"32 128 256"}
VARIANT_LIST=${VARIANT_LIST:-"baseline final"}
mkdir -p "$RES"

SUMMARY="$RES/matrix_summary.tsv"
STATUS="$RES/matrix_status.log"
: > "$SUMMARY"
: > "$STATUS"

printf "ep\tbs\texperts_per_rank\tvariant\tkernel\tstatus\trun_time_s\twall_s\thw_sim_0_s\thw_sim_1_s\ttrace_json\tcase_dir\n" > "$SUMMARY"

for ep in $EP_LIST; do
  experts_per_rank=$((128 / ep))
  for bs in $BS_LIST; do
    for variant in $VARIANT_LIST; do
      if [ "$variant" = baseline ]; then
        kernel="MoeEp${ep}Bs${bs}Baseline"
      else
        kernel="MoeEp${ep}Bs${bs}Final"
      fi

      obj="$BASE/op/$kernel/my_kernel.o"
      case_name="ep${ep}_bs${bs}_${variant}"
      out="$RES/$case_name"
      rm -rf "$out"
      mkdir -p "$out"
      runner_script="$out/run_case.sh"
      {
        echo '#!/usr/bin/env bash'
        echo 'set -euo pipefail'
        printf 'exec bash %q\n' "$BASE/run_moe_epcard_compare.sh"
      } > "$runner_script"
      chmod +x "$runner_script"

      echo "$(date '+%F %T') START $case_name kernel=$kernel experts_per_rank=$experts_per_rank obj=$obj" | tee -a "$STATUS"
      start=$(date +%s)
      set +e
      CASE_EP="$ep" CASE_BS="$bs" KERNEL_NAME="$kernel" KERNEL_OBJECT="$obj" \
        RUNNER_BIN="${RUNNER_BIN:-/tmp/tilexr_moe_epcard_compare_runner}" \
        timeout "$CASE_TIMEOUT" cannsim record -g -o "$out" -s "$SOC_VERSION" -n 0 -f "$obj" \
        "$runner_script" 2>&1 | tee "$out/record.log"
      rc=${PIPESTATUS[0]}
      set -e
      end=$(date +%s)
      wall=$((end - start))

      if grep -q "Simulation SUCCESS" "$out/record.log" && [ "$rc" -eq 0 ]; then
        status="SUCCESS"
      else
        status="FAIL_rc${rc}"
      fi

      run_time=$(grep -Eo 'run time [0-9.]+s' "$out/record.log" | tail -n1 | awk '{print $3}' | sed 's/s$//' || true)
      hw0=$(grep -Eo 'hardware sim [0-9.]+s / [0-9.]+s' "$out/record.log" | tail -n1 | awk '{print $3}' | sed 's/s$//' || true)
      hw1=$(grep -Eo 'hardware sim [0-9.]+s / [0-9.]+s' "$out/record.log" | tail -n1 | awk '{print $5}' | sed 's/s$//' || true)
      trace=$(find "$out" -path '*/report/trace_core0.json' -type f | head -n1 || true)

      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$ep" "$bs" "$experts_per_rank" "$variant" "$kernel" "$status" "${run_time:-}" "$wall" "${hw0:-}" "${hw1:-}" "$trace" "$out" >> "$SUMMARY"
      echo "$(date '+%F %T') END $case_name status=$status wall=${wall}s run_time=${run_time:-NA}s hw=${hw0:-NA}/${hw1:-NA}s trace=${trace:-NA}" | tee -a "$STATUS"
    done
  done
done

echo "$(date '+%F %T') ALL_DONE" | tee -a "$STATUS"

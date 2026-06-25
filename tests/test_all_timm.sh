#!/usr/bin/env bash
# Run test_timm_api.py across representative AuditConfig CLI combinations.
#
# Environment overrides:
#   PYTHON=python
#   DEVICE=cpu
#   MODELS="resnet18 convnext_tiny"
#   MAX_EVENTS=4
#   CASE_PATTERN="modules|operators"

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/test_timm_api.py"

PYTHON_BIN="${PYTHON:-python}"
DEVICE="${DEVICE:-cpu}"
MAX_EVENTS="${MAX_EVENTS:-4}"
CASE_PATTERN="${CASE_PATTERN:-}"

MODEL_ARGS=()
if [[ -n "${MODELS:-}" ]]; then
  # shellcheck disable=SC2206
  MODEL_ARGS=(--models ${MODELS})
fi

run_case() {
  local name="$1"
  shift

  if [[ -n "${CASE_PATTERN}" ]] && [[ ! "${name}" =~ ${CASE_PATTERN} ]]; then
    return 0
  fi

  echo
  echo "### ${name} ###"
  echo "+ ${PYTHON_BIN} ${TEST_SCRIPT} --device ${DEVICE} --max-events ${MAX_EVENTS} ${MODEL_ARGS[*]} $*"
  "${PYTHON_BIN}" "${TEST_SCRIPT}" \
    --device "${DEVICE}" \
    --max-events "${MAX_EVENTS}" \
    "${MODEL_ARGS[@]}" \
    "$@"
}

failures=0
ran=0

run_and_count() {
  local name="$1"
  shift
  if [[ -n "${CASE_PATTERN}" ]] && [[ ! "${name}" =~ ${CASE_PATTERN} ]]; then
    return 0
  fi
  ran=$((ran + 1))
  if ! run_case "${name}" "$@"; then
    failures=$((failures + 1))
  fi
}

run_and_count "none"

run_and_count "modules" \
  --modules

run_and_count "modules_shapes" \
  --modules \
  --record-shapes

run_and_count "modules_shapes_dtypes" \
  --modules \
  --record-shapes \
  --record-dtypes

run_and_count "modules_unbounded_depth" \
  --modules \
  --module-max-depth -1

run_and_count "modules_filtered_types" \
  --modules \
  --module-include-types Linear LayerNorm Attention

run_and_count "operators" \
  --operators

run_and_count "operators_flops" \
  --operators \
  --record-flops

run_and_count "operators_all_known_ops" \
  --operators \
  --op-include-names

run_and_count "combined" \
  --modules \
  --operators

run_and_count "combined_shapes" \
  --modules \
  --operators \
  --record-shapes

run_and_count "combined_flops_shapes" \
  --modules \
  --operators \
  --record-flops \
  --record-shapes

run_and_count "combined_dtypes" \
  --modules \
  --operators \
  --record-dtypes

run_and_count "combined_unknowns" \
  --modules \
  --operators \
  --include-unknown-ops \
  --include-unknown-modules \
  --op-include-names

run_and_count "combined_sync" \
  --modules \
  --operators \
  --sync

echo
echo "${ran} case(s) run; ${failures} failure(s)"
exit "${failures}"

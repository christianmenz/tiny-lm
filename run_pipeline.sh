#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PY="${PYTHON:-}"
if [[ -z "${PY}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PY="${ROOT_DIR}/.venv/bin/python"
  else
    PY="python3"
  fi
fi

STAGE="${1:-all}"
if [[ $# -gt 0 ]]; then
  shift
fi

DEVICE="${DEVICE:-auto}"
TRAIN_FILE="${TRAIN_FILE:-${ROOT_DIR}/wikitext-2/train.txt}"
VAL_FILE="${VAL_FILE:-${ROOT_DIR}/wikitext-2/validation.txt}"
TEST_FILE="${TEST_FILE:-${ROOT_DIR}/wikitext-2/test.txt}"
FINETUNE_DATA="${FINETUNE_DATA:-${ROOT_DIR}/openassist_pairs_en.txt}"

MODEL_OUT="${MODEL_OUT:-${ROOT_DIR}/tiny_gpt.pth}"
TOKENIZER_OUT="${TOKENIZER_OUT:-${ROOT_DIR}/tiny_gpt_tokenizer.json}"
FINETUNED_OUT="${FINETUNED_OUT:-${ROOT_DIR}/finetuned_tiny_gpt.pth}"

BASE_PROMPT="${BASE_PROMPT:-The history of machine learning}"
CHAT_PROMPT="${CHAT_PROMPT:-What is Python?}"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

run_train() {
  require_file "${TRAIN_FILE}" "training file"
  local args=(
    --train-file "${TRAIN_FILE}"
    --val-file "${VAL_FILE}"
    --test-file "${TEST_FILE}"
    --model-out "${MODEL_OUT}"
    --tokenizer-out "${TOKENIZER_OUT}"
    --tokenizer-type "${TOKENIZER_TYPE:-word}"
    --embed-dim "${EMBED_DIM:-128}"
    --num-heads "${NUM_HEADS:-4}"
    --num-layers "${NUM_LAYERS:-4}"
    --seq-len "${SEQ_LEN:-128}"
    --batch-size "${BATCH_SIZE:-32}"
    --epochs "${EPOCHS:-3}"
    --lr "${LR:-5e-4}"
    --stride "${STRIDE:-128}"
    --min-freq "${MIN_FREQ:-1}"
    --max-vocab "${MAX_VOCAB:-12000}"
    --device "${DEVICE}"
    --log-interval "${LOG_INTERVAL:-100}"
  )
  if [[ -f "${FINETUNE_DATA}" ]]; then
    args+=(--tokenizer-extra-file "${FINETUNE_DATA}")
  fi
  if [[ -n "${TRAIN_LIMIT_LINES:-}" ]]; then
    args+=(--limit-lines "${TRAIN_LIMIT_LINES}")
  fi
  if [[ -n "${VAL_LIMIT_LINES:-}" ]]; then
    args+=(--limit-val-lines "${VAL_LIMIT_LINES}")
  fi
  if [[ -n "${TEST_LIMIT_LINES:-}" ]]; then
    args+=(--limit-test-lines "${TEST_LIMIT_LINES}")
  fi
  "${PY}" "${ROOT_DIR}/train_tiny_gpt.py" "${args[@]}" "$@"
}

run_base_infer() {
  require_file "${MODEL_OUT}" "base model"
  require_file "${TOKENIZER_OUT}" "tokenizer"
  "${PY}" "${ROOT_DIR}/generate_tiny_gpt.py" \
    --model "${MODEL_OUT}" \
    --tokenizer "${TOKENIZER_OUT}" \
    --prompt "${BASE_PROMPT}" \
    --max-new-tokens "${BASE_TOKENS:-120}" \
    --temperature "${TEMPERATURE:-0.8}" \
    --top-k "${TOP_K:-40}" \
    --top-p "${TOP_P:-0.95}" \
    --repetition-penalty "${REPETITION_PENALTY:-1.1}" \
    --device "${DEVICE}" \
    "$@"
}

run_finetune() {
  require_file "${MODEL_OUT}" "base model"
  require_file "${TOKENIZER_OUT}" "tokenizer"
  require_file "${FINETUNE_DATA}" "fine-tune data"
  local args=(
    --data "${FINETUNE_DATA}"
    --model "${MODEL_OUT}"
    --tokenizer "${TOKENIZER_OUT}"
    --output "${FINETUNED_OUT}"
    --epochs "${FT_EPOCHS:-3}"
    --batch-size "${FT_BATCH_SIZE:-16}"
    --grad-accum "${FT_GRAD_ACCUM:-1}"
    --lr "${FT_LR:-5e-4}"
    --stride "${FT_STRIDE:-128}"
    --assistant-only-loss
    --sample-tokens "${FT_SAMPLE_TOKENS:-80}"
    --device "${DEVICE}"
    --log-interval "${FT_LOG_INTERVAL:-100}"
  )
  if [[ -n "${FT_LIMIT_LINES:-}" ]]; then
    args+=(--limit-lines "${FT_LIMIT_LINES}")
  fi
  if [[ "${TOKENIZER_TYPE:-word}" == "byte" ]]; then
    args+=(--sample-prompt "question<eol>${CHAT_PROMPT}<eol>answer<eol>")
  else
    args+=(--sample-prompt "question <eol> ${CHAT_PROMPT} <eol> answer <eol>")
  fi
  "${PY}" "${ROOT_DIR}/finetune_tiny_gpt.py" "${args[@]}" "$@"
}

run_chat_infer() {
  require_file "${FINETUNED_OUT}" "fine-tuned model"
  require_file "${TOKENIZER_OUT}" "tokenizer"
  "${PY}" "${ROOT_DIR}/generate_finetuned_tiny_gpt.py" \
    --model "${FINETUNED_OUT}" \
    --tokenizer "${TOKENIZER_OUT}" \
    --chat \
    --answer-only \
    --prompt "${CHAT_PROMPT}" \
    --max-new-tokens "${CHAT_TOKENS:-120}" \
    --temperature "${TEMPERATURE:-0.8}" \
    --top-k "${TOP_K:-40}" \
    --top-p "${TOP_P:-0.95}" \
    --repetition-penalty "${REPETITION_PENALTY:-1.1}" \
    --device "${DEVICE}" \
    "$@"
}

run_smoke() {
  SMOKE_DIR="${SMOKE_DIR:-/tmp/tiny-lm-smoke}"
  mkdir -p "${SMOKE_DIR}"
  MODEL_OUT="${SMOKE_MODEL_OUT:-${SMOKE_DIR}/smoke_tiny_gpt.pth}"
  TOKENIZER_OUT="${SMOKE_TOKENIZER_OUT:-${SMOKE_DIR}/smoke_tokenizer.json}"
  FINETUNED_OUT="${SMOKE_FINETUNED_OUT:-${SMOKE_DIR}/smoke_finetuned_tiny_gpt.pth}"
  TRAIN_LIMIT_LINES="${TRAIN_LIMIT_LINES:-120}"
  VAL_LIMIT_LINES="${VAL_LIMIT_LINES:-120}"
  TEST_LIMIT_LINES="${TEST_LIMIT_LINES:-120}"
  FT_LIMIT_LINES="${FT_LIMIT_LINES:-400}"
  EMBED_DIM="${EMBED_DIM:-64}"
  NUM_HEADS="${NUM_HEADS:-4}"
  NUM_LAYERS="${NUM_LAYERS:-2}"
  SEQ_LEN="${SEQ_LEN:-64}"
  BATCH_SIZE="${BATCH_SIZE:-8}"
  EPOCHS="${EPOCHS:-1}"
  STRIDE="${STRIDE:-64}"
  FT_BATCH_SIZE="${FT_BATCH_SIZE:-4}"
  FT_STRIDE="${FT_STRIDE:-64}"
  BASE_TOKENS="${BASE_TOKENS:-60}"
  CHAT_TOKENS="${CHAT_TOKENS:-60}"
  LOG_INTERVAL="${LOG_INTERVAL:-20}"
  FT_LOG_INTERVAL="${FT_LOG_INTERVAL:-50}"

  run_train
  echo
  run_base_infer
  echo
  run_finetune
  echo
  run_chat_infer
}

case "${STAGE}" in
  all)
    run_train "$@"
    echo
    run_base_infer "$@"
    echo
    run_finetune "$@"
    echo
    run_chat_infer "$@"
    ;;
  train)
    run_train "$@"
    ;;
  infer|base-infer)
    run_base_infer "$@"
    ;;
  finetune|fine-tune)
    run_finetune "$@"
    ;;
  chat|finetuned-infer)
    run_chat_infer "$@"
    ;;
  smoke)
    run_smoke
    ;;
  *)
    echo "Usage: $0 [all|smoke|train|infer|finetune|chat]" >&2
    exit 2
    ;;
esac

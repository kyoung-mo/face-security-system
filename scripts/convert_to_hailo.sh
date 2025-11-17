#!/usr/bin/env bash
set -euo pipefail

# 프로젝트 루트 / models 경로 계산
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODELS_DIR="${PROJECT_ROOT}/models"

echo "[INFO] PROJECT_ROOT : ${PROJECT_ROOT}"
echo "[INFO] MODELS_DIR   : ${MODELS_DIR}"

mkdir -p "${MODELS_DIR}"

convert_model () {
    local NAME="$1"
    local ONNX_PATH="${MODELS_DIR}/${NAME}.onnx"
    local HAR_PATH="${MODELS_DIR}/${NAME}_hailo.har"
    local HEF_PATH="${MODELS_DIR}/${NAME}_hailo.hef"

    if [[ ! -f "${ONNX_PATH}" ]]; then
        echo "[WARN] ${ONNX_PATH} 가 존재하지 않습니다. 건너뜁니다."
        return
    fi

    echo "========================================"
    echo "[INFO] Converting ${ONNX_PATH}"
    echo "       -> ${HAR_PATH}"
    echo "       -> ${HEF_PATH}"
    echo "========================================"

    python - << 'PY' "${ONNX_PATH}" "${HAR_PATH}" "${HEF_PATH}"
import sys
from pathlib import Path

# Hailo SDK client
from hailo_sdk_client import ClientRunner   # <-- 설치/버전 따라 모듈명 다르면 여기만 수정

onnx_path = Path(sys.argv[1])
har_path  = Path(sys.argv[2])
hef_path  = Path(sys.argv[3])

model_name = onnx_path.stem

print(f"[PY] ONNX : {onnx_path}")
print(f"[PY] HAR  : {har_path}")
print(f"[PY] HEF  : {hef_path}")
print(f"[PY] model_name = {model_name}")

# 1) ONNX -> HAR
runner = ClientRunner(hw_arch="hailo8")
hn, npz = runner.translate_onnx_model(onnx_path.as_posix(), model_name)
runner.save_har(har_path.as_posix())
print("[PY] Saved HAR")

# TODO: 필요하면 여기서 calibration / optimization 코드 추가
# ex) runner.optimize(), runner.save_har(... 재저장)

# 2) HAR -> HEF
runner = ClientRunner(har=har_path.as_posix())
hef = runner.compile()
hef_path.write_bytes(hef)
print("[PY] Saved HEF")
PY

    echo "[INFO] Done: ${NAME}"
}

# ─────────────────────────────────────────────
# 실제 변환 호출
# ─────────────────────────────────────────────
convert_model "yolov8_face"
convert_model "facenet"

echo "[INFO] All conversions finished."

import csv
import logging
from pathlib import Path
from datetime import datetime

# 이 파일 위치: face-security-system/src/utils/logging_utils.py
# 부모 구조:
#   parents[0] = src/utils
#   parents[1] = src
#   parents[2] = face-security-system  ← 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# 1) 출입 로그 CSV 기록 (기존 기능)
#    - 항상 프로젝트 루트 기준 경로로 저장되도록 보정
# ============================================================
def append_access_log(csv_path: str | Path,
                      user_id: str | None,
                      result: str,
                      distance: float | None = None):
    """
    출입 기록을 CSV 파일에 한 줄씩 추가한다.

    csv_path:
        - "logs/access_log.csv" 같이 상대경로로 넘겨도 되고
        - 절대경로도 가능함.
        항상 PROJECT_ROOT 기준으로 보정해서 저장.
    """
    path = Path(csv_path)

    # 상대경로로 들어오면 프로젝트 루트 기준으로 보정
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path.parent.mkdir(parents=True, exist_ok=True)

    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "user_id", "result", "distance"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            user_id if user_id is not None else "",
            result,
            f"{distance:.4f}" if distance is not None else ""
        ])


# ============================================================
# 2) debug.log 생성용 Python logging 설정
# ============================================================
def setup_logging(log_level: int = logging.DEBUG):
    """
    전체 프로젝트에서 공통으로 쓸 logging 설정.

    - logs/debug.log 파일 생성
    - 콘솔에도 동일한 로그 출력
    - PROJECT_ROOT/logs/debug.log 위치에 저장
    """
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    debug_log_path = log_dir / "debug.log"

    # 기본 로거 설정
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(debug_log_path, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

    # 선택: 시작 시 한 줄 남기기
    logging.getLogger(__name__).info(f"Logging started. debug.log: {debug_log_path}")

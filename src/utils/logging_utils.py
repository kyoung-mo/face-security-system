import csv
from pathlib import Path
from datetime import datetime

def append_access_log(csv_path: str | Path, user_id: str | None, result: str, distance: float | None = None):
    path = Path(csv_path)
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

from pathlib import Path
import yaml

# 프로젝트 루트: face-security-system/
# __file__ = .../src/utils/config_loader.py
# parent        -> src/utils
# parent.parent -> src
# parent.parent.parent -> face-security-system  ✅
BASE_DIR = Path(__file__).resolve().parent.parent.parent


def load_yaml(rel_path: str) -> dict:
    """
    프로젝트 루트(BASE_DIR)를 기준으로 한 상대 경로를 받아
    YAML 파일을 로드해서 dict로 반환한다.

    예:
        load_yaml("config/config.yaml")
        load_yaml("config/paths.yaml")
    """
    path = BASE_DIR / rel_path

    if not path.exists():
        raise FileNotFoundError(f"{path} not found (project root: {BASE_DIR})")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return data


def load_main_config() -> dict:
    return load_yaml("config/config.yaml")


def load_paths_config() -> dict:
    return load_yaml("config/paths.yaml")

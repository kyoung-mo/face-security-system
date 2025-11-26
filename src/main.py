import argparse
from modes.register_mode import run_register_mode
from modes.recognize_mode import run_recognize_mode
from utils.logging_utils import setup_logging

setup_logging()

def main():
    parser = argparse.ArgumentParser(description="Face Security System")
    parser.add_argument(
        "--mode",
        choices=["register", "recognize"],
        default="recognize",
        help="실행 모드 선택 (register | recognize)",
    )
    parser.add_argument(
        "--backend",
        choices=["cpu", "hailo"],
        default="cpu",
        help="추론 백엔드 선택 (cpu | hailo)",
    )

    args = parser.parse_args()

    if args.mode == "register":
        # 🔥 register 모드에서도 같은 Detector/backend를 쓸 수 있게 인자 전달
        run_register_mode(detector_backend=args.backend)
    else:
        run_recognize_mode(detector_backend=args.backend)

if __name__ == "__main__":
    main()

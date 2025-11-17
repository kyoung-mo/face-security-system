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
    args = parser.parse_args()

    if args.mode == "register":
        run_register_mode()
    else:
        run_recognize_mode()

if __name__ == "__main__":
    main()

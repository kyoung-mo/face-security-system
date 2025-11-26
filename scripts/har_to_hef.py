import os
import argparse
import traceback
from hailo_sdk_client import ClientRunner

def compile_har_to_hef(har_path: str, hef_path: str, hw_arch: str = "hailo8"):
    har_path = os.path.abspath(har_path)
    hef_path = os.path.abspath(hef_path)

    if not os.path.isfile(har_path):
        raise FileNotFoundError(f"HAR file not found: {har_path}")

    print("========================================")
    print(f"[INFO] HAR  path : {har_path}")
    print(f"[INFO] HEF path : {hef_path}")
    print(f"[INFO] HW arch  : {hw_arch}")
    print("========================================")

    try:
        runner = ClientRunner(hw_arch=hw_arch, har=har_path)

        print("[INFO] Compiling HAR to HEF...")
        hef = runner.compile()

        with open(hef_path, "wb") as f:
            f.write(hef)

        print("[INFO] Saved HEF:", hef_path)
        print("========================================")

    except Exception as e:
        print("[ERROR] Failed to compile HAR to HEF")
        print("Type   :", type(e).__name__)
        print("Message:", e)
        print("------ Traceback ------")
        traceback.print_exc()
        print("-----------------------")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Convert existing Hailo HAR file to HEF (HAR -> HEF only)."
    )
    parser.add_argument(
        "--har",
        type=str,
        default="yolov8_face_quantized.har",
        help="입력 HAR 파일 경로 (기본: yolov8_face_quantized.har)",
    )
    parser.add_argument(
        "--hef",
        type=str,
        default=None,
        help="출력 HEF 파일 경로 (생략 시 HAR 이름 기반으로 자동 생성)",
    )
    parser.add_argument(
        "--hw-arch",
        type=str,
        default="hailo8",
        help="Target HW arch (기본: hailo8)",
    )

    args = parser.parse_args()

    har_path = args.har
    if args.hef is None:
        base, _ = os.path.splitext(har_path)
        hef_path = base + ".hef"
    else:
        hef_path = args.hef

    compile_har_to_hef(har_path, hef_path, hw_arch=args.hw_arch)

if __name__ == "__main__":
    main()

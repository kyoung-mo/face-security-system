import os
import argparse
import numpy as np

from hailo_sdk_client import ClientRunner


def build_model_script(optimization_level=1, compression_level=0,
                       calibset_size=200, batch_size=8):
    """
    Hailo 모델 최적화/양자화 스크립트 생성
    """
    lines = [
        f"model_optimization_flavor(optimization_level={optimization_level}, "
        f"compression_level={compression_level})\n",
        "resources_param(max_control_utilization=1.0, "
        "max_compute_utilization=1.0, "
        "max_memory_utilization=1.0)\n",
        # fps 압박 주지 않기 위해 여기서 fps는 안 건드림
        f"model_optimization_config(calibration, batch_size={batch_size}, "
        f"calibset_size={calibset_size})\n",
    ]
    return "".join(lines)


def create_random_calib_data(input_height, input_width, input_ch, calib_size):
    """
    facenet용 random calibration 데이터 생성
    shape: (N, H, W, C), uint8
    """
    calib_data = np.random.randint(
        0, 255,
        size=(calib_size, input_height, input_width, input_ch),
        dtype=np.uint8
    )
    return calib_data


def facenet_convert(args):
    """
    facenet.onnx  → facenet_parsed.har → facenet_quantized.har → facenet_hailo.hef
    전체 파이프라인
    """
    onnx_path = args.onnx
    net_name = "facenet"

    # facenet은 Hailo 로그 기준으로 112x112x3 입력
    h, w, c = 112, 112, 3

    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    print("========================================")
    print(f"[INFO] ONNX path  : {onnx_path}")
    print(f"[INFO] Net name   : {net_name}")
    print(f"[INFO] Input      : (1, {c}, {h}, {w})")
    print(f"[INFO] Calib size : {args.calib_size}")
    print(f"[INFO] HW arch    : {args.hw_arch}")
    print("========================================")

    # 1) ONNX → 파싱된 HAR
    runner = ClientRunner(hw_arch=args.hw_arch)
    hn, npz = runner.translate_onnx_model(onnx_path, net_name)

    parsed_har = args.parsed_har
    runner.save_har(parsed_har)
    print(f"[INFO] Saved parsed HAR: {parsed_har}")

    # 2) Calibration 데이터 생성 (112x112x3)
    calib_data = create_random_calib_data(h, w, c, args.calib_size)
    print(f"[INFO] Calibration data shape: {calib_data.shape}")

    # 3) 모델 스크립트 로드 (양자화/최적화 설정)
    model_script = build_model_script(
        optimization_level=args.op,
        compression_level=args.comp,
        calibset_size=args.calib_size,
        batch_size=args.calib_batch_size,
    )
    runner.load_model_script(model_script)

    # 4) Optimize (full quantization)
    print("[INFO] Running optimization (quantization)...")
    runner.optimize(calib_data)

    quant_har = args.quant_har
    runner.save_har(quant_har)
    print(f"[INFO] Saved quantized HAR: {quant_har}")

    # 5) HAR → HEF 컴파일
    print("[INFO] Compiling HAR to HEF...")
    hef = runner.compile()
    hef_path = args.hef
    with open(hef_path, "wb") as f:
        f.write(hef)
    print(f"[INFO] Saved HEF: {hef_path}")
    print("============= DONE (facenet) ===========")


def main():
    parser = argparse.ArgumentParser(
        description="Convert facenet.onnx to Hailo HAR/HEF."
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default="facenet.onnx",
        help="facenet ONNX 경로 (기본: facenet.onnx)",
    )
    parser.add_argument(
        "--parsed-har",
        dest="parsed_har",
        type=str,
        default="facenet_parsed.har",
        help="파싱된 HAR 파일 이름 (기본: facenet_parsed.har)",
    )
    parser.add_argument(
        "--quant-har",
        dest="quant_har",
        type=str,
        default="facenet_quantized.har",
        help="양자화된 HAR 파일 이름 (기본: facenet_quantized.har)",
    )
    parser.add_argument(
        "--hef",
        type=str,
        default="facenet_hailo.hef",
        help="출력 HEF 파일 이름 (기본: facenet_hailo.hef)",
    )
    parser.add_argument(
        "--hw-arch",
        type=str,
        default="hailo8",
        help="Target HW arch (기본: hailo8)",
    )
    parser.add_argument(
        "--calib-size",
        type=int,
        default=200,
        help="Calibration 샘플 개수 (기본: 200)",
    )
    parser.add_argument(
        "--calib-batch-size",
        dest="calib_batch_size",
        type=int,
        default=8,
        help="Optimize 시 batch_size (기본: 8)",
    )
    parser.add_argument(
        "--op",
        type=int,
        default=1,
        help="optimization_level (기본: 1)",
    )
    parser.add_argument(
        "--comp",
        type=int,
        default=0,
        help="compression_level (기본: 0)",
    )

    args = parser.parse_args()
    facenet_convert(args)


if __name__ == "__main__":
    main()

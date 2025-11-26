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
        # 컴파일러 최적화 레벨만 최대한 켜주기
        "performance_param(compiler_optimization_level=max)\n",
        f"model_optimization_config(calibration, batch_size={batch_size}, "
        f"calibset_size={calibset_size})\n",
    ]
    return "".join(lines)


def get_model_config(model_name: str):
    """
    각 모델(yolov8_face, facenet)에 대한 기본 설정
    """
    if model_name == "yolov8_face":
        return {
            "onnx_path": "yolov8_face_320.onnx",
            "net_name": "yolov8_face",
            "input_height": 320,
            "input_width": 320,
            "input_ch": 3,
            # 🔥 Hailo 에러 메시지가 추천해 준 end node
            "end_nodes": ["/model.22/Concat_3"],
        }
    elif model_name == "facenet":
        return {
            "onnx_path": "facenet.onnx",
            "net_name": "facenet",
            # Facenet 입력 크기 (필요하면 112로 수정 가능)
            "input_height": 160,
            "input_width": 160,
            "input_ch": 3,
            "end_nodes": None,   # facenet은 전체 그래프 사용
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def create_random_calib_data(input_height, input_width, input_ch, calib_size):
    """
    랜덤 이미지로 calibration 데이터 생성
    shape: (N, H, W, C)
    """
    calib_data = np.random.randint(
        0, 255,
        size=(calib_size, input_height, input_width, input_ch),
        dtype=np.uint8
    )
    return calib_data


def convert_single_model(cfg, args):
    """
    ONNX 하나를:
      1) translate_onnx_model → parsed HAR
      2) optimize(calib_data) → quantized HAR
      3) compile() → HEF
    까지 처리
    """
    onnx_path = cfg["onnx_path"]
    net_name = cfg["net_name"]
    h = cfg["input_height"]
    w = cfg["input_width"]
    c = cfg["input_ch"]

    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    print(f"[INFO] Converting model: {net_name}")
    print(f"[INFO] ONNX path: {onnx_path}")
    print(f"[INFO] Input shape: (1, {c}, {h}, {w})")

    # 1) ONNX → Hailo 내부 포맷 (parse)
    runner = ClientRunner(hw_arch=args.hw_arch)

    # 🔥 yolov8_face는 DFL Reshape 에러 때문에 end_node_names 필요
    end_nodes = cfg.get("end_nodes", None)

    if end_nodes:
        hn, npz = runner.translate_onnx_model(
            onnx_path,
            net_name,
            end_node_names=end_nodes,
        )
    else:
        hn, npz = runner.translate_onnx_model(
            onnx_path,
            net_name,
        )

    parsed_har = f"{net_name}_parsed.har"
    runner.save_har(parsed_har)
    print(f"[INFO] Saved parsed HAR: {parsed_har}")

    # 2) Calibration 데이터 생성
    calib_data = create_random_calib_data(
        h, w, c, calib_size=args.calib_size
    )
    print(f"[INFO] Calibration data shape: {calib_data.shape}")

    # 3) 모델 스크립트 로드 (최적화 / 양자화 설정)
    model_script = build_model_script(
        optimization_level=args.op,
        compression_level=args.comp,
        calibset_size=args.calib_size,
        batch_size=args.calib_batch_size,
    )
    runner.load_model_script(model_script)

    # 4) Optimize(=양자화) 실행
    print("[INFO] Running optimization (quantization)...")
    runner.optimize(calib_data)

    quant_har = f"{net_name}_quantized.har"
    runner.save_har(quant_har)
    print(f"[INFO] Saved quantized HAR: {quant_har}")

    # 5) Compile → HEF 생성
    print("[INFO] Compiling to HEF...")
    hef = runner.compile()
    hef_path = f"{net_name}.hef"
    with open(hef_path, "wb") as f:
        f.write(hef)
    print(f"[INFO] Saved HEF: {hef_path}")
    print("=========================================")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX models (yolov8_face / facenet) to Hailo HEF."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolov8_face", "facenet"],
        help="변환할 모델 이름 리스트. 예) --models yolov8_face facenet",
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
        help="optimization_level (연구실 코드에서 op)",
    )
    parser.add_argument(
        "--comp",
        type=int,
        default=0,
        help="compression_level (연구실 코드에서 comp)",
    )

    args = parser.parse_args()

    for name in args.models:
        cfg = get_model_config(name)
        convert_single_model(cfg, args)


if __name__ == "__main__":
    main()

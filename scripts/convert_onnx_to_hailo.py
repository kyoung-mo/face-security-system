import os
import argparse
import numpy as np
import cv2
import glob
from hailo_sdk_client import ClientRunner


def build_model_script(optimization_level=1, compression_level=0,
                       calibset_size=200, batch_size=8, use_a16w16=False):
    lines = [
        f"model_optimization_flavor(optimization_level={optimization_level}, "
        f"compression_level={compression_level})\n",
        "performance_param(compiler_optimization_level=max)\n",
    ]
    if use_a16w16:
        lines += [
            "quantization_param({conv*}, precision_mode=a16_w16)\n",
            "\n",
        ]
    lines.append(
        f"model_optimization_config(calibration, batch_size={batch_size}, "
        f"calibset_size={calibset_size})\n"
    )
    return "".join(lines)


def get_model_config(model_name: str):
    if model_name == "yolov8_face":
        return {
            "onnx_path": "models/yolov8_face.onnx",
            "net_name": "yolov8_face",
            "input_height": 320,
            "input_width": 320,
            "input_ch": 3,
            "end_nodes": ["/model.22/Sigmoid", "/model.22/Concat"],
            "use_a16w16": False,
        }
    elif model_name == "facenet":
        return {
            "onnx_path": "models/facenet.onnx",
            "net_name": "facenet",
            "input_height": 160,
            "input_width": 160,
            "input_ch": 3,
            "end_nodes": None,
            "use_a16w16": False,   # 임베딩 정밀도 보존
        }
    elif model_name == "mobilefacenet":
        return {
            "onnx_path": "models/mobilefacenet.onnx",
            "net_name": "mobilefacenet",
            "input_height": 112,
            "input_width": 112,
            "input_ch": 3,
            "end_nodes": None,
            "use_a16w16": False,
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_real_calib_data(calib_dir, input_height, input_width, calib_size):
    image_paths = glob.glob(os.path.join(calib_dir, "*.jpg")) + \
                  glob.glob(os.path.join(calib_dir, "*.png"))

    if len(image_paths) == 0:
        raise FileNotFoundError(f"캘리브레이션 이미지 없음: {calib_dir}")

    if len(image_paths) < calib_size:
        print(f"[WARNING] 이미지 수({len(image_paths)})가 calib_size({calib_size})보다 적음 → 반복 사용")
        image_paths = (image_paths * ((calib_size // len(image_paths)) + 1))[:calib_size]

    calib_data = []
    for path in image_paths[:calib_size]:
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (input_width, input_height))
        calib_data.append(img)

    calib_array = np.array(calib_data, dtype=np.uint8)
    np.random.shuffle(calib_array)
    print(f"[INFO] 캘리브레이션 데이터 shape: {calib_array.shape}")
    return calib_array


def convert_single_model(cfg, args):
    onnx_path = cfg["onnx_path"]
    model_name = cfg["net_name"]
    net_name  = cfg["net_name"]
    h = cfg["input_height"]
    w = cfg["input_width"]

    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    print(f"\n{'='*50}")
    print(f"[INFO] 변환 시작: {net_name}")

    runner    = ClientRunner(hw_arch=args.hw_arch)
    end_nodes = cfg.get("end_nodes")

    # 1) ONNX 파싱
    if end_nodes:
        hn, npz = runner.translate_onnx_model(onnx_path, net_name, end_node_names=end_nodes)
    else:
        hn, npz = runner.translate_onnx_model(onnx_path, net_name)

    runner.save_har(f"models/{net_name}_parsed.har")
    print(f"[INFO] Parsed HAR 저장 완료")

    # 2) 실제 얼굴 이미지로 캘리브레이션
    calib_data = load_real_calib_data(args.calib_dir, h, w, args.calib_size)

    # 3) 모델 스크립트
    model_script = build_model_script(
        optimization_level=args.op,
        compression_level=args.comp,
        calibset_size=args.calib_size,
        batch_size=args.calib_batch_size,
        use_a16w16=cfg["use_a16w16"],
    )
    runner.load_model_script(model_script)

    # 4) 양자화
    print("[INFO] 양자화 실행 중...")
    runner.optimize(calib_data)
    runner.save_har(f"models/{net_name}_quantized.har")
    print(f"[INFO] Quantized HAR 저장 완료")

    # 5) HEF 컴파일
    print("[INFO] HEF 컴파일 중...")
    hef = runner.compile()
    hef_path = f"models/{net_name}.hef"
    with open(hef_path, "wb") as f:
        f.write(hef)
    runner.save_har(f"models/{net_name}_compiled.har")
    print(f"[INFO] HEF 저장 완료: {hef_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX models (yolov8_face / facenet) to Hailo HEF."
    )
    parser.add_argument("--models", nargs="+", default=["yolov8_face", "facenet"])
    parser.add_argument("--hw-arch", type=str, default="hailo8")
    parser.add_argument("--calib-dir", type=str, default="data/calib_images",
                        help="실제 얼굴 이미지 폴더 경로")
    parser.add_argument("--calib-size", type=int, default=200)
    parser.add_argument("--calib-batch-size", dest="calib_batch_size", type=int, default=8)
    parser.add_argument("--op",   type=int, default=1)
    parser.add_argument("--comp", type=int, default=0)
    args = parser.parse_args()

    for name in args.models:
        cfg = get_model_config(name)
        convert_single_model(cfg, args)


if __name__ == "__main__":
    main()

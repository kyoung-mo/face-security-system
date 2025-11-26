import cv2
import numpy as np
from pathlib import Path

from hailo_platform import (
    HEF,
    Device,
    VDevice,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
    HailoStreamInterface,
    InferVStreams,
    ConfigureParams,
)

from camera import Camera
from utils.config_loader import load_yaml


def main():
    print("\n===== HAILO RAW DEBUG =====")

    paths = load_yaml("config/paths.yaml")
    hef_path = str(Path(__file__).resolve().parent.parent / paths["models"]["yolov8_face_hailo_hef"])
    print(f"[DEBUG] HEF PATH = {hef_path}")

    # 1) Load HEF
    hef = HEF(hef_path)
    print("[Hailo RAW] Loaded HEF")

    # 2) Scan devices + create VDevice
    devices = Device.scan()
    if not devices:
        raise RuntimeError("No Hailo device found!")
    vdevice = VDevice(device_ids=devices)

    # 3) Configure network_group
    cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = vdevice.configure(hef, cfg)[0]
    ngp = network_group.create_params()

    # 4) Get input/output vstream info
    in_info = hef.get_input_vstream_infos()[0]
    out_info = hef.get_output_vstream_infos()[0]

    H, W = in_info.shape[0], in_info.shape[1]
    print(f"[Hailo RAW] INPUT SIZE = {W}x{H}")
    print(f"[Hailo RAW] OUTPUT SHAPE RAW = {out_info.shape}")

    # 5) Create vstream params
    in_params = InputVStreamParams.make_from_network_group(
        network_group,
        quantized=False,
        format_type=FormatType.FLOAT32,
    )

    out_params = OutputVStreamParams.make_from_network_group(
        network_group,
        quantized=False,
        format_type=FormatType.FLOAT32,
    )

    # 6) CAMERA
    camera = Camera(width=640, height=480, backend="picamera2")

    print("[DEBUG] Starting capture... Press Q to stop")

    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        resized = cv2.resize(frame, (W, H)).astype(np.float32)
        inputs = {in_info.name: resized[None, ...]}

        # ---- RUN INFERENCE ----
        with InferVStreams(
            network_group,
            in_params,
            out_params,
            tf_nms_format=False,
        ) as pipe:
            with network_group.activate(ngp):
                out = pipe.infer(inputs)[out_info.name]

        # Flatten output
        out2 = out.squeeze()
        print("\n========= RAW OUTPUT STATS =========")
        print("shape:", out2.shape)
        print("min:", np.min(out2))
        print("max:", np.max(out2))
        print("mean:", np.mean(out2))
        print("first row:", out2[0][:10])

        cv2.imshow("debug_raw", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

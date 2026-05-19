from pathlib import Path
import cv2
import numpy as np

from hailo_platform import (
    HEF, Device, VDevice,
    InputVStreamParams, OutputVStreamParams,
    FormatType, HailoStreamInterface,
    InferVStreams, ConfigureParams,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_HEF_PATH = BASE_DIR / "models" / "mobilefacenet_zoo.hef"


class FaceEmbedderHEF:
    def __init__(self, model_path=None):
        path = Path(model_path) if model_path else DEFAULT_HEF_PATH
        if not path.is_absolute():
            path = BASE_DIR / path
        self.model_path = str(path)
        self.input_size = (112, 112)

        print(f"[FaceEmbedderHEF] Loading: {self.model_path}")

        hef = HEF(self.model_path)
        devices = Device.scan()
        if not devices:
            raise RuntimeError("Hailo 장치를 찾을 수 없습니다.")

        self.vdevice = VDevice(device_ids=devices)
        cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        self.network_group        = self.vdevice.configure(hef, cfg)[0]
        self.network_group_params = self.network_group.create_params()

        self.input_info  = hef.get_input_vstream_infos()[0]
        self.output_info = hef.get_output_vstream_infos()[0]

        self.input_vstreams_params = InputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32)
        self.output_vstreams_params = OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32)

        print(f"[FaceEmbedderHEF] 입력: {self.input_info.name} {self.input_info.shape}")
        print(f"[FaceEmbedderHEF] 출력: {self.output_info.name}")

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(face_bgr, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        # HEF 내부에 정규화 포함 → 0~255 그대로 HWC
        return np.expand_dims(img, 0)  # (1, 112, 112, 3)

    def get_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        inp = self.preprocess(face_bgr)
        input_data = {self.input_info.name: inp}

        with InferVStreams(
            self.network_group,
            self.input_vstreams_params,
            self.output_vstreams_params,
        ) as pipeline:
            with self.network_group.activate(self.network_group_params):
                out = pipeline.infer(input_data)

        emb = out[self.output_info.name][0]  # (512,)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

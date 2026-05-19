from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from hailo_platform import (
        HEF, Device, VDevice,
        InputVStreamParams, OutputVStreamParams,
        FormatType, HailoStreamInterface,
        InferVStreams, ConfigureParams,
    )
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH       = BASE_DIR / "models" / "yolov8_face.onnx"
DEFAULT_HAILO_MODEL_PATH = BASE_DIR / "models" / "yolov8_face.hef"

# YOLOv8n 기본값
REG_MAX   = 16    # DFL regression max
STRIDES   = [8, 16, 32]
INPUT_W   = 320
INPUT_H   = 320


def _make_anchors(strides, input_h, input_w):
    """각 stride별 anchor 중심점 생성"""
    anchors = []
    for s in strides:
        grid_h = input_h // s
        grid_w = input_w // s
        gy, gx = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing='ij')
        anchor = np.stack([gx + 0.5, gy + 0.5], axis=-1).reshape(-1, 2) * s
        anchors.append(anchor)
    return np.concatenate(anchors, axis=0).astype(np.float32)  # (2100, 2)


def _dfl_decode(raw_box, reg_max=16):
    """
    DFL(Distribution Focal Loss) 디코딩
    raw_box: (N, 4*reg_max)
    return:  (N, 4) → ltrb 형식
    """
    N = raw_box.shape[0]
    raw = raw_box.reshape(N, 4, reg_max)
    # softmax
    raw = raw - raw.max(axis=-1, keepdims=True)
    exp  = np.exp(raw)
    prob = exp / exp.sum(axis=-1, keepdims=True)
    # weighted sum → 0~15 → ltrb
    idx  = np.arange(reg_max, dtype=np.float32)
    ltrb = (prob * idx).sum(axis=-1)  # (N, 4)
    return ltrb


def _ltrb_to_xyxy(ltrb, anchors):
    """ltrb (left/top/right/bottom offset) + anchor → xyxy"""
    x1 = anchors[:, 0] - ltrb[:, 0]
    y1 = anchors[:, 1] - ltrb[:, 1]
    x2 = anchors[:, 0] + ltrb[:, 2]
    y2 = anchors[:, 1] + ltrb[:, 3]
    return np.stack([x1, y1, x2, y2], axis=1)


def _nms(boxes, scores, iou_threshold=0.4):
    """간단한 NMS"""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w   = np.maximum(0, xx2 - xx1)
        h   = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_threshold]
    return keep


def _postprocess_yolo(outputs, anchors, conf_threshold, scale_x, scale_y,
                      orig_w, orig_h):
    """
    Hailo raw 출력 → bounding box 리스트
    outputs: dict {name: array}
    """
    # 출력 이름 기준으로 cv2/cv3 분류
    cv2_feats, cv3_feats = [], []
    for name, feat in sorted(outputs.items()):
        if 'cv2' in name:
            cv2_feats.append(feat)
        elif 'cv3' in name:
            cv3_feats.append(feat)

    if len(cv2_feats) != 3 or len(cv3_feats) != 3:
        # 이름 기반 분류 실패 시 순서로 분류
        items = sorted(outputs.items())
        cv2_feats = [items[0][1], items[2][1], items[4][1]]
        cv3_feats = [items[1][1], items[3][1], items[5][1]]

    all_boxes, all_scores = [], []

    for cv2_f, cv3_f, stride in zip(cv2_feats, cv3_feats, STRIDES):
        # cv2_f: (1, H, W, 4*reg_max) or (1, 4*reg_max, H, W)
        # cv3_f: (1, H, W, 1)         or (1, 1, H, W)

        # HWC 형식으로 변환
        if cv2_f.ndim == 4:
            if cv2_f.shape[1] != cv2_f.shape[2]:  # NHWC
                box_raw = cv2_f[0]   # (H, W, 64)
                cls_raw = cv3_f[0]   # (H, W, 1)
            else:                                   # NCHW
                box_raw = cv2_f[0].transpose(1,2,0)
                cls_raw = cv3_f[0].transpose(1,2,0)
        else:
            box_raw = cv2_f
            cls_raw = cv3_f

        H, W = box_raw.shape[:2]
        N = H * W

        box_raw = box_raw.reshape(N, -1)   # (N, 64)
        cls_raw = cls_raw.reshape(N, -1)   # (N, 1)

        # sigmoid → confidence
        scores = 1 / (1 + np.exp(-cls_raw[:, 0]))

        mask = scores >= conf_threshold
        if not mask.any():
            continue

        box_raw = box_raw[mask]
        scores  = scores[mask]

        # DFL 디코딩
        ltrb = _dfl_decode(box_raw, REG_MAX) * stride

        # anchor
        gy, gx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        anc = np.stack([gx + 0.5, gy + 0.5], axis=-1).reshape(-1, 2).astype(np.float32) * stride
        anc = anc[mask]

        boxes = _ltrb_to_xyxy(ltrb, anc)
        all_boxes.append(boxes)
        all_scores.append(scores)

    if not all_boxes:
        return []

    all_boxes  = np.concatenate(all_boxes,  axis=0)
    all_scores = np.concatenate(all_scores, axis=0)

    keep = _nms(all_boxes, all_scores)
    if not keep:
        return []

    detections = []
    for i in keep:
        x1, y1, x2, y2 = all_boxes[i]
        score = float(all_scores[i])

        x1 = int(np.clip(x1 * scale_x, 0, orig_w - 1))
        y1 = int(np.clip(y1 * scale_y, 0, orig_h - 1))
        x2 = int(np.clip(x2 * scale_x, 0, orig_w - 1))
        y2 = int(np.clip(y2 * scale_y, 0, orig_h - 1))

        if x2 > x1 and y2 > y1:
            detections.append((x1, y1, x2, y2, score))

    if not detections:
        return []

    largest = max(detections, key=lambda d: (d[2]-d[0]) * (d[3]-d[1]))
    return [largest]


class Detector:
    def __init__(self, model_path=None, conf_threshold: float = 0.2,
                 backend: str = "cpu"):
        self.backend = backend
        self.conf_threshold = conf_threshold

        # CPU
        if backend == "cpu":
            path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
            if not path.is_absolute():
                path = BASE_DIR / path
            self.model_path = str(path)
            print(f"[Detector] (CPU) Loading YOLO: {self.model_path}")
            self.model = YOLO(self.model_path, task="detect")

        # Hailo
        elif backend == "hailo":
            if not HAILO_AVAILABLE:
                raise ImportError("hailo_platform 패키지가 없습니다. HailoRT를 설치하세요.")

            path = Path(model_path) if model_path else DEFAULT_HAILO_MODEL_PATH
            if not path.is_absolute():
                path = BASE_DIR / path
            self.model_path = str(path)
            print(f"[Detector] (Hailo) HEF: {self.model_path}")

            hef     = HEF(self.model_path)
            devices = Device.scan()
            if not devices:
                raise RuntimeError("Hailo 장치를 찾을 수 없습니다.")

            self.vdevice = VDevice(device_ids=devices)
            cfg_params   = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
            self.network_group        = self.vdevice.configure(hef, cfg_params)[0]
            self.network_group_params = self.network_group.create_params()

            self.input_vstream_info   = hef.get_input_vstream_infos()[0]
            self.output_vstream_infos = hef.get_output_vstream_infos()

            shape = tuple(self.input_vstream_info.shape)
            if shape.index(3) == 0:
                self.hailo_format = "CHW"
                self.input_height, self.input_width = shape[1], shape[2]
            else:
                self.hailo_format = "HWC"
                self.input_height, self.input_width = shape[0], shape[1]

            print(f"[Hailo] input {self.hailo_format} H={self.input_height} W={self.input_width}")
            print(f"[Hailo] output streams: {[o.name for o in self.output_vstream_infos]}")

            self.input_vstreams_params = InputVStreamParams.make_from_network_group(
                self.network_group, quantized=False, format_type=FormatType.FLOAT32)
            self.output_vstreams_params = OutputVStreamParams.make_from_network_group(
                self.network_group, quantized=False, format_type=FormatType.FLOAT32)

            self.anchors = _make_anchors(STRIDES, self.input_height, self.input_width)

        else:
            raise ValueError(f"Unknown backend: {backend}")

    def detect(self, frame):
        h, w = frame.shape[:2]

        # CPU
        if self.backend == "cpu":
            results = self.model.predict(
                source=frame, imgsz=320,
                conf=self.conf_threshold, verbose=False)[0]
            boxes = results.boxes
            if boxes is None or len(boxes) == 0:
                return []
            xyxy  = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            dets  = [(int(x1), int(y1), int(x2), int(y2), float(c))
                     for (x1,y1,x2,y2), c in zip(xyxy, confs)]
            if not dets:
                return []
            return [max(dets, key=lambda d: (d[2]-d[0])*(d[3]-d[1]))]

        # Hailo
        elif self.backend == "hailo":
            resized = cv2.resize(frame, (self.input_width, self.input_height))
            inp = resized.astype(np.float32) / 255.0  # [0,1] 정규화

            input_data = {self.input_vstream_info.name: np.expand_dims(inp, 0)}

            with InferVStreams(
                self.network_group,
                self.input_vstreams_params,
                self.output_vstreams_params,
            ) as pipeline:
                with self.network_group.activate(self.network_group_params):
                    output_data = pipeline.infer(input_data)

            scale_x = w / self.input_width
            scale_y = h / self.input_height

            return _postprocess_yolo(
                output_data, self.anchors,
                self.conf_threshold, scale_x, scale_y, w, h
            )

    def detect_faces(self, frame, with_conf=False):
        results = self.detect(frame)
        if with_conf:
            return results
        return [(x1,y1,x2,y2) for (x1,y1,x2,y2,_) in results]
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
DEFAULT_HAILO_MODEL_PATH = BASE_DIR / "models" / "yolov8_face.hef"

REG_MAX = 16
STRIDES = [8, 16, 32]


def _dfl_decode(raw_box, reg_max=16):
    N = raw_box.shape[0]
    raw = raw_box.reshape(N, 4, reg_max)
    raw = raw - raw.max(axis=-1, keepdims=True)
    exp  = np.exp(raw)
    prob = exp / exp.sum(axis=-1, keepdims=True)
    idx  = np.arange(reg_max, dtype=np.float32)
    return (prob * idx).sum(axis=-1)


def _ltrb_to_xyxy(ltrb, anchors):
    x1 = anchors[:, 0] - ltrb[:, 0]
    y1 = anchors[:, 1] - ltrb[:, 1]
    x2 = anchors[:, 0] + ltrb[:, 2]
    y2 = anchors[:, 1] + ltrb[:, 3]
    return np.stack([x1, y1, x2, y2], axis=1)


def _nms(boxes, scores, iou_threshold=0.4):
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
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_threshold]
    return keep


def _postprocess_yolo(outputs, input_h, input_w, conf_threshold, scale_x, scale_y):
    # 출력 이름 정렬: conv41, conv42, conv52, conv53, conv62, conv63
    items = sorted(outputs.items())
    cv2_feats = [items[0][1], items[2][1], items[4][1]]  # bbox: conv41, conv52, conv62
    cv3_feats = [items[1][1], items[3][1], items[5][1]]  # class: conv42, conv53, conv63

    all_boxes, all_scores = [], []

    for cv2_f, cv3_f, stride in zip(cv2_feats, cv3_feats, STRIDES):
        # NHWC 형식 가정
        if cv2_f.ndim == 4:
            box_raw = cv2_f[0]  # (H, W, 64)
            cls_raw = cv3_f[0]  # (H, W, 1)
        else:
            box_raw = cv2_f
            cls_raw = cv3_f

        H, W = box_raw.shape[:2]
        N = H * W
        box_raw = box_raw.reshape(N, -1)
        cls_raw = cls_raw.reshape(N, -1)

        scores = cls_raw[:, 0]  # HEF에서 이미 sigmoid 적용
        mask = scores >= conf_threshold
        if not mask.any():
            continue

        box_raw = box_raw[mask]
        scores  = scores[mask]

        ltrb = _dfl_decode(box_raw, REG_MAX) * stride

        gy, gx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        anc = np.stack([gx+0.5, gy+0.5], axis=-1).reshape(-1, 2).astype(np.float32) * stride
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
        x1 = int(np.clip(x1 * scale_x, 0, input_w * scale_x - 1))
        y1 = int(np.clip(y1 * scale_y, 0, input_h * scale_y - 1))
        x2 = int(np.clip(x2 * scale_x, 0, input_w * scale_x - 1))
        y2 = int(np.clip(y2 * scale_y, 0, input_h * scale_y - 1))
        if x2 > x1 and y2 > y1:
            detections.append((x1, y1, x2, y2, score))

    if not detections:
        return []
    return [max(detections, key=lambda d: (d[2]-d[0])*(d[3]-d[1]))]


class Detector:
    def __init__(self, model_path=None, conf_threshold: float = 0.2, vdevice=None):
        path = Path(model_path) if model_path else DEFAULT_HAILO_MODEL_PATH
        if not path.is_absolute():
            path = BASE_DIR / path
        self.model_path = str(path)
        self.conf_threshold = conf_threshold

        print(f"[Detector HEF] Loading: {self.model_path}")

        hef = HEF(self.model_path)

        if vdevice is not None:
            self.vdevice = vdevice
        else:
            devices = Device.scan()
            if not devices:
                raise RuntimeError("Hailo 장치를 찾을 수 없습니다.")
            self.vdevice = VDevice(device_ids=devices)
        cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        self.network_group        = self.vdevice.configure(hef, cfg)[0]
        self.network_group_params = self.network_group.create_params()

        self.input_info  = hef.get_input_vstream_infos()[0]
        self.output_infos = hef.get_output_vstream_infos()

        shape = tuple(self.input_info.shape)
        if shape.index(min(shape)) == 0:
            self.input_h, self.input_w = shape[1], shape[2]
        else:
            self.input_h, self.input_w = shape[0], shape[1]

        print(f"[Detector HEF] input H={self.input_h} W={self.input_w}")
        print(f"[Detector HEF] outputs: {[o.name for o in self.output_infos]}")

        self.input_vstreams_params = InputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32)
        self.output_vstreams_params = OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32)

    def detect(self, frame):
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (self.input_w, self.input_h))
        inp = resized.astype(np.float32)
        input_data = {self.input_info.name: np.expand_dims(inp, 0)}

        with InferVStreams(
            self.network_group,
            self.input_vstreams_params,
            self.output_vstreams_params,
        ) as pipeline:
            with self.network_group.activate(self.network_group_params):
                output_data = pipeline.infer(input_data)

        scale_x = w / self.input_w
        scale_y = h / self.input_h

        return _postprocess_yolo(
            output_data, self.input_h, self.input_w,
            self.conf_threshold, scale_x, scale_y
        )

    def detect_faces(self, frame, with_conf=False):
        results = self.detect(frame)
        if with_conf:
            return results
        return [(x1,y1,x2,y2) for (x1,y1,x2,y2,_) in results]

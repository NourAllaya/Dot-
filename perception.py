"""Local ONNX detection. No inference API, no PyTorch dependency."""
import ast
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort


class Detector:
    def __init__(self, model=None, confidence=0.25, iou=0.45):
        model = model or Path(__file__).parent / 'models' / 'best.onnx'
        self.session = ort.InferenceSession(str(model), providers=['CPUExecutionProvider'])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.height, self.width = inp.shape[2:]
        if not isinstance(self.height, int) or not isinstance(self.width, int):
            raise ValueError('This adapter requires a fixed-size ONNX input')
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names = ast.literal_eval(meta['names'])
        self.end2end = meta.get('end2end', 'False').lower() == 'true'
        self.confidence, self.iou = confidence, iou
        self.dtype = np.float16 if inp.type == 'tensor(float16)' else np.float32

    def detect(self, frame):
        h, w = frame.shape[:2]
        scale = min(self.width / w, self.height / h)
        nw, nh = round(w * scale), round(h * scale)
        left, top = (self.width - nw) // 2, (self.height - nh) // 2
        image = np.full((self.height, self.width, 3), 114, np.uint8)
        image[top:top+nh, left:left+nw] = cv2.resize(frame, (nw, nh))
        tensor = image[:, :, ::-1].transpose(2, 0, 1)[None].astype(self.dtype) / 255
        raw = self.session.run(None, {self.input_name: tensor})[0][0]
        if self.end2end:
            boxes, scores, ids = raw[:, :4], raw[:, 4], raw[:, 5].astype(int)
        else:
            if raw.shape[0] != 4 + len(self.names):
                raise ValueError(f'Unexpected detection output: {raw.shape}')
            raw = raw.T
            ids = raw[:, 4:].argmax(axis=1)
            scores = raw[np.arange(len(raw)), 4 + ids]
            boxes = np.column_stack((raw[:, 0]-raw[:, 2]/2, raw[:, 1]-raw[:, 3]/2,
                                     raw[:, 0]+raw[:, 2]/2, raw[:, 1]+raw[:, 3]/2))
        good = np.isfinite(boxes).all(axis=1) & np.isfinite(scores) & (scores >= self.confidence)
        boxes, scores, ids = boxes[good].copy(), scores[good], ids[good]
        boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]]-left)/scale, 0, w)
        boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]]-top)/scale, 0, h)
        keep = []
        for cls in np.unique(ids):
            indices = np.flatnonzero(ids == cls)
            xywh = boxes[indices].copy()
            xywh[:, 2:] -= xywh[:, :2]
            selected = range(len(indices)) if self.end2end else np.asarray(cv2.dnn.NMSBoxes(
                xywh.tolist(), scores[indices].tolist(), self.confidence, self.iou)).reshape(-1)
            keep.extend(indices[list(selected)])
        out = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(dict(cls=self.names[int(ids[i])], u=float((x1+x2)/2), v=float((y1+y2)/2),
                            x=int(x1), y=int(y1), w=int(np.ceil(x2)-int(x1)),
                            h=int(np.ceil(y2)-int(y1)), area=int((x2-x1)*(y2-y1)),
                            conf=max(0, min(15, round(float(scores[i])*15))), score=float(scores[i])))
        return out


_detector = None


def detect(frame):
    global _detector
    if _detector is None:
        _detector = Detector()
    # Existing radio protocol has no rock_shadow class; a shadow is not a hazard.
    return [d for d in _detector.detect(frame) if d['cls'] in ('rock', 'crater', 'ice')]

import torch
import cv2
import numpy as np
from ultralytics import RTDETR
from typing import List


class KnkVision:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.redetr = RTDETR('rtdetr-l.pt')
        self.redetr.to(self.device)
        # self.cls_names = self.redetr.names
        # self.redert.names returning 1,2,3... instead of actual class names
        self.cls_names = ['person',
                          'bicycle',
                          'car',
                          'motorcycle',
                          'airplane',
                          'bus',
                          'train',
                          'truck',
                          'boat',
                          'traffic light',
                          'fire hydrant',
                          'stop sign',
                          'parking meter',
                          'bench',
                          'bird',
                          'cat',
                          'dog',
                          'horse',
                          'sheep',
                          'cow',
                          'elephant',
                          'bear',
                          'zebra',
                          'giraffe',
                          'backpack',
                          'umbrella',
                          'handbag',
                          'tie',
                          'suitcase',
                          'frisbee',
                          'skis',
                          'snowboard',
                          'sports ball',
                          'kite',
                          'baseball bat',
                          'baseball glove',
                          'skateboard',
                          'surfboard',
                          'tennis racket',
                          'bottle',
                          'wine glass',
                          'cup',
                          'fork',
                          'knife',
                          'spoon',
                          'bowl',
                          'banana',
                          'apple',
                          'sandwich',
                          'orange',
                          'broccoli',
                          'carrot',
                          'hot dog',
                          'pizza',
                          'donut',
                          'cake',
                          'chair',
                          'couch',
                          'potted plant',
                          'bed',
                          'dining table',
                          'toilet',
                          'tv',
                          'laptop',
                          'mouse',
                          'remote',
                          'keyboard',
                          'cell phone',
                          'microwave',
                          'oven',
                          'toaster',
                          'sink',
                          'refrigerator',
                          'book',
                          'clock',
                          'vase',
                          'scissors',
                          'teddy bear',
                          'hair drier',
                          'toothbrush']
        self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        self.midas.to(self.device)
        self.midas.eval()
        m_transform = torch.hub.load(
            "intel-isl/MiDaS", "transforms")
        self.midas_transform = m_transform.dpt_transform

    def depth(self, img: np.ndarray) -> np.ndarray:
        inp_img = self.midas_transform(img).to(self.device)
        with torch.no_grad():
            prediction = self.midas(inp_img)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        return prediction.cpu().numpy()

    def detect(self, img: np.ndarray):
        return self.redetr(img)

    def calc_dist(self, depth: np.ndarray, bbox: np.ndarray) -> float:
        return np.mean(depth[bbox[1]:bbox[3], bbox[0]:bbox[2]])


def main():
    vid_path = 'test_cam0.mp4'
    vision = KnkVision()
    cap = cv2.VideoCapture(vid_path)
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # depth = vision.depth(img)
                results = vision.detect(frame)[0]
                if results:
                    boxes = results.boxes.cpu().numpy()
                    for box in boxes:
                        if box.conf < 0.5:
                            continue
                        cls = int(box.cls[0])
                        cls_name = vision.cls_names[cls]
                        xyxy=box.xyxy[0].astype(int)
                        x1, y1, x2, y2 = xyxy
                        cv2.rectangle(frame, (x1, y1),
                                      (x2, y2), (0, 255, 0), 2)
                        # dist = vision.calc_dist(depth, box)
                        dist = 0
                        cv2.putText(frame, f'[{cls_name}] || {dist:.2f}m', (int(x1+3), int(
                            y1+3)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('frame_depth', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
    cap.release()


if __name__ == '__main__':
    main()

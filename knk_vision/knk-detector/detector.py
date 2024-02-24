import time
import torch
import cv2
import numpy as np
from ultralytics import RTDETR
from typing import List, Tuple, Optional


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
        # torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)
        # fix for timm>=0.9.8
        # issue : https://github.com/isl-org/ZoeDepth/issues/82
        self.zoe = torch.hub.load(
            "isl-org/ZoeDepth", "ZoeD_K", pretrained=False)
        pretrained_dict = torch.hub.load_state_dict_from_url(
            'https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_K.pt', map_location='cpu')
        self.zoe.load_state_dict(pretrained_dict['model'], strict=False)
        for b in self.zoe.core.core.pretrained.model.blocks:
            b.drop_path = torch.nn.Identity()
        self.zoe.to(self.device)
        self.zoe.eval()

    def depth(self, img: np.ndarray) -> np.ndarray:
        start_time = time.time()
        img /= 255.0
        img = torch.from_numpy(img).permute(
            2, 0, 1).unsqueeze(0).to(self.device)
        depth = self.zoe.infer(img)
        depth = depth.squeeze().detach().cpu().numpy()
        end_time = time.time()
        print(f'Inference time for ZoeDepth : {end_time-start_time:.2f}s')
        return depth

    def detect(self, img: np.ndarray):
        return self.redetr(img)

    def calc_dist(self, depth: np.ndarray, xyxy: List[int]) -> float:
        x1, y1, x2, y2 = xyxy
        dist = depth[y1:y2, x1:x2].mean()
        return dist

    def detect_and_calc_dist(self, frame: np.ndarray) -> Optional[List]:
        # return type of dict
        # {
        # class_name,
        # distance,
        # bounding_box
        # dimensions
        # }
        results = self.detect(frame)[0]
        if results:
            res_objects = []
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)
            depth = self.depth(img)
            boxes = results.boxes.cpu().numpy()
            for box in boxes:
                if box.conf < 0.5:
                    continue
                xyxy = box.xyxy[0].astype(int)
                dist = self.calc_dist(depth, xyxy)
                cls_name = self.cls_names[int(box.cls[0])]
                res_objects.append({
                    'class_name': cls_name,
                    'distance': dist,
                    'bounding_box': xyxy,
                })
            return res_objects
        else:
            return None


def main():
    vid_path = 'test_cam0.mp4'
    vision = KnkVision()
    cap = cv2.VideoCapture(vid_path)
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if ret:
                res_objects = vision.detect_and_calc_dist(frame)
                if res_objects:
                    for obj in res_objects:
                        cls_name = obj['class_name']
                        xyxy = obj['bounding_box']
                        x1, y1, x2, y2 = xyxy
                        cv2.rectangle(frame, (x1, y1),
                                      (x2, y2), (0, 255, 0), 2)
                        dist = obj['distance']
                        cv2.putText(frame, f'[{cls_name}] || {dist:.2f}m', (int(x1+3), int(
                            y1+3)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Knk-Vision', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
    cap.release()


if __name__ == '__main__':
    main()

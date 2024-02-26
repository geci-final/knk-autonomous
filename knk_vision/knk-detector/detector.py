import time
import random
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from ultralytics import RTDETR
from typing import List, Tuple, Optional
from vidar.utils.config import read_config
from vidar.utils.setup import setup_arch
from yolopv2.utils.utils import driving_area_mask, lane_line_mask, split_for_trace_model, non_max_suppression
from PIL import Image


class KnkVision:
    def __init__(self, height=480, width=640):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.height = height
        self.width = width

        # Object Detection and Lane Detection
        self.yolop = torch.jit.load('yolopv2.pt')
        self.yolop.to(self.device)
        self.yolop.eval()
        self.yolop_conf = 0.5
        self.yolop_iou = 0.45
        self.cls_names = ['car', 'truck', 'bus',
                          'motorcycle', 'bicycle', 'person']
        self.cls_colors = [[random.randint(0, 255) for _ in range(
            3)] for _ in range(len(self.cls_names))]

        packnet_cfg = read_config('packnet_config.yaml')
        self.packnet = setup_arch(packnet_cfg.arch, verbose=True)
        state_dict = torch.load(
            'PackNet_MR_selfsup_KITTI.ckpt', map_location="cpu")
        self.packnet.load_state_dict(state_dict["state_dict"], strict=False)
        self.packnet.to(self.device)
        self.packnet.eval()
        self._lane_depth_transform = transforms.Compose([transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
            transforms.ToTensor()
        ])

    def _lane_depth_transform(self, img: np.ndarray) -> torch.tensor:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self._lane_depth_transform(img)
        img = img.permute(2, 0, 1).unsqueeze(0)/255.0
        return img

    def depth(self, img: np.ndarray) -> np.ndarray:
        preprocess_start = time.time()
        img = self._lane_depth_transform(img).to(self.device)
        preprocess_end = time.time()
        depth = self.packnet(img)
        depth = depth[0].squeeze().detach().cpu().numpy()
        depth_end = time.time()
        print(
            f'Preprocess time : {(preprocess_end-preprocess_start)*100:.2f}ms , Inference time for Vidar : {(depth_end-preprocess_end)*100:.2f}ms')
        return depth

    def lane_obj_detect(self, img: np.ndarray):
        img = self._lane_depth_transform(img).to(self.device)
        with torch.no_grad():
            pred = self.yolop(img)
        return pred

    def calc_dist(self, depth: np.ndarray, xyxy: List[int]) -> float:
        x1, y1, x2, y2 = xyxy
        dist = depth[y1:y2, x1:x2].mean()
        return dist

    def vision_analyze(self, frame: np.ndarray) -> Optional[List]:
        # return type of dict
        # {
        # class_name,
        # distance,
        # bounding_box
        # dimensions
        # }
        [pred, anchor_grid], seg, ll = self.lane_obj_detect(frame)
        pred = split_for_trace_model(pred, anchor_grid)
        pred = non_max_suppression(
            pred, self.yolop_conf, self.yolop_iou, classes=None, agnostic=None)
        da_seg_mask = driving_area_mask(seg)
        ll_seg_make = lane_line_mask(ll)
        det=pred[0]
        


def main():
    vid_path = "test_vid/test_vid.mp4"
    vision = KnkVision()
    cap = cv2.VideoCapture(vid_path)
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (vision.width, vision.height))
                res_objects = vision.vision_analyze(frame)
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

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
from yolop.core.general import non_max_suppression, scale_coords
from yolop.utils.augmentations import letterbox_for_img
from PIL import Image


class KnkVision:
    def __init__(self, height=480, width=640):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.height = height
        self.width = width

        # Object Detection and Lane Detection
        self.yolop = torch.hub.load(
            'hustvl/yolop', 'yolop', pretrained=True, trust_repo=True)
        self.yolop.to(self.device)
        self.yolop.eval()
        self.yolop_conf = 0.5
        self.yolop_iou = 0.45
        self.cls_names = ['vehicle']
        self.cls_colors = [[random.randint(0, 255) for _ in range(
            3)] for _ in range(len(self.cls_names))]

        packnet_cfg = read_config('configs/packnet_config.yaml')
        self.packnet = setup_arch(packnet_cfg.arch, verbose=True)
        state_dict = torch.load(
            'weights/PackNet_MR_selfsup_KITTI.ckpt', map_location=self.device)
        self.packnet.load_state_dict(state_dict["state_dict"], strict=False)
        self.packnet.to(self.device)
        self.packnet.eval()
        self._l_d_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        ])

    def _depth_transform(self, img: np.ndarray) -> torch.tensor:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self._l_d_transform(img)
        img = img.unsqueeze(0)/255.0
        return img

    def depth(self, img: np.ndarray) -> np.ndarray:
        preprocess_start = time.time()
        img = self._depth_transform(img).to(self.device)
        preprocess_end = time.time()
        print("depth img shape : ", img.shape)
        depth = self.packnet(img)
        depth = depth[0].squeeze().detach().cpu().numpy()
        depth_end = time.time()
        print(
            f'Preprocess time : {(preprocess_end-preprocess_start)*100:.2f}ms , Inference time for Vidar : {(depth_end-preprocess_end)*100:.2f}ms')
        return depth

    def yolop_infer(self, img_det: np.ndarray):
        img, ratio, pad = letterbox_for_img(img_det)
        h0, w0 = img_det.shape[:2]
        h, w = img.shape[:2]
        shapes = (h0, w0), ((h/h0, w/w0), pad)
        img = np.ascontiguousarray(img)
        img = self._l_d_transform(img).to(self.device)
        img = img.unsqueeze(0)
        print("lane image shape : ", img.shape)
        with torch.no_grad():
            det_out, da_seg_out, ll_seg_out = self.yolop(img)
        inf_out, _ = det_out
        det_pred = non_max_suppression(
            inf_out, conf_thres=self.yolop_conf, iou_thres=self.yolop_iou, classes=None, agnostic=False)
        det = det_pred[0]
        _, _, height, width = img.shape
        h, w, _ = img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        da_predict = da_seg_out[:, :, pad_h:(
            height-pad_h), pad_w:(width-pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(
            da_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()

        ll_predict = ll_seg_out[:, :, pad_h:(
            height-pad_h), pad_w:(width-pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(
            ll_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

        if len(det):
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], img_det.shape).round()

        return det, da_seg_mask, ll_seg_mask

    def calc_dist(self, depth: np.ndarray, xyxy: List[int]) -> float:
        x1, y1, x2, y2 = xyxy
        dist = depth[y1:y2, x1:x2].mean()
        return dist

    def vision_analyze(self, frame: np.ndarray) -> Optional[List]:
        # type Det= #{
        # class_name,
        # distance,
        # bounding_box
        # dimensions,
        # }
        # return {
        # obj_det:List[Det],
        # drive_area_mask:tensor,
        # lane_line_mask:tensor
        # }
        det, da_seg, ll_seg = self.yolop_infer(frame)
        print("det : ", det)
        obj_det = []
        if len(det):
            depth = self.depth(frame)
            for *xyxy, conf, cls in reversed(det):
                cls_name = self.cls_names[int(cls)]
                if conf > self.yolop_conf:
                    x1, y1, x2, y2 = xyxy
                    dist = self.calc_dist(
                        depth, [int(x1), int(y1), int(x2), int(y2)])
                    obj_det.append({
                        'class_name': cls_name,
                        'distance': dist,
                        'bounding_box': (int(x1), int(y1), int(x2), int(y2))
                    })

        return {
            'obj_det': obj_det,
            'drive_area_mask': da_seg,
            'lane_line_mask': ll_seg
        }


def main():
    vid_path = "test_vid/test_vid.mp4"
    # vid_path = "example.jpg"
    vision = KnkVision()
    cap = cv2.VideoCapture(vid_path)
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (vision.width, vision.height))
                vision_res = vision.vision_analyze(frame)
                print("vision res : ", vision_res)
                res_objects = vision_res['obj_det']
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

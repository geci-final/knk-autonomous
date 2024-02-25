import time
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from ultralytics import RTDETR
from typing import List, Tuple, Optional
from vidar.utils.config import read_config
from vidar.utils.setup import setup_arch
from ufld2.utils.config import Config as LaneConfig
from ufld2.utils.knk_common import get_model
from ufld2.data.constant import culane_row_anchor, culane_col_anchor
from PIL import Image


class KnkVision:
    def __init__(self, height=480, width=640):
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
        # self.zoe = torch.hub.load(
        #     "isl-org/ZoeDepth", "ZoeD_K", pretrained=False)
        # pretrained_dict = torch.hub.load_state_dict_from_url(
        #     'https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_K.pt', map_location='cpu')
        # self.zoe.load_state_dict(pretrained_dict['model'], strict=False)
        # for b in self.zoe.core.core.pretrained.model.blocks:
        #     b.drop_path = torch.nn.Identity()
        # self.zoe.to(self.device)
        # self.zoe.eval()
        self.height = height
        self.width = width
        packnet_cfg = read_config('packnet_config.yaml')
        self.packnet = setup_arch(packnet_cfg.arch, verbose=True)
        state_dict = torch.load(
            'PackNet_MR_selfsup_KITTI.ckpt', map_location="cpu")
        self.packnet.load_state_dict(state_dict["state_dict"], strict=False)
        self.packnet.to(self.device)
        self.packnet.eval()

        # setup lane detection model using Ultra Fast Lane Detection v2
        lane_cfg = LaneConfig.fromfile("culane_res18.py")
        self.ufld2 = get_model(lane_cfg)
        ufld2_state_dict = torch.load(
            "culane_res18.pth", map_location="cpu")['model']
        ufld2_compatible_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                ufld2_compatible_state_dict[k[7:]] = v
            else:
                ufld2_compatible_state_dict[k] = v
        self.ufld2.load_state_dict(ufld2_compatible_state_dict, strict=False)
        self.ufld2.to(self.device)
        self.ufld2.eval()

        self.ufld2_row_anchor = culane_row_anchor
        self.ufld2_col_anchor = culane_col_anchor
        self.ufld2_transform = transforms.Compose([
            # transforms.Resize((int(self.lane_cfg.train_height /
            #                   self.lane_cfg.crop_ratio), self.lane_cfg.train_width)),
            transforms.Resize((lane_cfg.train_height,
                              lane_cfg.train_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def _lane_pred2coords(self, pred, row_anchor, col_anchor, local_width=1, original_image_width=1640, original_image_height=590) -> List:
        batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
        batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape
        max_indices_row = pred['loc_row'].argmax(1).cpu()
        # n , num_cls, num_lanes
        valid_row = pred['exist_row'].argmax(1).cpu()
        # n, num_cls, num_lanes
        max_indices_col = pred['loc_col'].argmax(1).cpu()
        # n , num_cls, num_lanes
        valid_col = pred['exist_col'].argmax(1).cpu()
        # n, num_cls, num_lanes
        pred['loc_row'] = pred['loc_row'].cpu()
        pred['loc_col'] = pred['loc_col'].cpu()
        coords = []
        row_lane_idx = [1, 2]
        col_lane_idx = [0, 3]
        for i in row_lane_idx:
            tmp = []
            if valid_row[0, :, i].sum() > num_cls_row / 2:
                for k in range(valid_row.shape[1]):
                    if valid_row[0, k, i]:
                        all_ind = torch.tensor(list(range(max(0, max_indices_row[0, k, i] - local_width), min(
                            num_grid_row-1, max_indices_row[0, k, i] + local_width) + 1)))

                        out_tmp = (pred['loc_row'][0, all_ind, k, i].softmax(
                            0) * all_ind.float()).sum() + 0.5
                        out_tmp = out_tmp / \
                            (num_grid_row-1) * original_image_width
                        if k < len(row_anchor):
                            tmp.append(
                                (int(out_tmp), int(row_anchor[k] * original_image_height)))
                coords.append(tmp)

        for i in col_lane_idx:
            tmp = []
            if valid_col[0, :, i].sum() > num_cls_col / 4:
                for k in range(valid_col.shape[1]):
                    if valid_col[0, k, i]:
                        all_ind = torch.tensor(list(range(max(0, max_indices_col[0, k, i] - local_width), min(
                            num_grid_col-1, max_indices_col[0, k, i] + local_width) + 1)))

                        out_tmp = (pred['loc_col'][0, all_ind, k, i].softmax(
                            0) * all_ind.float()).sum() + 0.5

                        out_tmp = out_tmp / \
                            (num_grid_col-1) * original_image_height
                        if k < len(col_anchor):
                            tmp.append(
                                (int(col_anchor[k] * original_image_width), int(out_tmp)))
                coords.append(tmp)

        return coords

    def _packnet_transform(self, img: np.ndarray) -> torch.tensor:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float()
        img = img.permute(2, 0, 1).unsqueeze(0)/255.0
        return img

    def lane_detect(self, img: np.ndarray) -> List:
        preprocess_start = time.time()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.ufld2_transform(img).unsqueeze(0).to(self.device)
        print("Lane image shpae : ", img.shape)
        preprocess_end = time.time()
        with torch.no_grad():
            pred = self.ufld2(img)
        infer_end = time.time()
        post_process_start = time.time()
        coords = self._lane_pred2coords(pred, self.ufld2_row_anchor, self.ufld2_col_anchor,
                                        original_image_width=self.width, original_image_height=self.height)
        post_process_end = time.time()
        print(
            f'Lane Detection | Preprocess time : {(preprocess_end-preprocess_start)*100:.2f}ms , Inference time : {(infer_end-preprocess_end)*100:.2f}ms, Postprocess time : {(post_process_end-post_process_start)*100:.2f}ms')
        return coords

    def depth(self, img: np.ndarray) -> np.ndarray:
        preprocess_start = time.time()
        img = self._packnet_transform(img).to(self.device)
        preprocess_end = time.time()
        depth = self.packnet(img)
        depth = depth[0].squeeze().detach().cpu().numpy()
        depth_end = time.time()
        print(
            f'Preprocess time : {(preprocess_end-preprocess_start)*100:.2f}ms , Inference time for Vidar : {(depth_end-preprocess_end)*100:.2f}ms')
        return depth

    def obj_detect(self, img: np.ndarray):
        return self.redetr(img)

    def calc_dist(self, depth: np.ndarray, xyxy: List[int]) -> float:
        x1, y1, x2, y2 = xyxy
        dist = depth[y1:y2, x1:x2].mean()
        return dist

    def detect_obj_and_dist(self, frame: np.ndarray) -> Optional[List]:
        # return type of dict
        # {
        # class_name,
        # distance,
        # bounding_box
        # dimensions
        # }
        results = self.obj_detect(frame)[0]
        if results:
            res_objects = []
            depth = self.depth(frame)
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

    def draw_lanes(self, frame: np.ndarray, lanes_coord: List) -> np.ndarray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for lane in lanes_coord:
            lane = np.array(lane).reshape((-1, 1, 2))
            cv2.polylines(frame, [lane], False, (0, 255, 255), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame


def main():
    vid_path = "test_vid/test_vid.mp4"
    vision = KnkVision()
    cap = cv2.VideoCapture(vid_path)
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (vision.width, vision.height))
                res_objects = vision.detect_obj_and_dist(frame)
                lanes_coord = vision.lane_detect(frame)
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
                frame = vision.draw_lanes(frame, lanes_coord)
                cv2.imshow('Knk-Vision', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
    cap.release()


if __name__ == '__main__':
    main()

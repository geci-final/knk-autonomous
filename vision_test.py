import cv2
import os
from knk_vision.vision import KnkVision


def main():
    vid_path = "test_vid/test_vid1.mp4"
    if not os.path.exists(vid_path):
        print(f"Video file not found: {vid_path}")
        return
    vision = KnkVision()
    cap = cv2.VideoCapture(vid_path)
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (vision.width, vision.height))
                vision_res = vision.vision_analyze(frame)
                da_seg_mask = vision_res['drive_area_mask']
                ll_seg_mask = vision_res['lane_line_mask']
                frame = vision.draw_seg(frame, (da_seg_mask, ll_seg_mask))
                res_objects = vision_res['obj_det']
                if res_objects:
                    vision.draw_obj(frame, res_objects)
                cv2.imshow('Knk-Vision', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
    cap.release()


if __name__ == '__main__':
    main()

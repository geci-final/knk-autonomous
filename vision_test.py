import cv2
from knk_vision.vision import KnkVision


def main():
    vid_path = "knk_vision/test_vid/test_cam1.mp4"
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
                    for obj in res_objects:
                        cls_name = obj['class_name']
                        xyxy = obj['bounding_box']
                        x1, y1, x2, y2 = xyxy
                        cv2.rectangle(frame, (x1, y1),
                                      (x2, y2), (0, 255, 0), 2)
                        dist = obj['distance']
                        cv2.putText(frame, f'{dist:.2f}m', (int(x1+3), int(
                            y1+3)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Knk-Vision', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
    cap.release()


if __name__ == '__main__':
    main()

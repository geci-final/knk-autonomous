from knk_vision.vision import KnkVision
from metadrive import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.component.sensors.rgb_camera import RGBCamera
import cv2


def main():
    vision = KnkVision()
    res = (800, 600)
    config = dict(
        num_scenarios=100,
        agent_policy=IDMPolicy,
        traffic_density=0.2,
        image_observation=True,
        use_render=False,
        vehicle_config=dict(image_source="rgb_camera"),
        sensors={"rgb_camera": (RGBCamera, *res)},
        show_interface=False,
        show_logo=False,
        show_fps=False,
    )
    env = MetaDriveEnv(config)
    try:
        o, i = env.reset()
        frame = o["image"][..., -1]
        cv2.imshow('Knk-Vision-sim', frame)
        for i in range(20000):
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            o, r, d, _, _ = env.step(env.action_space.sample())
            frame = o["image"][..., -1]
            frame = cv2.resize(frame, (vision.width, vision.height))
            vision_res = vision.vision_analyze(frame)
            da_seg_mask = vision_res['drive_area_mask']
            ll_seg_mask = vision_res['lane_line_mask']
            # frame = vision.draw_seg(frame, (da_seg_mask, ll_seg_mask))
            res_objects = vision_res['obj_det']
            if res_objects:
                vision.draw_obj(frame, res_objects,dist=True)
            cv2.imshow('Knk-Vision-sim', frame)
            cv2.waitKey(1)
            if d:
                env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()

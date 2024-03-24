from knk_vision import KnkVision
from metadrive import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.component.sensors.rgb_camera import RGBCamera
import cv2


def main():
    vision=KnkVision()
    res = (800, 600)
    config = dict(
        num_scenarios=100,
        agent_policy=IDMPolicy,
        traffic_density=1.0,
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
            cv2.imshow('Knk-Vision-sim', frame)
            cv2.waitKey(1)
            if d:
                env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()

from transformers import PretrainedConfig
from typing import List


class MonoSceneConfig(PretrainedConfig):

    def __init__(
        self,
        dataset="kitti",
        n_classes=20,
        feature=64,
        project_scale=2,
        full_scene_size=(256, 256, 32),
        **kwargs,
    ):
        self.dataset = dataset
        self.n_classes = n_classes
        self.feature = feature
        self.project_scale = project_scale
        self.full_scene_size = full_scene_size
        super().__init__(**kwargs)






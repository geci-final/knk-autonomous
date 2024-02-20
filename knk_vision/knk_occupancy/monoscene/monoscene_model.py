from transformers import PreTrainedModel
from .config import MonoSceneConfig
from monoscene.monoscene import MonoScene


class MonoSceneModel(PreTrainedModel):
    config_class = MonoSceneConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = MonoScene(
            dataset=config.dataset,
            n_classes=config.n_classes,
            feature=config.feature,
            project_scale=config.project_scale,
            full_scene_size=config.full_scene_size
        )
     

    def forward(self, tensor):
        return self.model.forward(tensor)
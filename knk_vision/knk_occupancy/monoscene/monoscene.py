import pytorch_lightning as pl
import torch
import torch.nn as nn
from monoscene.unet3d_nyu import UNet3D as UNet3DNYU
from monoscene.unet3d_kitti import UNet3D as UNet3DKitti
from monoscene.flosp import FLoSP
import numpy as np
import torch.nn.functional as F
from monoscene.unet2d import UNet2D
import time


class MonoScene(pl.LightningModule):
    def __init__(
        self,
        n_classes,
        feature,
        project_scale,
        full_scene_size,
        dataset,
        project_res=["1", "2", "4", "8"],
        n_relations=4,
        context_prior=True,
        fp_loss=True,
        frustum_size=4,
        relation_loss=False,
        CE_ssc_loss=True,
        geo_scal_loss=True,
        sem_scal_loss=True,
        lr=1e-4,
        weight_decay=1e-4,
    ):
        super().__init__()

        self.project_res = project_res
        self.fp_loss = fp_loss
        self.dataset = dataset
        self.context_prior = context_prior
        self.frustum_size = frustum_size
        self.relation_loss = relation_loss
        self.CE_ssc_loss = CE_ssc_loss
        self.sem_scal_loss = sem_scal_loss
        self.geo_scal_loss = geo_scal_loss
        self.project_scale = project_scale
        self.lr = lr
        self.weight_decay = weight_decay

        self.projects = {}
        self.scale_2ds = [1, 2, 4, 8]  # 2D scales
        for scale_2d in self.scale_2ds:
            self.projects[str(scale_2d)] = FLoSP(
                full_scene_size, project_scale=self.project_scale, dataset=self.dataset
            )
        self.projects = nn.ModuleDict(self.projects)

        self.n_classes = n_classes
        if self.dataset == "NYU":
            self.net_3d_decoder = UNet3DNYU(
                self.n_classes,
                nn.BatchNorm3d,
                n_relations=n_relations,
                feature=feature,
                full_scene_size=full_scene_size,
                context_prior=context_prior,
            )
        elif self.dataset == "kitti":
            self.net_3d_decoder = UNet3DKitti(
                self.n_classes,
                nn.BatchNorm3d,
                project_scale=project_scale,
                feature=feature,
                full_scene_size=full_scene_size,
                context_prior=context_prior,
            )
        self.net_rgb = UNet2D.build(out_feature=feature, use_decoder=True)

    def forward(self, batch):

        img = batch["img"].cuda()
        bs = len(img)

        out = {}

        start_rgb = time.time()
        x_rgb = self.net_rgb(img)
        end_rgb = time.time()
        print(f"RGB Time: {end_rgb-start_rgb:.2f} seconds")

        x3ds = []
        start_feat3d = time.time()
        for i in range(bs):
            x3d = None
            for scale_2d in self.project_res:

                # project features at each 2D scale to target 3D scale
                scale_2d = int(scale_2d)
                projected_pix = batch["projected_pix_{}".format(
                    self.project_scale)][i].cuda()
                fov_mask = batch["fov_mask_{}".format(
                    self.project_scale)][i].cuda()

                # Sum all the 3D features
                if x3d is None:
                    x3d = self.projects[str(scale_2d)](
                        x_rgb["1_" + str(scale_2d)][i],
                        # torch.div(projected_pix, scale_2d, rounding_mode='floor'),
                        projected_pix // scale_2d,
                        fov_mask,
                    )
                else:
                    x3d += self.projects[str(scale_2d)](
                        x_rgb["1_" + str(scale_2d)][i],
                        # torch.div(projected_pix, scale_2d, rounding_mode='floor'),
                        projected_pix // scale_2d,
                        fov_mask,
                    )
            x3ds.append(x3d)
        end_feat3d = time.time()
        print(f"Feat3D Time: {end_feat3d-start_feat3d:.2f} seconds")
        input_dict = {
            "x3d": torch.stack(x3ds),
        }
        start_3d = time.time()
        out_dict = self.net_3d_decoder(input_dict)
        end_3d = time.time()
        print(f"3D Time: {end_3d-start_3d:.2f} seconds")

        ssc_pred = out_dict["ssc_logit"]

        y_pred = ssc_pred.detach().cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)

        return y_pred

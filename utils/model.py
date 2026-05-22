from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .config import FPN_CHANNELS, NUM_CLASSES


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DetectionHead(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.cls_tower = nn.Sequential(
            ConvBNAct(in_ch, in_ch),
            ConvBNAct(in_ch, in_ch),
        )
        self.reg_tower = nn.Sequential(
            ConvBNAct(in_ch, in_ch),
            ConvBNAct(in_ch, in_ch),
        )

        self.cls_out = nn.Conv2d(in_ch, num_classes, kernel_size=1)
        self.reg_out = nn.Conv2d(in_ch, 4, kernel_size=1)
        self.center_out = nn.Conv2d(in_ch, 1, kernel_size=1)

        self._init_params()

    def _init_params(self) -> None:
        nn.init.normal_(self.cls_out.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.cls_out.bias, -2.2)

        nn.init.normal_(self.reg_out.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.reg_out.bias, 1.0)

        nn.init.normal_(self.center_out.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.center_out.bias, -2.2)

    def forward(self, x: torch.Tensor):
        cls_feat = self.cls_tower(x)
        reg_feat = self.reg_tower(x)

        cls_logits = self.cls_out(cls_feat)
        reg_preds = F.relu(self.reg_out(reg_feat))
        center_logits = self.center_out(reg_feat)
        center_probs = torch.sigmoid(center_logits)

        return {
            "cls_logits": cls_logits,
            "reg_preds": reg_preds,
            "center_logits": center_logits,
            "center_probs": center_probs,
        }


class AnchorFreeDetector(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        backbone_name: str = "resnet34",
        feat_channels: int = FPN_CHANNELS,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name

        if backbone_name == "resnet34":
            backbone = self._build_resnet34(pretrained)
        else:
            backbone = self._build_resnet18(pretrained)
        out_ch = 512

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.head_in = ConvBNAct(out_ch, feat_channels, k=1, s=1, p=0)
        self.head = DetectionHead(feat_channels, num_classes)

    @staticmethod
    def _build_resnet18(pretrained: bool):
        if not pretrained:
            return models.resnet18(weights=None)
        try:
            return models.resnet18(weights="IMAGENET1K_V1")
        except Exception:
            try:
                from torchvision.models import ResNet18_Weights

                return models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            except Exception:
                return models.resnet18(weights=None)

    @staticmethod
    def _build_resnet34(pretrained: bool):
        if not pretrained:
            return models.resnet34(weights=None)
        try:
            return models.resnet34(weights="IMAGENET1K_V1")
        except Exception:
            try:
                from torchvision.models import ResNet34_Weights

                return models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            except Exception:
                return models.resnet34(weights=None)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head_in(x)
        outputs = self.head(x)
        pred = torch.cat(
            [outputs["cls_logits"], outputs["reg_preds"], outputs["center_logits"]],
            dim=1,
        ).permute(0, 2, 3, 1)
        outputs["pred"] = pred
        return outputs

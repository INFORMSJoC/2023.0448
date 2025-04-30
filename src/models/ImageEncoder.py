import os
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from transformers import ViTFeatureExtractor, ViTModel
from ..models.classifier import Classifier
# from classifier import Classifier
from transformers import AutoModelForImageClassification

# vit base model from https://huggingface.co/google/vit-base-patch16-224
# vit large model from https://huggingface.co/google/vit-large-patch16-224

class ViTClf(nn.Module):
    def __init__(self, args, image_encoder='vit_base'):
        """
        image_encoder: base / large
        """
        super(ViTClf, self).__init__()
        assert image_encoder in ['vit_base', 'vit_large']

        # directory is fine
        if image_encoder in ['vit_base']:
            # self.tokenizer = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
            self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")
        else:
            # self.tokenizer = ViTFeatureExtractor.from_pretrained("google/vit-large-patch16-224")
            self.image_encoder = ViTModel.from_pretrained("google/vit-large-patch16-224")

        self.clf = Classifier(dropout=args.dropout, in_dim=768, post_dim=256, out_dim=args.n_classes)


    def forward(self, pixel_values):
        """
        pixel_values:
        """
        # pixel_values = self.tokenizer(images=image, return_tensors="pt").pixel_values
        x = self.image_encoder(pixel_values=pixel_values).last_hidden_state[:, 0, :]
        out = self.clf(x)
        return out


# torchvision.models.vit_b_16
class VitImageEncoder(nn.Module):
    def __init__(self):
        super(VitImageEncoder, self).__init__()

        # self.model=torchvision.models.vit_b_16(weights="IMAGENET1K_V1")
        self.model=torchvision.models.vit_b_16(weights="IMAGENET1K_SWAG_E2E_V1")
        # self.model=torchvision.models.vit_l_16(weights="IMAGENET1K_SWAG_LINEAR_V1")
        # self.model=torchvision.models.vit_h_14(weights="IMAGENET1K_SWAG_LINEAR_V1")

        # self.model=torch.hub.load("pytorch/vision", "vit_b_16", weights="IMAGENET1K_V1")
        # self.model=torch.hub.load("pytorch/vision", "vit_b_16", weights="IMAGENET1K_SWAG_E2E_V1")
        # self.model=torch.hub.load("pytorch/vision", "vit_l_16", weights="IMAGENET1K_SWAG_LINEAR_V1")
        # self.model=torch.hub.load("pytorch/vision", "vit_h_14", weights="IMAGENET1K_SWAG_LINEAR_V1")

    def forward(self, x):
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)
        out = x[:, 0]
        return out


class torchViTClf(nn.Module):
    def __init__(self, args):
        """
        image_encoder: base / large
        """
        super(torchViTClf, self).__init__()
        self.image_encoder = VitImageEncoder()
        in_features=self.image_encoder.model.heads.head.in_features
        self.clf = Classifier(dropout=args.dropout, in_dim=in_features, post_dim=256, out_dim=args.n_classes)

    def forward(self, x):
        """
        pixel_values:
        """
        x = self.image_encoder(x)
        out = self.clf(x)
        return out


if __name__ == "__main__":
    x=torch.randn(1,3,224,224)
    vit_normal = VitImageEncoder()
    a=vit_normal(x)



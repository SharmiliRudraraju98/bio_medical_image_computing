#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import argparse
import torchvision
import numpy as np
import torchio as tio
import SimpleITK as sitk
import nibabel as nib
import pytorch_lightning as pl
from monai.inferers import sliding_window_inference
from pathlib import Path

from utils.general import (
    unified_img_reading,
    busca_path,
    salvaImageRebuilt,
    convert_to_nifti,
    collect_images_verbose,
    post_processing_lung,
)

from model.unet_diedre import UNet_Diedre
from utils.transform3D import CTHUClip


HOME = os.getenv("HOME")
TEMP_IMAGES = 'temp_images'
RAW_DATA_FOLDER = 'raw_images'


# ---------------------------------------------------------
#               CPU-SAFE get_sample_image()
# ---------------------------------------------------------
def get_sample_image(npz_path):
    ID_image = os.path.basename(npz_path).replace('.npz', '')
    print(f'\tImage name: {ID_image}')

    npz = np.load(npz_path)
    img = npz["image"][:].astype(np.float32)

    img = img.transpose(2, 1, 0)

    if len(img.shape) == 3:
        img = np.expand_dims(img, 0)

    subject = tio.Subject(image=tio.ScalarImage(tensor=img))
    transform = tio.Resize((128, 128, 128))
    transformed = transform(subject)
    img_high = transformed.image.numpy()

    # CPU tensors
    img_high = torch.tensor(img_high, dtype=torch.float32).unsqueeze(0)
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    return {"image_h": img_high, "image": img}


# ---------------------------------------------------------
#                   CPU-ONLY LungModule
# ---------------------------------------------------------
class LungModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        if self.hparams.mode == "segmentation":
            self.model_low = UNet_Diedre(
                n_channels=1, n_classes=1,
                norm="instance", dim='3d',
                init_channel=16, joany_conv=False, dict_return=False
            )
            self.model = UNet_Diedre(
                n_channels=2, n_classes=1,
                norm="instance", dim='3d',
                init_channel=16, joany_conv=False, dict_return=False
            )

    # ---------------------------------------------------------
    #           High-resolution lung prediction (CPU)
    # ---------------------------------------------------------
    def forward_per_lobe(self, x, y_seg_resize):

        x_new = torch.cat((x, y_seg_resize), dim=1)

        output_lung = sliding_window_inference(
            x_new,                         # CPU only
            roi_size=(128, 128, 128),
            sw_batch_size=1,
            predictor=self.model,           # stays on CPU
            mode="gaussian",
            progress=False,
            device=torch.device('cpu')
        )

        return output_lung.sigmoid()

    # ---------------------------------------------------------
    #       Low-resolution rough lung segmentation (CPU)
    # ---------------------------------------------------------
    def forward_low(self, x):
        out = self.model_low(x).sigmoid()
        return out

    # ---------------------------------------------------------
    #                  Full forward
    # ---------------------------------------------------------
    def forward(self, x_high, x):
        out_low = self.forward_low(x_high)

        y_low_resize = torch.nn.functional.interpolate(
            out_low.detach(), size=x[0, 0].shape, mode='nearest'
        )
        out_high = self.forward_per_lobe(x, y_low_resize)

        return y_low_resize, out_high

    # ---------------------------------------------------------
    #               test_step for inference
    # ---------------------------------------------------------
    @torch.no_grad()
    def test_step(self, batch_dict):
        x_high, x = batch_dict["image_h"], batch_dict["image"]
        _, out_high = self.forward(x_high, x)
        return out_high.cpu()

    # ---------------------------------------------------------
    #              wrapper to generate lung mask
    # ---------------------------------------------------------
    def predict_lung(self, npz_path):
        sample = get_sample_image(npz_path)

        self.eval()
        with torch.no_grad():
            out = self.test_step(sample).cpu()

        if torch.is_tensor(out):
            out = out.squeeze().numpy()

        out = post_processing_lung(out, largest=2)
        return out

    def predict(self, sample, ID_image):
        self.eval()
        with torch.no_grad():
            out = self.test_step(sample).cpu()

        if torch.is_tensor(out):
            out = out.squeeze().numpy()

        out = post_processing_lung(out, largest=2)
        return out


# ---------------------------------------------------------
#              MAIN CPU-ONLY PIPELINE
# ---------------------------------------------------------
def main(args):

    print("Parameters:", args)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="inputs", type=str)
    parser.add_argument("--output", "-o", default="outputs", type=str)
    parser.add_argument("--delete", "-d", action="store_true")
    args = parser.parse_args()

    image_original_path = args.input
    output_path = args.output

    print(f"Input: {image_original_path}")
    print(f"Output: {output_path}")

    all_images = collect_images_verbose(image_original_path)
    if len(all_images) == 0:
        print("No images found.")
        return 0

    for image_original_path in all_images:
        path = Path(image_original_path)
        ext = "".join(path.suffixes)

        if ext in ['.mhd', '.mha']:
            image_original_path = convert_to_nifti(image_original_path)

        ID_image = (
            os.path.basename(image_original_path)
            .replace('.nii.gz', '')
            .replace('.nii', '')
            .replace('.mhd', '')
            .replace('.mha', '')
        )
        print(f"Image ID: {ID_image}")

        os.makedirs(TEMP_IMAGES, exist_ok=True)
        out_iso = os.path.join(
            TEMP_IMAGES, 'output_convert_cliped_isometric/images'
        )
        os.makedirs(out_iso, exist_ok=True)

        iso_path = os.path.join(out_iso, ID_image + ".nii.gz")

        # ------------------ Create isometric version -------------------
        if not os.path.exists(iso_path):
            image, label, lung, airway, spacing, shape = unified_img_reading(
                image_original_path,
                torch_convert=False,
                isometric=True,
                convert_to_onehot=6
            )

            transform = torchvision.transforms.Compose([
                CTHUClip(-1024, 600)
            ])
            image = transform((image, None))

            sitk.WriteImage(
                sitk.GetImageFromArray(image), iso_path
            )
        else:
            print("Isometric volume exists.")

        # ------------------ Build minimal npz -------------------
        save_path = os.path.join(TEMP_IMAGES, 'registered_images/npz_rigid')
        os.makedirs(save_path, exist_ok=True)
        out_npz = os.path.join(save_path, f"{ID_image}.npz")

        data = nib.load(iso_path).get_fdata()
        np.savez_compressed(out_npz, image=data)

        # ------------------ Predict Lung -------------------------
        ckpt = "weights/LightningLung.ckpt"
        model = LungModule.load_from_checkpoint(
            ckpt, strict=False, map_location=torch.device('cpu')
        )

        lung_mask = model.predict_lung(out_npz)

        salvaImageRebuilt(
            lung_mask.squeeze(),
            image_original_path,
            rigid_path=None,
            ID_image=ID_image,
            msg="lung",
            output_path=output_path
        )

    return 0


if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    sys.exit(main(sys.argv))

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import argparse
import torch
import torchvision
import numpy as np
import torchio as tio
import SimpleITK as sitk
import nibabel as nib
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from monai.inferers import sliding_window_inference
from pathlib import Path

from model.unet_diedre import UNet_SeisDecoders
from utils.general import pos_processamento, post_processing_dist_lung, post_processing_lung
from utils.general import unified_img_reading, busca_path, salvaImageRebuilt, convert_to_nifti, collect_images_verbose
from utils.to_onehot import mask_to_onehot
from utils.transform3D import CTHUClip
from predict_lung  import LungModule

HOME = os.getenv("HOME")
TEMP_IMAGES = 'temp_images'


# ------------------------------------------------------------------------------------
# CPU-SAFE get_sample()
# ------------------------------------------------------------------------------------
def get_sample(npz_path):
    ID_image = os.path.basename(npz_path).replace('.npz','').replace('_affine3D','').replace('_rigid3D','')

    npz = np.load(npz_path)
    img = npz["image"][:].astype(np.float32)

    img = img.transpose(2,1,0)

    if len(img.shape)==3:
        img = np.expand_dims(img, 0)

    subject = tio.Subject(image=tio.ScalarImage(tensor=img))
    transform = tio.Resize((128, 128, 128))
    transformed = transform(subject)
    img_high = transformed.image.numpy()

    img_high = torch.tensor(img_high, dtype=torch.float32).unsqueeze(0).cpu()
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).cpu()

    return {"image_h": img_high, "image": img, "ID_image": ID_image}


# ------------------------------------------------------------------------------------
# CPU-ONLY LoberModuleNormal
# ------------------------------------------------------------------------------------
class LoberModuleNormal(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        if self.hparams.mode == "segmentation":
            # Low-res model
            self.model_low = UNet_SeisDecoders(
                n_channels=1, n_classes=1, norm="instance",
                dim='3d', init_channel=16, joany_conv=False, dict_return=False
            ).cpu()

            # High-res model
            self.model = UNet_SeisDecoders(
                n_channels=7, n_classes=1, norm="instance",
                dim='3d', init_channel=16, joany_conv=False, dict_return=False
            ).cpu()


    # -------------------------------------------------------------------------
    # CPU-SAFE forward_per_lobe
    # -------------------------------------------------------------------------
    def forward_per_lobe(self, x, y_seg_resize):

        x_new = torch.cat((x.cpu(), y_seg_resize.cpu()), dim=1)

        output_one, output_two, output_three, output_four, output_five, output_lung = sliding_window_inference(
            x_new,
            roi_size=(128, 128, 128),
            sw_batch_size=1,
            predictor=self.model.cpu(),
            mode="gaussian",
            progress=False,
            device=torch.device('cpu')
        )

        output_one = output_one.sigmoid()
        output_two = output_two.sigmoid()
        output_three = output_three.sigmoid()
        output_four = output_four.sigmoid()
        output_five = output_five.sigmoid()
        output_lung = output_lung.sigmoid()

        buffer = [output_one, output_two, output_three, output_four, output_five]
        output_lobes = torch.cat(buffer, dim=1)

        lung = output_lobes.sum(dim=1).squeeze()
        bg_heatmap = 1 - torch.clip(lung, 0, 1)
        output_lobes = torch.cat([bg_heatmap.unsqueeze(0), output_lobes[0]], dim=0)
        output_lobes = output_lobes.unsqueeze(0)

        return output_lobes.cpu(), output_lung.cpu()


    # -------------------------------------------------------------------------
    # CPU-SAFE forward_low
    # -------------------------------------------------------------------------
    def forward_low(self, x):
        o1, o2, o3, o4, o5, lung_output = self.model_low(x.cpu())

        o1 = o1.sigmoid()
        o2 = o2.sigmoid()
        o3 = o3.sigmoid()
        o4 = o4.sigmoid()
        o5 = o5.sigmoid()
        lung_output = lung_output.sigmoid()

        buffer = [o1, o2, o3, o4, o5]
        output_low = torch.cat(buffer, dim=1)

        lung = output_low.sum(dim=1).squeeze()
        bg_heatmap = 1 - torch.clip(lung, 0, 1)
        output_low = torch.cat([bg_heatmap.unsqueeze(0), output_low[0]], dim=0)
        output_low = output_low.unsqueeze(0)

        return output_low.cpu(), lung_output.cpu()


    # -------------------------------------------------------------------------
    # CPU-SAFE forward() pipeline
    # -------------------------------------------------------------------------
    def forward(self, x_high, x):
        output_low, output_low_lung = self.forward_low(x_high.cpu())
        y_low_resize = torch.nn.functional.interpolate(output_low.detach(), size=x[0,0].shape, mode='nearest')
        output_lobes, output_lung = self.forward_per_lobe(x.cpu(), y_low_resize.cpu())
        return y_low_resize, output_lobes, output_low_lung, output_lung


    @torch.no_grad()
    def test_step(self, batch):
        x_high, x = batch["image_h"].cpu(), batch["image"].cpu()
        _, output_lobes, _, output_lung = self.forward(x_high, x)
        return output_lobes.cpu(), output_lung.cpu()


    # -------------------------------------------------------------------------
    # CPU-SAFE PREDICT
    # -------------------------------------------------------------------------
    def predict(self, npz_path, image_original_path, output_path,
                post_processed=True, save_image=False, rebuild=False):

        if rebuild:
            assert save_image == True

        sample = get_sample(npz_path)
        ID_image = os.path.basename(image_original_path).replace('.nii.gz','').replace('.nii','')

        # ---------------------------
        # LOAD Lung model (CPU only)
        # ---------------------------
        lung_ckpt = "weights/LightningLung.ckpt"
        test_model_lung = LungModule.load_from_checkpoint(
            lung_ckpt,
            strict=False,
            map_location=torch.device("cpu")
        )
        test_model_lung = test_model_lung.cpu()

        lung = test_model_lung.predict(sample, ID_image)
        salvaImageRebuilt(lung.squeeze(), image_original_path, None, ID_image, "lung", output_path)

        # ---------------------------
        # LOBE segmentation
        # ---------------------------
        self.eval()
        with torch.no_grad():
            output_lobes, output_lung = self.test_step(sample)

        output_lung = post_processing_lung(output_lung.squeeze().numpy())
        output_lung = torch.tensor(output_lung).float().unsqueeze(0).unsqueeze(0)

        image = output_lobes

        if post_processed:
            image = mask_to_onehot(image)
            image = np.expand_dims(image, 0)

            for ch in range(1, image.shape[1]):
                image[0, ch] = post_processing_lung(image[0, ch])

            image = torch.tensor(image)
            image = image.squeeze().argmax(dim=0).numpy().astype(np.int8)
            image = post_processing_dist_lung(image, lung)

            salvaImageRebuilt(image, image_original_path, None, ID_image, output_path)

        del image, lung


# ------------------------------------------------------------------------------------
# MAIN (CPU ONLY)
# ------------------------------------------------------------------------------------
def main(args):
    print("Parameters:", args)

    delete_data = False
    output_path = os.path.join(TEMP_IMAGES, 'outputs')

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="inputs")
    parser.add_argument("--output", "-o", default="outputs")
    parser.add_argument("--delete", "-d", action="store_true")
    args = parser.parse_args()

    image_original_path = args.input
    output_path = args.output
    delete_data = args.delete

    print(f"Input: {image_original_path}")
    print(f"Output: {output_path}")
    print(f"Prior Information: False (Normal mode)")
    print(f"Delete temporary files: {delete_data}")

    all_images = collect_images_verbose(image_original_path)
    if len(all_images) == 0:
        print("No images found.")
        return 0

    for image_original_path in all_images:
        path = Path(image_original_path)
        ext = "".join(path.suffixes)
        if ext in [".mhd", ".mha"]:
            image_original_path = convert_to_nifti(image_original_path)

        ID_image = os.path.basename(image_original_path).replace(".nii.gz","").replace(".nii","")
        print(f"Image ID: {ID_image}")

        # ------------------------------------------------------------------
        # Create isometric version
        # ------------------------------------------------------------------
        iso_dir = os.path.join(TEMP_IMAGES, "output_convert_cliped_isometric/images")
        os.makedirs(iso_dir, exist_ok=True)

        iso_path = os.path.join(iso_dir, ID_image + ".nii.gz")
        if not os.path.exists(iso_path):
            image, _, _, _, _, _ = unified_img_reading(
                image_original_path,
                torch_convert=False,
                isometric=True,
                convert_to_onehot=6
            )

            transform = torchvision.transforms.Compose([CTHUClip(-1024, 600)])
            image = transform((image, None))
            sitk.WriteImage(sitk.GetImageFromArray(image), iso_path)
        else:
            print("Isometric image already exists.")

        # Save NPZ (normal mode; no registration)
        npz_dir = os.path.join(TEMP_IMAGES, "npz_without_registration")
        os.makedirs(npz_dir, exist_ok=True)

        img_data = nib.load(iso_path).get_fdata()
        npz_path = os.path.join(npz_dir, f"{ID_image}.npz")
        np.savez_compressed(npz_path, image=img_data, ID=ID_image)

        # --------------------------------------------------------------
        # Load Lober model - CPU only
        # --------------------------------------------------------------
        ckpt = "weights/LightningLobes_no_template.ckpt"
        test_model = LoberModuleNormal.load_from_checkpoint(
            ckpt,
            strict=False,
            map_location=torch.device("cpu")
        )
        test_model = test_model.cpu()

        test_model.predict(npz_path, image_original_path, output_path,
                           post_processed=True, save_image=True, rebuild=True)

    return 0


if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    sys.exit(main(sys.argv))

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import shutil
import argparse
import torchvision
import numpy as np
import torchio as tio
import SimpleITK as sitk
import nibabel as nib
import pytorch_lightning as pl
import multiprocessing as mp
from monai.inferers import sliding_window_inference
from pathlib import Path
from tqdm import tqdm

from utils.general import (
    analyze_registration_quality,
    find_best_registration,
    post_processing_lung,
    register_single,
    teste_pickle_by_image,
    process_images,
    unified_img_reading,
    busca_path,
    salvaImageRebuilt,
    convert_to_nifti,
    remove_directories_if_exist,
    collect_images_verbose,
)

from model.unet_diedre import UNet_Diedre
from utils.transform3D import CTHUClip

HOME = os.getenv("HOME")
TEMP_IMAGES = "temp_images"
RAW_DATA_FOLDER = "raw_images"


# ============================================================
#          CPU-SAFE IMAGE PREPARATION
# ============================================================
def get_sample_image(npz_path):
    ID_image = (
        os.path.basename(npz_path)
        .replace(".npz", "")
        .replace("_affine3D", "")
        .replace("_rigid3D", "")
    )
    print(f"\tImage name: {ID_image}")

    npz = np.load(npz_path)
    img = npz["image"][:].astype(np.float32)

    img = img.transpose(2, 1, 0)
    if len(img.shape) == 3:
        img = np.expand_dims(img, 0)

    subject = tio.Subject(image=tio.ScalarImage(tensor=img))
    transform = tio.Resize((128, 128, 128))
    transformed = transform(subject)
    img_high = transformed.image.numpy()

    # CPU-only tensors
    img_high = torch.tensor(img_high, dtype=torch.float32).unsqueeze(0)
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    return {"image_h": img_high, "image": img}


# ============================================================
#                   CPU-ONLY Lung Module
# ============================================================
class LungModule(pl.LightningModule):
    def __init__(self, hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)

        # segmentation mode
        self.model_low = UNet_Diedre(
            n_channels=1,
            n_classes=1,
            norm="instance",
            dim="3d",
            init_channel=16,
            joany_conv=False,
            dict_return=False,
        )
        self.model = UNet_Diedre(
            n_channels=2,
            n_classes=1,
            norm="instance",
            dim="3d",
            init_channel=16,
            joany_conv=False,
            dict_return=False,
        )

    # ------------------- LOW RES -----------------------
    def forward_low(self, x):
        out = self.model_low(x).sigmoid()
        return out

    # ------------------- HIGH RES ----------------------
    def forward_per_lobe(self, x, y_seg_resize):
        x_new = torch.cat((x, y_seg_resize), dim=1)

        output_lung = sliding_window_inference(
            x_new,
            roi_size=(128, 128, 128),
            sw_batch_size=1,
            predictor=self.model,
            mode="gaussian",
            progress=False,
            device=torch.device("cpu"),
        )

        return output_lung.sigmoid()

    # ------------------- FULL FORWARD ------------------
    def forward(self, x_high, x):
        out_low = self.forward_low(x_high)

        y_low_resize = torch.nn.functional.interpolate(
            out_low.detach(), size=x[0, 0].shape, mode="nearest"
        )

        out_high = self.forward_per_lobe(x, y_low_resize)
        return y_low_resize, out_high

    # ------------------- TEST STEP ---------------------
    @torch.no_grad()
    def test_step(self, batch):
        x_high, x = batch["image_h"], batch["image"]
        _, out_high = self.forward(x_high, x)
        return out_high.cpu()

    # ------------------- WRAPPERS ----------------------
    def predict_lung(self, npz_path):
        sample = get_sample_image(npz_path)

        self.eval()
        with torch.no_grad():
            out = self.test_step(sample)

        if torch.is_tensor(out):
            out = out.squeeze().numpy()

        return post_processing_lung(out, largest=2)

    def predict(self, sample, ID_image):
        self.eval()
        with torch.no_grad():
            out = self.test_step(sample)

        if torch.is_tensor(out):
            out = out.squeeze().numpy()

        return post_processing_lung(out, largest=2)


# ============================================================
#                   MAIN SCRIPT (CPU-ONLY)
# ============================================================
def main(args):
    print("Parameters:", args)

    delete_data = False
    output_path = os.path.join(TEMP_IMAGES, "outputs")

    parser = argparse.ArgumentParser(
        description="Lung lobe segmentation on CT images using prior information."
    )
    parser.add_argument("--input", "-i", default="inputs", type=str)
    parser.add_argument("--output", "-o", default="outputs", type=str)
    parser.add_argument("--nworkers", "-nw", default=mp.cpu_count() // 2, type=int)
    parser.add_argument("--normal", "-n", action="store_true")
    parser.add_argument("--delete", "-d", action="store_true")
    parser.add_argument("--pool", "-p", action="store_true")
    args = parser.parse_args()

    image_original_path = args.input
    output_path = args.output
    parallel_processing = args.pool
    N_THREADS = args.nworkers

    print(f"Input: {image_original_path}")
    print(f"Output: {output_path}")
    print(f"Parallel processing: {parallel_processing}")

    all_images = collect_images_verbose(image_original_path)
    if len(all_images) == 0:
        print("No input images found.")
        return 0

    for image_original_path in all_images:
        path = Path(image_original_path)
        ext = "".join(path.suffixes)
        if ext in [".mhd", ".mha"]:
            image_original_path = convert_to_nifti(image_original_path)

        ID_image = (
            os.path.basename(image_original_path)
            .replace(".nii.gz", "")
            .replace(".nii", "")
            .replace(".mhd", "")
            .replace(".mha", "")
        )
        print(f"Image ID: {ID_image}")

        iso_dir = os.path.join(
            TEMP_IMAGES, "output_convert_cliped_isometric", "images"
        )
        os.makedirs(iso_dir, exist_ok=True)

        iso_path = os.path.join(iso_dir, ID_image + ".nii.gz")

        # ---------------- STEP 1: Create Isometric Image ----------------
        if not os.path.exists(iso_path):
            image, *_ = unified_img_reading(
                image_original_path,
                torch_convert=False,
                isometric=True,
                convert_to_onehot=6,
            )

            transform = torchvision.transforms.Compose([CTHUClip(-1024, 600)])
            image = transform((image, None))

            sitk.WriteImage(sitk.GetImageFromArray(image), iso_path)
        else:
            print("Isometric image exists")

        # ---------------- STEP 2: Registration Pipeline ------------------
        process_images(iso_path, ID_image, N_THREADS, parallel_processing)

        print("Registration completed successfully!")

        image = nib.load(iso_path).get_fdata()
        registered_folder = os.path.join(RAW_DATA_FOLDER, "images_npz")

        # Select best registration
        results = analyze_registration_quality(image, ID_image, registered_folder)
        best_image, _ = find_best_registration(results)

        # Read best registered NPZ
        group = np.load(best_image)["group"]
        image_path = os.path.join(
            TEMP_IMAGES,
            "registered_images/groups",
            f"group_{group}",
            "npz_rigid",
            ID_image + ".npz",
        )

        # ---------------- STEP 3: LUNG PREDICTION (CPU) ------------------
        ckpt = "weights/LightningLung.ckpt"
        model = LungModule.load_from_checkpoint(
            ckpt, strict=False, map_location=torch.device("cpu")
        )

        lung_mask = model.predict_lung(image_path)

        rigid_path = busca_path(ID_image, group)
        salvaImageRebuilt(
            lung_mask.squeeze(),
            image_original_path,
            rigid_path=rigid_path,
            ID_image=ID_image,
            msg="lung",
            output_path=output_path,
        )

    if args.delete:
        dirs = [os.path.join(TEMP_IMAGES, "output_convert_cliped_isometric")]
        remove_directories_if_exist(dirs)

    return 0


if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    sys.exit(main(sys.argv))

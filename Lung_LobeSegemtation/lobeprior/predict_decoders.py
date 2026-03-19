#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import torch
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

from model.unet_diedre import UNet_SeteDecoders
from predict_lung import LungModule
from utils.general import (
    pos_processamento,
    post_processing_dist_lung,
    post_processing_lung,
)
from utils.general import (
    register_single,
    teste_pickle_by_image,
    process_images,
)
from utils.general import (
    unified_img_reading,
    busca_path,
    salvaImageRebuilt,
    convert_to_nifti,
    remove_directories_if_exist,
    collect_images_verbose,
)
from utils.general import analyze_registration_quality, find_best_registration
from utils.to_onehot import mask_to_onehot
from utils.transform3D import CTHUClip

HOME = os.getenv("HOME")
TEMP_IMAGES = "temp_images"
RAW_DATA_FOLDER = "raw_images"  # os.path.join(HOME, 'raw_images')


def get_sample_image(npz_path):
    ID_image = os.path.basename(npz_path).replace(".npz", "")
    print(f"\tImage name: {ID_image}")

    npz = np.load(npz_path)
    img = npz["image"][:].astype(np.float32)

    group = npz["group"]

    npz_template_path = os.path.join(
        RAW_DATA_FOLDER, f"model_fusion/group_{group}.npz"
    )
    template = np.load(npz_template_path)["model"][:].astype(np.float32)

    # (z, y, x) -> (c, x, y, z)
    img = img.transpose(2, 1, 0)
    if len(img.shape) == 3:
        img = np.expand_dims(img, 0)

    subject = tio.Subject(image=tio.ScalarImage(tensor=img))
    transform = tio.Resize((128, 128, 128))
    transformed = transform(subject)
    img_high = transformed.image.numpy()

    
    img_high = torch.tensor(img_high, dtype=torch.float32).unsqueeze(dim=0)
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(dim=0)
    template = torch.tensor(template, dtype=torch.float32).unsqueeze(dim=0)

    return {
        "image_h": img_high,
        "image": img,
        "template": template,
        "npz_path": npz_path,
        "ID_image": ID_image,
    }


class LoberModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # PL expects attribute named hparams
        self.save_hyperparameters(hparams)

        if self.hparams.mode == "segmentation":
            # Low: full FOV, low resolution
            self.model_low = UNet_SeteDecoders(
                n_channels=1,
                n_classes=1,
                norm="instance",
                dim="3d",
                init_channel=16,
                joany_conv=False,
                dict_return=False,
            )
            # High-res patch
            self.model = UNet_SeteDecoders(
                n_channels=14,
                n_classes=1,
                norm="instance",
                dim="3d",
                init_channel=16,
                joany_conv=False,
                dict_return=False,
            )

    def forward_per_lobe(self, x, template, y_seg_resize):
        """
        x: (B, 1, D, H, W)
        template: (B, T, D, H, W)  (atlas / prior)
        y_seg_resize: (B, 7, D, H, W) or similar
        """
        x_new = torch.cat((x, y_seg_resize, template), dim=1)

        # CPU-only sliding window inference
        output_lung, output_one, output_two, output_three, output_four, output_five, output_airway = (
            sliding_window_inference(
                x_new,
                roi_size=(96, 96, 96),
                sw_batch_size=1,
                predictor=self.model,  # model stays on CPU
                mode="gaussian",
                progress=False,
                device=torch.device("cpu"),
            )
        )

        # Apply sigmoids
        output_lung = output_lung.sigmoid()
        output_lung = 1 - output_lung  # invert foreground/background

        output_one = output_one.sigmoid()
        output_two = output_two.sigmoid()
        output_three = output_three.sigmoid()
        output_four = output_four.sigmoid()
        output_five = output_five.sigmoid()
        output_airway = output_airway.sigmoid()

        buffer = [
            output_lung,
            output_one,
            output_two,
            output_three,
            output_four,
            output_five,
        ]
        output_lobes = torch.cat(buffer, dim=1)

        return output_lung, output_lobes, output_airway

    def forward_low(self, x):
        """
        Low-resolution multi-branch decoder.
        """
        (
            output_lung,
            output_one,
            output_two,
            output_three,
            output_four,
            output_five,
            airway,
        ) = self.model_low(x)

        output_lung = output_lung.sigmoid()
        output_lung = 1 - output_lung

        output_one = output_one.sigmoid()
        output_two = output_two.sigmoid()
        output_three = output_three.sigmoid()
        output_four = output_four.sigmoid()
        output_five = output_five.sigmoid()
        airway = airway.sigmoid()

        buffer = [
            output_lung,
            output_one,
            output_two,
            output_three,
            output_four,
            output_five,
            airway,
        ]
        output_low = torch.cat(buffer, dim=1)

        return output_low

    def forward(self, x_high, x, template):
        """
        x_high: low-res full volume
        x: high-res (or original-res) volume
        template: prior atlas
        """
        output_low = self.forward_low(x_high)

        # Resize low-res segmentation to x's spatial size
        y_low_resize = torch.nn.functional.interpolate(
            output_low.detach(), size=x[0, 0].shape, mode="nearest"
        )

        output_lung, output_lobes, output_airway = self.forward_per_lobe(
            x, template, y_low_resize
        )

        return output_lung, output_lobes, output_airway

    @torch.no_grad()
    def test_step(self, test_batch):
        x_high, x, template = (
            test_batch["image_h"],
            test_batch["image"],
            test_batch["template"],
        )

        output_lung, output_lobes, output_airway = self.forward(
            x_high, x, template
        )

        return (
            output_lung.cpu(),
            output_lobes.cpu(),
            output_airway.cpu(),
        )

    def predict(
        self,
        npz_path,
        image_original_path,
        output_path,
        group=None,
        post_processed=True,
    ) -> np.ndarray:

        ID_image = (
            os.path.basename(image_original_path)
            .replace(".npz", "")
            .replace(".nii.gz", "")
            .replace(".nii", "")
            .replace(".mhd", "")
            .replace(".mha", "")
        )

        sample = get_sample_image(npz_path)

        rigid_path = busca_path(ID_image, group)

        # ---------------- LungModule (CPU) ----------------
        pre_trained_model_lung_path = "weights/LightningLung.ckpt"
        test_model_lung = LungModule.load_from_checkpoint(
            pre_trained_model_lung_path,
            strict=False,
            map_location=torch.device("cpu"),
        )

        lung = test_model_lung.predict(sample, ID_image)

        salvaImageRebuilt(
            lung.squeeze(),
            image_original_path,
            rigid_path=rigid_path,
            ID_image=ID_image,
            msg="lung",
            output_path=output_path,
        )

        template = sample["template"]

        self.eval()
        with torch.no_grad():
            _, image, airway = self.test_step(sample)

        airway = post_processing_lung(airway.squeeze().numpy())
        airway = torch.from_numpy(airway).float()
        airway = airway.unsqueeze(dim=0).unsqueeze(dim=0)

        if post_processed:
            lung_t = torch.from_numpy(lung).float()
            image = pos_processamento(
                output=image.cpu(),
                template=template.cpu(),
                segmentation=lung_t.unsqueeze(dim=0).unsqueeze(dim=0),
            )
            lung = lung_t.numpy().astype(np.uint8)

            image = image.squeeze().numpy()

            # one-hot -> postprocess each lobe channel
            image = mask_to_onehot(image)  # (C, D, H, W)
            image = torch.from_numpy(image).unsqueeze(dim=0).float()  # (1, C, D, H, W)

            image_np = mask_to_onehot(image)  # NOTE: original code calls twice
            image_np = np.expand_dims(image_np, 0)

            for channel in range(1, image_np.shape[1]):
                image_np[0, channel] = post_processing_lung(image_np[0, channel])

            image = torch.from_numpy(image_np)
            image = image.squeeze().argmax(dim=0).numpy().astype(np.int8)

            image = post_processing_dist_lung(image, lung)

        assert (
            image.min() == 0 and image.max() == 5
        ), f"MinMax incorretos {image.shape}: {image.min()} e {image.max()}"

        salvaImageRebuilt(
            image.squeeze(),
            image_original_path,
            rigid_path=rigid_path,
            ID_image=ID_image,
            output_path=output_path,
        )

        del lung
        del template
        del image


def main(args):
    print("Parameters:", args)

    delete_data = False
    output_path = os.path.join(TEMP_IMAGES, "outputs")

    parser = argparse.ArgumentParser(
        description="Lung lobe segmentation on CT images using prior information."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="inputs",
        help="Input image or folder with volumetric images.",
        type=str,
    )
    parser.add_argument(
        "--output",
        "-o",
        default="outputs",
        help="Directory to store the final segmentation.",
        type=str,
    )
    parser.add_argument(
        "--nworkers",
        "-nw",
        default=mp.cpu_count() // 2,
        help="Number of workers",
        type=int,
    )
    parser.add_argument(
        "--normal",
        "-n",
        action="store_true",
        help="Use Prior Information.",
    )  # true se passou --normal
    parser.add_argument(
        "--delete",
        "-d",
        action="store_true",
        help="Delete temporary files.",
    )  # true se passou --delete
    parser.add_argument(
        "--pool",
        "-p",
        action="store_true",
        help="Parallel processing.",
    )  # true se passou --pool

    args = parser.parse_args()

    image_original_path = args.input
    output_path = args.output
    modo_normal = args.normal
    delete_data = args.delete
    parallel_processing = args.pool
    N_THREADS = args.nworkers

    print(f"Input: {image_original_path}")
    print(f"Output: {output_path}")
    print(f"Prior Information: {not modo_normal}")
    print(f"Delete temporary files : {delete_data}")
    print(f"Parallel processing: {parallel_processing}")
    if parallel_processing:
        print(f"Number of processes: {N_THREADS}")

    all_images = collect_images_verbose(image_original_path)

    if len(all_images) == 0:
        print("Either the image path is incorrect or the input image is missing.")
        print("python predict_decoders.py -i <input.nii.gz>")
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

        if not os.path.exists(
            os.path.join(
                TEMP_IMAGES,
                "output_convert_cliped_isometric/images",
                ID_image + ".nii.gz",
            )
        ):
            os.makedirs(TEMP_IMAGES, exist_ok=True)
            os.makedirs(
                os.path.join(
                    TEMP_IMAGES, "output_convert_cliped_isometric/images"
                ),
                exist_ok=True,
            )

            (
                image,
                label,
                lung,
                airway,
                spacing,
                shape,
            ) = unified_img_reading(
                image_original_path,
                torch_convert=False,
                isometric=True,
                convert_to_onehot=6,
            )

            transform = torchvision.transforms.Compose([CTHUClip(-1024, 600)])
            image = transform((image, None))

            output_image = sitk.GetImageFromArray(image)

            sitk.WriteImage(
                output_image,
                os.path.join(
                    TEMP_IMAGES,
                    "output_convert_cliped_isometric/images",
                    ID_image + ".nii.gz",
                ),
            )
        else:
            print("Isomeric images successfully created!")

        image_path = os.path.join(
            TEMP_IMAGES, "output_convert_cliped_isometric/images", ID_image + ".nii.gz"
        )

        process_images(
            image_path, ID_image, N_THREADS, parallel_processing=parallel_processing
        )

        image = nib.load(image_path).get_fdata()

        # Analyze all registered images
        registered_folder = os.path.join(RAW_DATA_FOLDER, "images_npz")
        results = analyze_registration_quality(image, ID_image, registered_folder)

        best_image, best_score = find_best_registration(results)
        print("Registration completed successfully!")

        ID_template = os.path.basename(best_image).replace(".npz", "")

        template_path = os.path.join(registered_folder, ID_template + ".npz")
        template_array = np.load(template_path)["image"][:].astype(np.float32)
        template_array = template_array.transpose(2, 1, 0)
        group = np.load(template_path)["group"]

        image_path = os.path.join(
            TEMP_IMAGES,
            f"registered_images/groups/group_{group}",
            "npz_rigid",
            ID_image + ".npz",
        )
        image_array = np.load(image_path)["image"][:].astype(np.float32)
        image_array = image_array.transpose(2, 1, 0)

        pre_trained_model_path = "weights/LightningLobes.ckpt"

        # IMPORTANT: load model on CPU
        test_model = LoberModule.load_from_checkpoint(
            pre_trained_model_path,
            strict=False,
            map_location=torch.device("cpu"),
        )

        test_model.predict(
            image_path,
            image_original_path,
            output_path,
            group=group,
            post_processed=True,
        )

    if delete_data:
        dirs = [
            os.path.join(TEMP_IMAGES, "output_convert_cliped_isometric")
        ]
        remove_directories_if_exist(dirs)

    return 0


if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    sys.exit(main(sys.argv))

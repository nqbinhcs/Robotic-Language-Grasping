import glob
import os
import re

import pickle
import torch

from utils.dataset_processing import grasp, image, mask
from .grasp_data import GraspDatasetBase
from inference.models.clip_embedder import CLIPTextEmbedder
import os
from tqdm import tqdm


class GraspAnythingDataset(GraspDatasetBase):
    """
    Dataset wrapper for the Grasp-Anything dataset.
    """

    def __init__(self, file_path, ds_rotate=0, **kwargs):
        """
        :param file_path: Grasp-Anything Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(GraspAnythingDataset, self).__init__(**kwargs)

        self.grasp_files = glob.glob(os.path.join(file_path, "grasp_label", "*.pt"))
        self.prompt_files = glob.glob(
            os.path.join(file_path, "grasp_instructions", "*.pkl")
        )
        self.rgb_files = glob.glob(os.path.join(file_path, "image", "*.jpg"))
        # self.mask_files = glob.glob(os.path.join(file_path, 'mask', '*.npy'))

        # if kwargs["seen"]:
        #     with open(os.path.join('split/grasp-anything/seen.obj'), 'rb') as f:
        #         idxs = pickle.load(f)

        #     self.grasp_files = list(filter(lambda x: x.split('/')[-1].split('.')[0] in idxs, self.grasp_files))
        # else:
        #     with open(os.path.join('split/grasp-anything/unseen.obj'), 'rb') as f:
        #         idxs = pickle.load(f)

        #     self.grasp_files = list(filter(lambda x: x.split('/')[-1].split('.')[0] in idxs, self.grasp_files))

        self.grasp_files.sort()
        self.prompt_files.sort()
        self.rgb_files.sort()

        # self.mask_files.sort()

        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError(
                "No dataset files found. Check path: {}".format(file_path)
            )

        if ds_rotate:
            self.grasp_files = (
                self.grasp_files[int(self.length * ds_rotate) :]
                + self.grasp_files[: int(self.length * ds_rotate)]
            )

        self.text_embeddings = self.prepare_text_embedding()
        print(f"{len(self.text_embeddings)} text embeddings are prepared")
        # self.text_embedder = CLIPTextEmbedder(device="cpu")

    def prepare_text_embedding(self, batch_size=1):
        cache_file = "text_embeddings_cache.pkl"

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                text_embeddings = pickle.load(f)
        else:
            text_embedder = CLIPTextEmbedder(device='cuda')
            text_embeddings = []
            num_prompts = len(self.prompt_files)
            for i in tqdm(range(0, num_prompts, batch_size)):
                batch_prompts = []
                for j in range(i, min(i + batch_size, num_prompts)):
                    with open(self.prompt_files[j], "rb") as f:
                        prompt = pickle.load(f)
                    batch_prompts.append(prompt)

                ext_embeddings = text_embedder(batch_prompts).to('cpu')
                text_embeddings.append(ext_embeddings)

            # del text_embedder

            # with open(cache_file, "wb") as f:
            #     pickle.dump(text_embeddings, f)

        return text_embeddings

    # def get_text_embedding(self, idx):
    #     with open(self.prompt_files[idx], "rb") as f:
    #         prompt = pickle.load(f)
    #     return self.text_embedder([prompt])

    def get_text_embedding(self, idx):
        return self.text_embeddings[idx]

    def get_prompt(self, idx):
        with open(self.prompt_files[idx], "rb") as f:
            prompt = pickle.load(f)
        return prompt

    def _get_crop_attrs(self, idx):
        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(
            self.grasp_files[idx]
        )
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 416 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 416 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        # Jacquard try
        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(
            self.grasp_files[idx], scale=self.output_size / 416.0
        )

        # print("gtbbs", gtbbs)

        c = self.output_size // 2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))

        # Cornell try
        # gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx])
        # center, left, top = self._get_crop_attrs(idx)
        # gtbbs.rotate(rot, center)
        # gtbbs.offset((-top, -left))
        # gtbbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop(
            (top, left),
            (min(480, top + self.output_size), min(640, left + self.output_size)),
        )
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        # mask_file = self.grasp_files[idx].replace("positive_grasp", "mask").replace(".pt", ".npy")
        # mask_img = mask.Mask.from_file(mask_file)
        rgb_file = re.sub(r"_\d{1}_\d{1}\.pt", ".jpg", self.grasp_files[idx])
        rgb_file = rgb_file.replace("grasp_label", "image")
        rgb_img = image.Image.from_file(rgb_file)
        # rgb_img = image.Image.mask_out_image(rgb_img, mask_img)

        # Jacquard try
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

        # Cornell try
        # center, left, top = self._get_crop_attrs(idx)
        # rgb_img.rotate(rot, center)
        # rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        # rgb_img.zoom(zoom)
        # rgb_img.resize((self.output_size, self.output_size))
        # if normalise:
        #     rgb_img.normalise()
        #     rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        # return rgb_img.img

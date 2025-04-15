import os
import torch
import pickle
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, root, image_numbers, transform=None):
        self.root = root
        self.image_numbers = image_numbers
        self.transform = transform

    def __len__(self):
        return len(self.image_numbers)

    def __getitem__(self, idx):
        ir_path = os.path.join(self.root, "infrared/train/", f"{idx + 1}.jpg")
        vis_path = os.path.join(self.root, "visible/train/", f"{idx + 1}.jpg")
        visNF_path = os.path.join(self.root, "visible_focus_near/train/", f"{idx + 1}.jpg")
        visFF_path = os.path.join(self.root, "visible_focus_far/train/", f"{idx + 1}.jpg")

        ir_img = Image.open(ir_path).convert("L")
        vis_img = Image.open(vis_path).convert("L")
        visNF_img = Image.open(visNF_path).convert("L")
        visFF_img = Image.open(visFF_path).convert("L")

        if self.transform:
            ir_img = self.transform(ir_img)
            vis_img = self.transform(vis_img)
            visNF_img = self.transform(visNF_img)
            visFF_img = self.transform(visFF_img)

        return ir_img, vis_img, visNF_img, visFF_img


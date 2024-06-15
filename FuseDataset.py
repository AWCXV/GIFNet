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

class TestDataset(Dataset):
    def __init__(self, root, image_numbers, transform=None):
        self.root = root
        self.image_numbers = image_numbers
        self.transform = transform

    def __len__(self):
        return len(self.image_numbers)

    def __getitem__(self, idx):
        ir_path = os.path.join(self.root, "ir", f"{idx + 1}.png")
        vis_path = os.path.join(self.root, "vis_grayscale", f"{idx + 1}.png")

        ir_img = Image.open(ir_path).convert("L")
        vis_img = Image.open(vis_path).convert("L")

        if self.transform:
            ir_img = self.transform(ir_img)
            vis_img = self.transform(vis_img)

        return ir_img, vis_img

class TestDataset_rgb(Dataset):
    def __init__(self, root, image_numbers, transform=None):
        self.root = root
        self.image_numbers = image_numbers
        self.transform = transform

    def __len__(self):
        return len(self.image_numbers)

    def __getitem__(self, idx):
        ir_path = os.path.join(self.root, "ir", f"{idx + 1}.png")
        vis_path = os.path.join(self.root, "vis", f"{idx + 1}.png")

        ir_img = Image.open(ir_path).convert("L")
        vis_img = Image.open(vis_path).convert("RGB")

        if self.transform:
            ir_img = self.transform(ir_img)
            vis_img = self.transform(vis_img)

        return ir_img, vis_img

class MFI_WHU_TestDataset(Dataset):
    def __init__(self, root, image_numbers, transform=None):
        self.root = root
        self.image_numbers = image_numbers
        self.transform = transform

    def __len__(self):
        return len(self.image_numbers)

    def __getitem__(self, idx):
        ir_path = os.path.join(self.root, "ir_y", f"{idx + 1}.jpg")
        vis_path = os.path.join(self.root, "vis_y", f"{idx + 1}.jpg")

        ir_img = Image.open(ir_path).convert("L")
        vis_img = Image.open(vis_path).convert("L")

        if self.transform:
            ir_img = self.transform(ir_img)
            vis_img = self.transform(vis_img)

        return ir_img, vis_img
        
class DSCIE_TestDataset(Dataset):
    def __init__(self, root, image_numbers, transform=None):
        self.root = root
        self.image_numbers = image_numbers
        self.transform = transform

    def __len__(self):
        return len(self.image_numbers)

    def __getitem__(self, idx):
        ir_path = os.path.join(self.root, "vis_d4_y", f"{idx + 1}.jpg")
        vis_path = os.path.join(self.root, "ir_d4_y", f"{idx + 1}.jpg")

        ir_img = Image.open(ir_path).convert("L")
        vis_img = Image.open(vis_path).convert("L")

        if self.transform:
            ir_img = self.transform(ir_img)
            vis_img = self.transform(vis_img)

        return ir_img, vis_img        
        
class NIR_VIS_TestDataset(Dataset):
    def __init__(self, root, image_numbers, transform=None):
        self.root = root
        self.image_numbers = image_numbers
        self.transform = transform

    def __len__(self):
        return len(self.image_numbers)

    def __getitem__(self, idx):
        nir_path = os.path.join(self.root, "nir/test/", f"{idx + 1}.png")
        vis_path = os.path.join(self.root, "vis_y/test/", f"{idx + 1}.png")

        nir_img = Image.open(nir_path).convert("L")
        vis_img = Image.open(vis_path).convert("L")

        if self.transform:
            nir_img = self.transform(nir_img)
            vis_img = self.transform(vis_img)

        return nir_img, vis_img     
        
class Harvard_TestDataset(Dataset):
    def __init__(self, root, image_numbers, transform=None):
        self.root = root
        self.image_numbers = image_numbers
        self.transform = transform

    def __len__(self):
        return len(self.image_numbers)

    def __getitem__(self, idx):
        nir_path = os.path.join(self.root, "ir_y/", f"{idx + 1}.png")
        vis_path = os.path.join(self.root, "vis/", f"{idx + 1}.png")

        nir_img = Image.open(nir_path).convert("L")
        vis_img = Image.open(vis_path).convert("L")

        if self.transform:
            nir_img = self.transform(nir_img)
            vis_img = self.transform(vis_img)

        return nir_img, vis_img            
        
class Quickbird_TestDataset(Dataset):
    def __init__(self, root, image_numbers, transform=None):
        self.root = root
        self.image_numbers = image_numbers
        self.transform = transform

    def __len__(self):
        return len(self.image_numbers)

    def __getitem__(self, idx):
        band1_path = os.path.join(self.root, "1/", f"{idx + 1}.tif")
        band2_path = os.path.join(self.root, "2/", f"{idx + 1}.tif")
        band3_path = os.path.join(self.root, "3/", f"{idx + 1}.tif")
        band4_path = os.path.join(self.root, "4/", f"{idx + 1}.tif")
        vis_path = os.path.join(self.root, "vis/", f"{idx + 1}.tif")

        band1_img = Image.open(band1_path).convert("L")
        band2_img = Image.open(band2_path).convert("L")
        band3_img = Image.open(band3_path).convert("L")
        band4_img = Image.open(band4_path).convert("L")
        vis_img = Image.open(vis_path).convert("L")

        if self.transform:
            band1_img = self.transform(band1_img)
            band2_img = self.transform(band2_img)
            band3_img = self.transform(band3_img)
            band4_img = self.transform(band4_img)
            vis_img = self.transform(vis_img)

        return band1_img, band2_img, band3_img, band4_img, vis_img                    
        

class COCOval2017_TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.file_names = os.listdir(root);
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        vis_path = os.path.join(self.root, self.file_names[idx])

        vis_img = Image.open(vis_path).convert("RGB")

        if self.transform:
            vis_img = self.transform(vis_img)

        return self.file_names[idx], vis_img    
 
class COCOtrain2017_TrainDataset(Dataset):
    def __init__(self, root, image_numbers, transform=None):
        self.root = root
        self.file_names = os.listdir(root);
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        vis_path = os.path.join(self.root, self.file_names[idx])

        vis_img = Image.open(vis_path).convert("RGB")

        if self.transform:
            vis_img = self.transform(vis_img)

        return self.file_names[idx], vis_img    
        
class LLVIP2000_TrainDataset(Dataset):
    def __init__(self, root, image_numbers, transform=None):
        self.root = root
        self.image_numbers = image_numbers
        self.transform = transform

    def __len__(self):
        return len(self.image_numbers)

    def __getitem__(self, idx):
        ir_path = os.path.join(self.root, "ir", f"{idx + 1}.png")
        vis_path = os.path.join(self.root, "vis", f"{idx + 1}.png")

        ir_img = Image.open(ir_path).convert("L")
        vis_img = Image.open(vis_path).convert("L")

        if self.transform:
            ir_img = self.transform(ir_img)
            vis_img = self.transform(vis_img)

        return ir_img, vis_img
        

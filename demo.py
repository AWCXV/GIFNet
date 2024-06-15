import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import time
import numpy as np
from torch.autograd import Variable
from GIFNet_model import TwoBranchesFusionNet
from args import Args as args
import utils
import matplotlib.pyplot as plt  
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

def resize_images(images, target_size=(128, 128)):

    return F.interpolate(images, size=target_size, mode='bilinear', align_corners=False)

def load_model(model_path_twoBranches):
    model = TwoBranchesFusionNet(args.s, args.n, args.channel, args.stride)

    model.load_state_dict(torch.load(model_path_twoBranches))

    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))
    
    total = sum([param.nelement() for param in model.parameters()])
    print('Number    of    parameter: {:4f}M'.format(total / 1e6))
    
    model.eval()
    if (args.cuda):
        model.cuda()

    return model

def run(model, ir_test_batch, vis_test_batch, output_path, img_name):

    img_ir = ir_test_batch
    img_vi = vis_test_batch

    img_ir = Variable(img_ir, requires_grad=False)
    img_vi = Variable(img_vi, requires_grad=False)

    fea_com = model.forward_encoder(img_ir, img_vi)    
    fea_fused = model.forward_MultiTask_branch(fea_com_ivif = fea_com, fea_com_mfif = fea_com)            
    out_f = model.forward_mixed_decoder(fea_com, fea_fused);    

    path_out = output_path + "/" + img_name + '.jpg'
    utils.save_image(out_f, path_out)    
    
    print('Image->'+ img_name + ' Done......')    


def main():

    test_path = "./images/"
    imgs_paths_ir, names = utils.list_images(test_path)
    num = len(imgs_paths_ir)

    model_path_twoBranches = 'model/Final.model'

    output_path_root = 'outputs/'  

    if os.path.exists(output_path_root) is False:
        os.mkdir(output_path_root)

    output_path = output_path_root;
    if os.path.exists(output_path) is False:
        os.mkdir(output_path)
        
    #task_type: seen, unseen
    task_type = "seen";

    with torch.no_grad():
        model = load_model(model_path_twoBranches)
        
        transform = transforms.Compose([
            transforms.ToTensor()  
        ])        
                
        test_root_dir = "./images/" + task_type + "/"

        ir_path = os.path.join(test_root_dir, "ir.jpg")
        vis_path = os.path.join(test_root_dir, "vis.jpg")

        ir_img = Image.open(ir_path).convert("L")
        vis_img = Image.open(vis_path).convert("L")

        ir_img = transform(ir_img)
        vis_img = transform(vis_img)        
        
        if (args.cuda):
            ir_img = ir_img.cuda();
            vis_img = vis_img.cuda();
        
        ir_test_batch = ir_img.unsqueeze(0);
        vis_test_batch = vis_img.unsqueeze(0);

        img_name = task_type + "_fused";
        run(model, ir_test_batch, vis_test_batch, output_path, img_name)
            
if __name__ == '__main__':
    main()

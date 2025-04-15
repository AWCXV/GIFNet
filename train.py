

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import scipy.io as scio
import torch
from torch.optim import Adam
from torch.autograd import Variable
from utils import gradient
from testMat import showLossChart
from torch import nn

# from net import LRR_NET, Vgg16
# from net_nuclear import LRR_NET, Vgg16
from GIFNet_model import TwoBranchesFusionNet
from args import Args as args
import utils
import random
import torchvision.models as models
import pytorch_msssim
from GIFNetDataset import CustomDataset
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--trainDataRoot', type=str, default='./train_data_path', help='Root path of the train data')

opt = parser.parse_args()

EPSILON = 1e-5


def load_data(path, train_num):
    imgs_path, _ = utils.list_images(path)
    imgs_path = imgs_path[:train_num]
    random.shuffle(imgs_path)
    return imgs_path

       
def lossChartSave(temp_loss,lossName,lossList):        
    # save item1_spe loss
    loss_filename_path = lossName + temp_loss
    save_loss_path = os.path.join(os.path.join(args.save_loss_dir), loss_filename_path)
    scio.savemat(save_loss_path, {'Loss': lossList})
    showLossChart(save_loss_path,os.path.join(args.save_loss_dir)+"/"+lossName+'.png')        


def main():

    densenet = models.densenet121(pretrained=True)    
    densenet.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    densenet = densenet.to(device)
    features = list(densenet.features.children())        
    
    features_1 = nn.Sequential(*features[:4]);
    features_2 = nn.Sequential(*features[4:6]);
    features_3 = nn.Sequential(*features[6:8]);
    features_4 = nn.Sequential(*features[8:10]);
    features_5 = nn.Sequential(*features[10:11]);

    twoBranchesFusionModel = TwoBranchesFusionNet(args.s, args.n, args.channel, args.stride)

    optimizer_TB = Adam(twoBranchesFusionModel.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()
    ssim_loss = pytorch_msssim.msssim

    if args.cuda:
        twoBranchesFusionModel.cuda()
    

    #------------ Begin Load Data
    root_dir = opt.trainDataRoot
    image_numbers = list(range(1, args.train_num))

    transform = transforms.Compose([
        transforms.ToTensor()  # 转为Tensor
    ])


    #CombinationAutoEncoder, TwoBranchesFusionNet

    custom_dataset = CustomDataset(root_dir, image_numbers, transform=transform)
    batch_size = args.batch_size
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
    #------------ End Load Data
    
    # tbar = trange(args.epochs)
    print('Start training.....')

    step = 10

    # creating save path
    temp_path_model = os.path.join(args.save_fusion_model)
    if os.path.exists(temp_path_model) is False:
        os.mkdir(temp_path_model)
        
    temp_path_loss = os.path.join(args.save_loss_dir)
    if os.path.exists(temp_path_loss) is False:
        os.mkdir(temp_path_loss)

    Loss_list_item1_spe = []
    Loss_list_item1_com = []
    Loss_list_item1_learning = []
    Loss_list_item2_spe = []
    Loss_list_item2_com = []

    Loss_list_all = []

    viz_index = 0
    loss_item1_spe = 0.
    loss_item1_com = 0.
    loss_item1_learning = 0.
    loss_item2_spe = 0.
    loss_item2_com = 0.
    twoBranchesFusionModel.train()
    for e in range(args.epochs):

        batch_num = len(data_loader)
        loss_item1_spe = 0.
        loss_item2_spe = 0.
        loss_item1_com = 0.
        loss_item2_com = 0.        
        for idx, batch in enumerate(data_loader):
            img_ir, img_vi, img_ir_NF, img_vi_FF = batch

            batch_ir = Variable(img_ir, requires_grad=False)
            batch_vi = Variable(img_vi, requires_grad=False)
            img_ir_NF = Variable(img_ir_NF, requires_grad=False)
            img_vi_FF = Variable(img_vi_FF, requires_grad=False)

            if args.cuda:
                batch_ir = batch_ir.cuda()
                batch_vi = batch_vi.cuda()
                img_ir_NF = img_ir_NF.cuda()
                img_vi_FF = img_vi_FF.cuda()                            

            IVIF_step = 1;
            MFIF_step = 1;
                
            #IVIF branch
            for _idx in range(IVIF_step):
                optimizer_TB.zero_grad()                                

                fea_com = twoBranchesFusionModel.forward_encoder(batch_ir,batch_vi)
                with torch.no_grad():                
                    fea_com_mfif = twoBranchesFusionModel.forward_encoder(img_ir_NF,img_vi_FF)
                    
                out_rec = twoBranchesFusionModel.forward_rec_decoder(fea_com)
                
                fea_fused = twoBranchesFusionModel.forward_MultiTask_branch(fea_com_ivif = fea_com, fea_com_mfif = fea_com_mfif, trainingTag = 1);
                
                out_f = twoBranchesFusionModel.forward_mixed_decoder(fea_com, fea_fused);                                                
                
                #计算源图像的信息量的度量。
                with torch.no_grad():
                    t_batch_ir = batch_ir.clone();
                    t_batch_vi = batch_vi.clone();
                    dup_ir = torch.cat([t_batch_ir,t_batch_ir,t_batch_ir],1);
                    dup_vi = torch.cat([t_batch_vi,t_batch_vi,t_batch_vi],1);                    
                    
                    # -----------------获取CNN特征权重 开始------------------
                    layer1_feature_ir = features_1(dup_ir)
                    layer2_feature_ir = features_2(layer1_feature_ir)
                    layer3_feature_ir = features_3(layer2_feature_ir)
                    layer4_feature_ir = features_4(layer3_feature_ir)
                    layer5_feature_ir = features_5(layer4_feature_ir)

                    layer1_feature_vi = features_1(dup_vi)
                    layer2_feature_vi = features_2(layer1_feature_vi)
                    layer3_feature_vi = features_3(layer2_feature_vi)
                    layer4_feature_vi = features_4(layer3_feature_vi)
                    layer5_feature_vi = features_5(layer4_feature_vi)
                    
                    layer1_feature_ir = gradient(layer1_feature_ir)**2;
                    layer2_feature_ir = gradient(layer2_feature_ir)**2;
                    layer3_feature_ir = gradient(layer3_feature_ir)**2;
                    layer4_feature_ir = gradient(layer4_feature_ir)**2;
                    layer5_feature_ir = gradient(layer5_feature_ir)**2;
                    grad_ir_cnn = torch.mean(layer1_feature_ir)+torch.mean(layer2_feature_ir)+torch.mean(layer3_feature_ir)+torch.mean(layer4_feature_ir)+torch.mean(layer5_feature_ir);
                    grad_ir_cnn /= 5;
                     
                    layer1_feature_vi = gradient(layer1_feature_vi)**2;
                    layer2_feature_vi = gradient(layer2_feature_vi)**2;
                    layer3_feature_vi = gradient(layer3_feature_vi)**2;
                    layer4_feature_vi = gradient(layer4_feature_vi)**2;                     
                    layer5_feature_vi = gradient(layer5_feature_vi)**2;                     
                    grad_vi_cnn = torch.mean(layer1_feature_vi) + torch.mean(layer2_feature_vi)+ torch.mean(layer3_feature_vi)+ torch.mean(layer4_feature_vi)+ torch.mean(layer5_feature_vi);                    
                    grad_vi_cnn /= 5;
                    
                    if args.cuda:
                        grad_ir_cnn = grad_ir_cnn.cuda(args.device);
                        grad_vi_cnn = grad_vi_cnn.cuda(args.device);
                        
                    weightNonInterestedIR_cnn = torch.exp(grad_ir_cnn)/(torch.exp(grad_ir_cnn)+torch.exp(grad_vi_cnn));
                    weightNonInterestedVI_cnn = torch.exp(grad_vi_cnn)/(torch.exp(grad_ir_cnn)+torch.exp(grad_vi_cnn));            

                    
                    # -----------------获取CNN特征权重 结束--------------------                    

                #item1
                item1_IM_loss_cnn = weightNonInterestedIR_cnn*mse_loss(out_f, batch_ir) + weightNonInterestedVI_cnn*mse_loss(out_f,batch_vi);                        
                item1_commonLoss = 1 - ssim_loss(out_rec, batch_vi, normalize = True) + mse_loss((out_rec),(batch_vi)); 
                item1_IM_loss =  item1_IM_loss_cnn + item1_commonLoss;
                item1_IM_loss.backward();                
                optimizer_TB.step()                

            loss_item1_spe += item1_IM_loss_cnn;
            loss_item1_com += item1_commonLoss;

            #MFIF branch
            for _idx in range(MFIF_step):
                optimizer_TB.zero_grad()                                

                
                fea_com = twoBranchesFusionModel.forward_encoder(img_ir_NF,img_vi_FF)
                with torch.no_grad():                
                    fea_com_ivif = twoBranchesFusionModel.forward_encoder(batch_ir,batch_vi)
                    
                out_rec = twoBranchesFusionModel.forward_rec_decoder(fea_com)
                
                fea_fused = twoBranchesFusionModel.forward_MultiTask_branch(fea_com_ivif = fea_com_ivif, fea_com_mfif = fea_com, trainingTag = 2);
                
                out_f = twoBranchesFusionModel.forward_mixed_decoder(fea_com, fea_fused);                                                
                
                item2_commonLoss = 1 - ssim_loss(out_rec, batch_vi, normalize = True) + mse_loss((out_rec),(batch_vi));
                item2_supLoss = mse_loss(out_f,batch_vi)
                item2_clarity_loss = item2_supLoss + item2_commonLoss;
                item2_clarity_loss.backward();                
                optimizer_TB.step()                
                

            loss_item2_spe += item2_supLoss;            
            loss_item2_com += item2_commonLoss;

            


            if idx % step == 0:

                loss_item1_spe /= step
                loss_item2_spe /= step
                loss_item1_com /= step
                loss_item2_com /= step

                mesg = "{}\t Count {} \t Epoch {}/{} \t Batch {}/{} \n " \
                       "IM loss: {:.6f} \n". \
                    format(time.ctime(), idx, e + 1, args.epochs, idx + 1, batch_num, item1_IM_loss_cnn.item())
                print(mesg)

                Loss_list_item1_spe.append(loss_item1_spe.item());
                Loss_list_item1_com.append(loss_item1_com.item());
                Loss_list_item2_spe.append(loss_item2_spe.item());
                Loss_list_item2_com.append(loss_item2_com.item());                

                loss_item1_spe = 0.
                loss_item2_spe = 0.
                loss_item1_com = 0.
                loss_item2_com = 0.

            if (idx+1) % 300 == 0:
                temp_loss = "epoch_" + str(e + 1) + "_batch_" + str(idx + 1) + \
                            "_block_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + ".mat"
                lossChartSave(temp_loss,"item1_spe_loss",Loss_list_item1_spe);
                lossChartSave(temp_loss,"item1_com_loss",Loss_list_item1_com);
                lossChartSave(temp_loss,"item2_spe_loss",Loss_list_item2_spe);
                lossChartSave(temp_loss,"item2_com_loss",Loss_list_item2_com);
                

            if (idx+1) % 700 == 0:
                # save model ever 700 iter.
                #twoBranchesFusionModel.eval()
                twoBranchesFusionModel.cpu()
                
                save_model_filename = "MTFusion_net_epoch_" + str(e + 1) + "_count_" + str(idx+1) + "_twoBranches"  + ".model"
                save_model_path = os.path.join(temp_path_model, save_model_filename)
                torch.save(twoBranchesFusionModel.state_dict(), save_model_path)
                
                
                print('Saving model at ' + save_model_path + '......')
                ##############
                #twoBranchesFusionModel.train()
                if (args.cuda):
                    twoBranchesFusionModel.cuda()

        # save model
        twoBranchesFusionModel.eval()
        twoBranchesFusionModel.cpu()
        save_model_filename = "MTFusion_net" + "_epoch_" + str(e + 1) + "_twoBranches"  + ".model"
        save_model_path = os.path.join(temp_path_model, save_model_filename)
        torch.save(twoBranchesFusionModel.state_dict(), save_model_path)
        ##############
        twoBranchesFusionModel.train()
        if (args.cuda):
            twoBranchesFusionModel.cuda()
        print("\nCheckpoint, trained model saved at: " + save_model_path)


    print("\nDone, trained model saved at", save_model_path)

if __name__ == "__main__":
    main()

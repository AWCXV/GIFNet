

## GIFNet [CVPR 2025]
This is the offical implementation for the paper titled "One Model for ALL: Low-Level Task Interaction Is a Key to Task-Agnostic Image Fusion".

[Paper & Supplement](https://openaccess.thecvf.com/content/CVPR2025/html/Cheng_One_Model_for_ALL_Low-Level_Task_Interaction_Is_a_Key_CVPR_2025_paper.html)

## <img width="40" src="images/environment.png">  Environment
You can setup the required Anaconda environment by running the following prompts:

```cpp
conda create -n GIFNet python=3.8.17
conda activate GIFNet
pip install -r requirements.txt
```

## <img width="32" src="images/test.png"> Test

The **single required checkpoint** is avaiable in the folder "model"

<img width="20" src="images/set1.png"> Arguments:

```cpp
"--test_ir_root": Root path for the infrared input.
"--test_vis_root": Root path for the visible input.
"--VIS_IS_RGB": Visible input is stored in the RGB format.
"--IR_IS_RGB": Infrared input is stored in the RGB format.
"--save_path": Root path for the fused image.
```

<img width="20" src="images/task.png"> Infrared and Visible Image Fusion (IVIF):

```cpp
python test.py  --test_ir_root "images/IVIF/ir" --test_vis_root "images/IVIF/vis" --save_path "outputsIVIF" --VIS_IS_RGB 
```

<img width="20" src="images/task.png"> Multi-Focus Image Fusion (MFIF):

```cpp
python test.py  --test_ir_root "images/MFIF/nf" --test_vis_root "images/MFIF/ff" --save_path "outputsMFIF" --IR_IS_RGB --VIS_IS_RGB
```

<img width="20" src="images/task.png"> Multi-Exposure Image Fusion (MEIF):

```cpp
python test.py  --test_ir_root "images/MEIF/oe" --test_vis_root "images/MEIF/ue" --save_path "outputsMEIF" --IR_IS_RGB --VIS_IS_RGB 
```

<img width="20" src="images/task.png"> Medical Image Fusion:

```cpp
python test.py  --test_ir_root "images/Medical/pet" --test_vis_root "images/Medical/mri" --save_path "outputsMedical" --IR_IS_RGB
```

<img width="20" src="images/task.png"> Near-Infrared and Visible Image Fusion (NIR-VIS)

```cpp
python test.py  --test_ir_root "images/NIR-VIS/nir" --test_vis_root "images/NIR-VIS/vis" --save_path "outputsNIR-VIS" --VIS_IS_RGB
```

<img width="20" src="images/task.png"> Remote Sensing Image Fusion (Remote)

Step1 : Seprately fuse different bands of the multispectral image with the panchromatic image

(Python)
```cpp
python test.py  --test_ir_root "images/Remote/MS_band1" --test_vis_root "images/Remote/PAN" --save_path "outputsRemoteBand1"
python test.py  --test_ir_root "images/Remote/MS_band2" --test_vis_root "images/Remote/PAN" --save_path "outputsRemoteBand2"
python test.py  --test_ir_root "images/Remote/MS_band3" --test_vis_root "images/Remote/PAN" --save_path "outputsRemoteBand3"
python test.py  --test_ir_root "images/Remote/MS_band4" --test_vis_root "images/Remote/PAN" --save_path "outputsRemoteBand4"
```

Step2: Aggregate the separate fused channels together

(Matlab Environment)
```
Matlab_SeparateChannelsIntoFused
```

## <img width="32" src="images/train.png"> Training 

<img width="20" src="images/dataset1.png"> Training Set
- [Baidu Drive (code: x2i6)](https://pan.baidu.com/s/16lCjucwC476dFuxtfFbP3g?pwd=x2i6)
- [Google Drive](https://drive.google.com/file/d/1REIsHqnXEmGGIs4SQoIquUJGzvHDDCUd/view?usp=sharing)

<img width="20" src="images/task.png"> Instructions
1. Extract training data from the zip and put them in the "train_data" folder.

2. Run the following prompt to start the training (important parameters can be modified in the "args.py" file):

```cpp
python train.py --trainDataRoot "./train_data"
```

The trained models will be saved in the "model" folder (automatically created).

## <img width="32" src="images/announcement.png"> Announcement
- 2025-04-15 The training code is now available.
- 2025-03-11 The test code for all image fusion tasks is now available.
- 2025-02-27 This paper has been accepted by CVPR 2025.

## <img width="32" src="images/email.png"> Contact Informaiton
If you have any questions, please contact me at <chunyang_cheng@163.com>.

(Please clearly note your identity, institution, purpose)

## <img width="32" src="images/highlight.png"> Highlight

- **Collaborative Training**: Uniquely demonstrates that collaborative training between low-level fusion tasks yields significant performance improvements by leveraging cross-task synergies.
- **Bridging the Domain Gap**: Introduces a reconstruction task and an augmented RGB-focused joint dataset to improve feature alignment and facilitate effective cross-task collaboration, enhancing model robustness.
- **Versatility**: Advances versatility over multi-task fusion methods by reducing computational costs and eliminating the need for task-specific adaptation.
- **Single-Modality Enhancement**: Pioneers the integration of image fusion with single-modality enhancement, broadening the flexibility and adaptability of fusion models.

### <img width="32" src="images/citation.png"> Citation
If this work is helpful to you, please cite it as:
```
@InProceedings{Cheng_2025_CVPR,
    author    = {Cheng, Chunyang and Xu, Tianyang and Feng, Zhenhua and Wu, Xiaojun and Tang, Zhangyong and Li, Hui and Zhang, Zeyang and Atito, Sara and Awais, Muhammad and Kittler, Josef},
    title     = {One Model for ALL: Low-Level Task Interaction Is a Key to Task-Agnostic Image Fusion},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {28102-28112}
}
```

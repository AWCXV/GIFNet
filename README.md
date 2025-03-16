

<div align="center">
  <img src="images/fig1_11_22.jpg" width="700px" />
  <p>Fig. Supporting single-modality tasks, the adopted low-level interaction between fusion tasks advances the learning of task-agnostic image features, leading to more generalised and efficient image fusion. </p>
</div>

## 1 GIFNet [CVPR 2025]
This is the offical implementation for the paper titled "One Model for ALL: Low-Level Task Interaction Is a Key to Task-Agnostic Image Fusion".

[Paper](https://arxiv.org/abs/2502.19854)

## 2 Environment
You can setup the required Anaconda environment by running the following prompts:

```cpp
conda create -n GIFNet python=3.8.17
conda activate GIFNet
pip install -r requirements.txt
```

## 3 Test

The **single required checkpoint** is avaiable in the folder "model"

### 3.1 Infrared and Visible Image Fusion (IVIF):

(If visible images are stored in the grayscale format, please remove the '--VIS_IS_RGB' prompt.)

```cpp
python test.py  --test_ir_root "images/IVIF/ir" --test_vis_root "images/IVIF/vis" --save_path "outputsIVIF" --VIS_IS_RGB 
```

### 3.2 Multi-Focus Image Fusion (MFIF):

```cpp
python test.py  --test_ir_root "images/MFIF/nf" --test_vis_root "images/MFIF/ff" --save_path "outputsMFIF" --IR_IS_RGB --VIS_IS_RGB
```

### 3.3 Multi-Exposure Image Fusion (MEIF):

```cpp
python test.py  --test_ir_root "images/MEIF/oe" --test_vis_root "images/MEIF/ue" --save_path "outputsMEIF" --IR_IS_RGB --VIS_IS_RGB 
```

### 3.4 Medical Image Fusion:

**The "test.py" file is updated a little on 2025/03/04.**

```cpp
python test.py  --test_ir_root "images/Medical/pet" --test_vis_root "images/Medical/mri" --save_path "outputsMedical" --IR_IS_RGB
```

### 3.5 Near-Infrared and Visible Image Fusion (NIR-VIS)

```cpp
python test.py  --test_ir_root "images/NIR-VIS/nir" --test_vis_root "images/NIR-VIS/vis" --save_path "outputsNIR-VIS" --VIS_IS_RGB
```

### 3.6 Remote Sensing Image Fusion (Remote)

**The "test.py" file is updated a little on 2025/03/11.**

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

## 4 Training

Training Set: [Baidu Drive (code: x2i6)](https://pan.baidu.com/s/16lCjucwC476dFuxtfFbP3g?pwd=x2i6)

Coming...

## 5 Announcement
- 2025-03-11 The test code for all image fusion tasks is now available.
- 2025-02-27 This paper has been accepted by CVPR 2025.

## 6 Contact Informaiton
If you have any questions, please contact me at <chunyang_cheng@163.com>.

(Please clearly note your identity, institution, purpose)

## 7 Highlight

- **Collaborative Training**: Uniquely demonstrates that collaborative training between low-level fusion tasks yields significant performance improvements by leveraging cross-task synergies.
- **Bridging the Domain Gap**: Introduces a reconstruction task and an augmented RGB-focused joint dataset to improve feature alignment and facilitate effective cross-task collaboration, enhancing model robustness.
- **Versatility**: Advances versatility over multi-task fusion methods by reducing computational costs and eliminating the need for task-specific adaptation.
- **Single-Modality Enhancement**: Pioneers the integration of image fusion with single-modality enhancement, broadening the flexibility and adaptability of fusion models.

### 8 Citation
If this work is helpful to you, please cite it as:
```
@inproceedings{cheng2025gifnet,
  title={One Model for ALL: Low-Level Task Interaction Is a Key to Task-Agnostic Image Fusion},
  author={Cheng, Chunyang and Xu, Tianyang and Feng, Zhenhua and Wu, Xiaojun and Tang, Zhangyong and Li, Hui and Zhang, Zeyang and Atito, Sara and Awais, Muhammad and Kittler, Josef},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

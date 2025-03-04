

<div align="center">
  <img src="images/fig1_11_22.jpg" width="700px" />
  <p>Fig. Supporting single-modality tasks, the adopted low-level interaction between fusion tasks advances the learning of task-agnostic image features, leading to more generalised and efficient image fusion. </p>
</div>

## 1 GIFNet [CVPR 2025]
This is the offical implementation for the paper titled "One Model for ALL: Low-Level Task Interaction Is a Key to Task-Agnostic Image Fusion".

[Paper](https://arxiv.org/abs/2502.19854)

## 2 Environment
You can setup the required Anaconda environment by running the following prompts:

```
conda create -n GIFNet python=3.8.17
conda activate GIFNet
pip install -r requirements.txt
```

## 3 Usage

The **single required checkpoint** is avaiable in the folder "model"

### 3.1 Infrared and Visible Image Fusion (IVIF):

(If visible images are stored in the grayscale format, please remove the '--VIS_IS_RGB' prompt.)

```
python test.py  --test_ir_root "images/IVIF/ir" --test_vis_root "images/IVIF/vis" --save_path "outputsIVIF" --VIS_IS_RGB 
```

### 3.2 Multi-Focus Image Fusion (MFIF):

```
python test.py  --test_ir_root "images/MFIF/nf" --test_vis_root "images/MFIF/ff" --save_path "outputsMFIF" --IR_IS_RGB --VIS_IS_RGB
```

### 3.3 Multi-Exposure Image Fusion (MEIF):

```
python test.py  --test_ir_root "images/MEIF/oe" --test_vis_root "images/MEIF/ue" --save_path "outputsMEIF" --IR_IS_RGB --VIS_IS_RGB 
```

### 3.4 Medical Image Fusion:

**The "test.py" file is updated on 2025/03/04.**

```
python test.py  --test_ir_root "images/Medical/pet" --test_vis_root "images/Medical/mri" --save_path "outputsMedical" --IR_IS_RGB
```

### 3.5 Near-Infrared and Visible Image Fusion (NIR-VIS)

```
python test.py  --test_ir_root "images/NIR-VIS/nir" --test_vis_root "images/NIR-VIS/vis" --save_path "outputsNIR-VIS" --VIS_IS_RGB
```

More is coming...

## 4 Announcement
- 2025-02-27 This paper has been accepted by CVPR 2025.

## 5 Contact Informaiton
If you have any questions, please contact me at <chunyang_cheng@163.com>.

## 6 Highlight

- **Collaborative Training**: Uniquely demonstrates that collaborative training between low-level fusion tasks yields significant performance improvements by leveraging cross-task synergies.
- **Bridging the Domain Gap**: Introduces a reconstruction task and an augmented RGB-focused joint dataset to improve feature alignment and facilitate effective cross-task collaboration, enhancing model robustness.
- **Versatility**: Advances versatility over multi-task fusion methods by reducing computational costs and eliminating the need for task-specific adaptation.
- **Single-Modality Enhancement**: Pioneers the integration of image fusion with single-modality enhancement, broadening the flexibility and adaptability of fusion models.

### 7 Citation
If this work is helpful to you, please cite it as:
```
@inproceedings{cheng2025gifnet,
  title={One Model for ALL: Low-Level Task Interaction Is a Key to Task-Agnostic Image Fusion},
  author={Cheng, Chunyang and Xu, Tianyang and Feng, Zhenhua and Wu, Xiaojun and Tang, Zhangyong and Li, Hui and Zhang, Zeyang and Atito, Sara and Awais, Muhammad and Kittler, Josef},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

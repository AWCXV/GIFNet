

<div align="center">
  <img src="images/fig1_11_22.jpg" width="700px" />
  <p>Fig. Supporting single-modality tasks, the adopted low-level interaction between fusion tasks advances the learning of task-agnostic image features, leading to more generalised and efficient image fusion. </p>
</div>

## 1 GIFNet
This is the offical implementation for the paper titled "One Model for ALL: Low-Level Task Interaction Is a Key to Task-Agnostic Image Fusion".

[Paper](https://arxiv.org/abs/2502.19854)

## 2 Environment
```
python 3.8.1
opencv-python 4.9.0.80
torch 2.3.0
matplotlib 3.7.5
```

## 3 Usage

#### The pre-trained model is avaiable in the folder "model"

### Infrared and Visible Image Fusion (IVIF):

(If visible images are stored in the grayscale format, please remove the '--VIS_IS_RGB' prompt.)

```
python test.py  --test_ir_root "images/IVIF/ir" --test_vis_root "images/IVIF/vis" --save_path "outputsIVIF" --VIS_IS_RGB 
```

### Multi-Exposure Image Fusion (MEIF):

```
python test.py  --test_ir_root "images/MEIF/ir" --test_vis_root "images/MEIF/vis" --save_path "outputsMEIF" --IR_IS_RGB --VIS_IS_RGB 
```

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

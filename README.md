# GIFNet

#### Thank you very much for the time devoted to handling and reviewing our work.

## Highlight
- **Innovative Training Strategy**: Introduced a three-branch design with a cross-fusion gating mechanism for effective task interaction.
- **Joint Dataset Creation**: Developed the first RGB-based joint dataset R-MFIV for MFIF and IVIF tasks using a shared reconstruction task and data augmentation.
- **Efficiency and Versatility**: Offered a low-computation, generalized alternative to current high-computation, task-specific methods.
- **Groundbreaking Integration**: Pioneered the integration of image fusion with single-modality vision tasks, extending capabilities beyond traditional image fusion.

<div align="center">
  <img src="images/motivation_1.png" width="500px" />
  <p>Fig. 1: The adopted low-level interaction between fusion tasks advances the learning of task-independent image features, leading to more generalised and efficient image fusion.</p>
</div>

## Environment
```
python 3.8.1
opencv-python 4.9.0.80
torch 2.3.0
matplotlib 3.7.5
```

## Usage
To quickly test our GIFNet on the seen and unsee image fusion tasks, please run the following prompt:

```
python demo.py
```

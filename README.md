# 재활용 품목 분류를 위한 Sementic Segmentation in Bostcamp

## 💻 하나둘셋Net()

### 😎 Members

---

|                                     [공은찬](https://github.com/Chanchan2)                                      |                                       [곽민구](https://github.com/deokgu)                                       |                                      [김준섭](https://github.com/Aweseop)                                       |                                     [김진용](https://github.com/Kim-jy0819)                                     |                                       [심용철](https://github.com/ShimYC)                                       |                               [오재석](https://github.com/dmole20)                                |                                     [최현진](https://github.com/hyeonjini)                                      |
| :-------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------: |
| ![image](https://user-images.githubusercontent.com/35412566/138591221-5c2b12cc-c2db-4679-892f-a0aa034cdf77.png) | ![image](https://user-images.githubusercontent.com/35412566/138591171-7b883dcd-7b83-492e-a251-9eb2960d6e62.png) | ![image](https://user-images.githubusercontent.com/35412566/138591221-5c2b12cc-c2db-4679-892f-a0aa034cdf77.png) | ![image](https://user-images.githubusercontent.com/35412566/138591221-5c2b12cc-c2db-4679-892f-a0aa034cdf77.png) | ![image](https://user-images.githubusercontent.com/35412566/138591221-5c2b12cc-c2db-4679-892f-a0aa034cdf77.png) |                ![image](https://avatars.githubusercontent.com/u/52789601?s=40&v=4)                | ![image](https://github.com/hyeonjini.png) |
| [**TIL**](https://flint-failing-3c9.notion.site/006b28bf92104405834e3fb3ef1fdc99)                                                                                                             |                                [**TIL**](https://github.com/deokgu/deokgu/wiki)                                 |                                                                                                                 |                                                                                                                 |                                                                                                                 | [**TIL**](https://fair-dahlia-cc2.notion.site/BoostCamp-AI-Tech-48bd706756aa49e0b74ca2d2ffda962a) |[**Devlog**](https://velog.io/@choihj94)                                                                                                                 |

---

## 🔎 Competition Overview

<br>

![image](https://user-images.githubusercontent.com/35412566/139359859-ea1469d8-8bd9-41f3-b09e-4b190ab795db.png)

##### 잘 분리 배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 됩니다.

##### 따라서 우리는 사진에서 쓰레기를 Segmentation하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해경을 위한 데이터셋으로는 배경, 일반 쓰레기, 플라스틱, 종이, 유리 등 11종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

##### 여러분에 의해 만들어진 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다.

---

## 🎉 수행결과 best score?

### ✨ 리더보드(대회 진행): **11위** mIoU : 0.751

### 🎊 리더보드(최종):

<br>

---

## 🎮 Requirements

- Linux version 4.4.0-59-generic
- Python >= 3.8.5
- PyTorch >= 1.7.1
- conda >= 4.9.2
- tensorboard >= 2.4.1

## Example
- python train.py \
--model UPlusPlus_Efficient_b5 \
--epochs 200 \
--loss FocalLoss \
--val_json kfold_0_val.json \
--train_json kfold_0_train.json \
--train_augmentation CustomTrainAugmentation \
--batch_size 5
- python test.py python test.py --model_dir model/exp --model_name epoch10.pth --augmentation TestAugmentation

### ⌨ Hardware

- CPU: Intel(R) Xeon(R) Gold 5220 CPU @ 2.20GHz
- GPU: Tesla V100-SXM2-32GB
  <br>

## 🔍 Reference

- [MMsegmentation](https://github.com/open-mmlab/mmsegmentation)
- [RAdam](https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam/radam.py)
  <br>

# 🔨 수행 과정

---

## 1. EDA

<br>

---

## 2. 검증 전략, cv 전략

<br>

---

## 3. 아키텍쳐

<br>

---

### 4. copy & :

<br>

---

## 📂 Archive contents

```
TODO
```

---

## 🛒 Quickstart

# copy_paste_segmentation_coco_format

# 소개
- coco format으로 segmentation 되어있는 쓰레기 데이터를 [Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation](https://arxiv.org/pdf/2012.07177v1.pdf) 기법을 적용하는 코드입니다. 

- main 원본 이미지와 source 원본 이미지가 필요합니다. coco json 파일에서 각 이미지에 대한 segmentation 정보를 추출하여 마스크를 생성합니다. 아래 소개된 어그멘테이션 기법을 적용한 후 source 마스크를 잘라내어, 원본 이미지에 붙이는 방식으로 진행됩니다.


# Augmentation Method used in this repo:
1. Random Horizontal Flip
2. Large Scale Jittering
3. Copy-Paste


# Content
```
segmentation
│
├── baseline_code
├── copy_paste_segmentation_coco_format
│   ├─ check_copy_paste.ipynb
│   ├─ copy_paste.py
│   ├─ create-copy-paste-dataset.py
│   ├─ get_coco_mask.py
│   ├─ README.md
│   ├─ requirements.txt
├── input
│   ├─ data
'''''

```


# Usage 

1. 쉘 스크립트를 사용하고자 한다면 해당 디렉토리에 들어가 다음 명령어를 입력한다.
```
./copy_paste.sh
```


2. 명령어를 따로 입력하고자 한다면 다음과 같은 순서로 명령어를 입력한다.
## 1. 모듈 설치하기
```
pip install -r requirements.txt
```


## 2. 원본 이미지와 json 파일을 통해 segmentation mask 생성
```
python get_coco_mask.py  --input_dir ../../input/data/ --split train_all
```


## 3. 원본 이미지, 원본 mask, 랜덤 이미지, 랜덤 mask로부터 copy_paste
```
python copy_paste.py --input_dir ../../input/data/ --output_dir ../../input/data/ 
```
- 예시)

![image](https://user-images.githubusercontent.com/63527907/139034387-1ec9d9c8-3dcd-4859-9f8a-5ead0fe54f40.png)


## 4. copy_paste mask로부터 coco format json 파일 만들기
```
python create-copy-paste-dataset.py 
```


# check_copy_paste.ipynb 실행 결과

![image](https://user-images.githubusercontent.com/63527907/139034173-370fd30e-42dc-40c3-afe1-00ed559fbb2a.png)


# 참고
- https://github.com/qq995431104/Copy-Paste-for-Semantic-Segmentation
- https://github.com/chrise96/image-to-coco-json-converter



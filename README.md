# MTL (Multi Task Learning) of Flatfish Disease Monitoring Service

## Environment

-   **Python**: `3.8` 이상
-   **NVIDIA Driver Version**: `570.172.08` 이상 (CUDA 11.7 호환 드라이버)
-   **PyTorch**: `1.13.1` (**CUDA 11.7** 버전)
-   **CUDA**: '11.7'

## Installation

1.  **Clone**
    ```bash
    git clone ~
    cd your-repository
    ```

2.  **PyTorch 설치 (CUDA 11.7)**
    기존에 PyTorch가 설치되어 있다면 삭제 후, 아래 명령어로 버전에 맞는 PyTorch를 설치 (Conda 환경 권장)

    ```bash
    # Conda를 사용한 설치
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
    ```

## Usage

1.  datasets 폴더 아래에 cls_labels와 det_labels 폴더에는 각각의 라벨 파일 생성(cls는 하나의 class 번호가 적힌 txt파일, det는 yolo 형식의 txt파일, train/val을 하위 폴더로 가져야함)
2.  images 폴더 아래에 train/val 하위폴더를 갖는 이미지 폴더 생성
3.  yaml 폴더 아래에 data.yaml을 생성
```
path: /path/to/your/datasets/
det_label_dir: det_labels
cls_label_dir: cls_labels
train: images/train
val: images/val
det_labels_train: det_labels/train
cls_labels_train: cls_labels/train
det_labels_val: det_labels/val
cls_labels_val: cls_labels/val
names:
  0: class1
  1: class2
  ...
names_cls:
  0: class1
  1: class2
  2: class3
  ...
```
4.  model의 yaml 파일에 있는 '- [21, 1, Classify, [11]]' 부분의 [11] 값을 cls의 클래스 개수로 변경
5.  위에서 정의한 yaml파일들을 기반으로 train.py 실행

## Conda environment
1. Activate environment
```
$ conda activate fdm-mtl
```

2. Deactivate environment
```
$ conda deactivate
```


## Experiment Server
- sbs71 (177.77)
  - /home/user/fdm/halibut-mtl
- sbs32 (184.197)
  - /home/fabric/fdm/halibut-mtl
- data (not synced on GitHub)
  - halibut-mtl/datasets
```
drwxr-xr-x 10 fabric docker   180 Aug  8 17:25 .
drwxr-xr-x 13 fabric docker  4096 Aug  8 18:37 ..
drwxr-xr-x  4 fabric docker    42 Aug  8 17:18 cls_labels
drwxr-xr-x  4 fabric docker    42 Aug  8 17:20 cls_labels_merged
drwxr-xr-x  4 fabric docker    42 Aug  8 17:16 det_labels
drwxr-xr-x  5 fabric docker    59 Aug  8 17:14 images
drwxr-xr-x  4 fabric docker    42 Aug  8 17:22 jsons
drwxr-xr-x  2 fabric docker 20480 Aug  8 17:23 jsons_orphaned
drwxr-xr-x  4 fabric docker    86 Aug  8 17:27 labels
drwxr-xr-x  4 fabric docker    86 Aug  8 17:25 shrink
```  
  - *.pt
```
./halibut/result/mtl2/weights/last.pt
./halibut/result/mtl2/weights/best.pt
./halibut/result/mtl4/weights/last.pt
./halibut/result/mtl4/weights/best.pt
./halibut/result/mtl10/weights/last.pt
./halibut/result/mtl10/weights/best.pt
./halibut/result/mtl11/weights/last.pt
./halibut/result/mtl11/weights/best.pt
./halibut/test3/weights/last.pt
./halibut/test3/weights/best.pt
./halibut/y11n_merged_no_weighted_class_v3/weights/last.pt
./halibut/y11n_merged_no_weighted_class_v3/weights/best.pt
./weights/yolo11l.pt
./weights/yolo11m.pt
./weights/yolo11s.pt
./weights/yolo11x.pt
./weights/yolo11n.pt
./weights/yolov8l.pt
./weights/yolov8m.pt
./weights/yolov8n.pt
./weights/yolov8s.pt
./weights/yolov8x.pt
./yolo11n.pt
./result/y8n_merged_no_weighted_class_300/weights/last.pt
./result/y8n_merged_no_weighted_class_300/weights/best.pt
./result/y8n_merged_weighted_class_300/weights/last.pt
./result/y8n_merged_weighted_class_300/weights/best.pt
./result/y8s_merged_no_weighted_class/weights/last.pt
./result/y8s_merged_no_weighted_class/weights/best.pt
./result/y8s_merged_weighted_class/weights/last.pt
./result/y8s_merged_weighted_class/weights/best.pt
./result/y11n_merged_no_weighted_class/weights/last.pt
./result/y11n_merged_no_weighted_class/weights/best.pt
./result/cls_original/weights/last.pt
./result/cls_original/weights/best.pt
./result/det_original/weights/last.pt
./result/det_original/weights/best.pt
./result/det_shrink/weights/last.pt
./result/det_shrink/weights/best.pt
./result/y11n_merged_weighted_class/weights/last.pt
./result/y11n_merged_weighted_class/weights/best.pt
```
  - *.log
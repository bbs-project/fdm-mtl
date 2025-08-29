import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

# ultralytics의 LetterBox와 augmentations 네임스페이스를 가져옵니다.
from ultralytics.data.augment import LetterBox, Compose, Format, v8_transforms, classify_transforms
from ultralytics.utils import LOGGER
from ultralytics.utils.instance import Instances

class MultiTaskDataset(Dataset):
    """
    A custom dataset for multi-task learning, designed to be compatible with the ultralytics framework.
    It returns data in a format that our custom collate_fn can process, which in turn is compatible
    with the rest of the ultralytics pipeline.
    """
    def __init__(self, image_dir, det_label_dir, cls_label_dir, imgsz=640, augment=False, rect=False, hyp=None):
        self.imgsz = imgsz
        self.augment = augment
        self.rect = rect

        # 이미지와 라벨 경로 설정
        self.samples = self._gather_samples(image_dir, det_label_dir, cls_label_dir)
        
        # 샘플링 로직
        if len(self.samples) > 200:
            import random
            random.shuffle(self.samples)
            self.samples = self.samples[:200]
            LOGGER.info(f"--- Sampled 200 images for faster debugging ---")

        # 변환(transform) 설정
        self.transforms = self._build_transforms(hyp)

    def _gather_samples(self, image_dir, det_label_dir, cls_label_dir):
        """Gathers all image and label paths."""
        samples = []
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

        for img_path in image_paths:
            base_filename = os.path.splitext(os.path.basename(img_path))[0]
            det_lbl_path = os.path.join(det_label_dir, base_filename + '.txt')
            cls_lbl_path = os.path.join(cls_label_dir, base_filename + '.txt')

            if os.path.exists(det_lbl_path) or os.path.exists(cls_lbl_path):
                samples.append({
                    'image': img_path,
                    'det_label_path': det_lbl_path if os.path.exists(det_lbl_path) else None,
                    'cls_label_path': cls_lbl_path if os.path.exists(cls_lbl_path) else None
                })
        return samples

    def _build_transforms(self, hyp):
        """Builds and returns the image transformations pipeline."""
        # hyp가 None일 경우를 대비해 기본값 설정
        hyp = hyp if hyp is not None else {}
        
        # LetterBox를 Compose 리스트 안에 넣어서 다른 변환과 함께 사용 가능하도록 함
        transforms = Compose([LetterBox(self.imgsz, auto=self.rect, stride=32)])
        
        # 표준 YOLODataset과 동일하게 Format 변환을 추가
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                batch_idx=True,
                # 우리 모델은 mask, keypoint, obb를 사용하지 않으므로 False로 설정
                return_mask=False,
                return_keypoint=False,
                return_obb=False,
                # hyp에서 관련 값을 가져오거나 기본값 사용
                mask_ratio=hyp.get('mask_ratio', 4),
                mask_overlap=hyp.get('overlap_mask', True)
            )
        )
        return transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        img_path = sample_info['image']

        # 이미지 로드
        img = cv2.imread(img_path)
        assert img is not None, f"Image Not Found {img_path}"
        h0, w0 = img.shape[:2]
        
        # 라벨 로드
        det_labels = self._load_det_labels(sample_info['det_label_path'])
        cls_label = self._load_cls_label(sample_info['cls_label_path']) # 이미지 전체 분류 라벨

        # Detection을 위한 Instances 객체와 cls 분리
        if det_labels.size > 0:
            instances = Instances(bboxes=det_labels[:, 1:], segments=np.zeros((len(det_labels), 0, 2), dtype=np.float32))
            cls = det_labels[:, 0:1] # 객체 분류 라벨 (N, 1) shape 유지
        else:
            instances = Instances(bboxes=np.empty((0, 4)), segments=np.empty((0, 0, 2), dtype=np.float32))
            cls = np.empty((0, 1))

        # 변환을 위한 labels 딕셔너리
        labels = {
            'img': img,
            'instances': instances,
            'cls': cls,
            'im_file': img_path,
            'ori_shape': (h0, w0),
            'resized_shape': (self.imgsz, self.imgsz),
        }

        # 변환 적용
        labels = self.transforms(labels)
        
        # Format 변환은 'img'를 [C, H, W]의 torch.Tensor로 반환하지만, 0-255 범위의 uint8입니다.
        # 0-1 범위의 float32로 정규화해야 합니다.
        # labels['img']는 이미 torch.Tensor이므로 from_numpy가 필요 없습니다.
        labels['img'] = labels['img'].float() / 255.0
        
        # 우리 태스크에 필요한 custom_cls_label을 추가합니다.
        labels['custom_cls_label'] = torch.tensor(cls_label, dtype=torch.long)
        
        # Validator가 요구하는 ratio_pad가 없는 경우를 대비해 추가
        if 'ratio_pad' not in labels or labels['ratio_pad'] is None:
            r = min(self.imgsz / h0, self.imgsz / w0)
            new_unpad = int(round(w0 * r)), int(round(h0 * r))
            dw, dh = self.imgsz - new_unpad[0], self.imgsz - new_unpad[1]
            dw /= 2  # divide padding into 2 sides
            dh /= 2
            labels['ratio_pad'] = (r, r), (dw, dh)

        # Format 변환이 반환한 딕셔너리가 최종 결과물입니다.
        return labels

    def _load_det_labels(self, path):
        """Loads detection labels from a file."""
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                lines = f.readlines()
            labels = [list(map(float, line.strip().split())) for line in lines if len(line.strip().split()) == 5]
            return np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)
        return np.zeros((0, 5), dtype=np.float32)

    def _load_cls_label(self, path):
        """Loads a classification label from a file."""
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                return int(f.read().strip())
        return -1  # -1은 CrossEntropyLoss에서 무시됩니다.

    @staticmethod
    def collate_fn(batch):
        """
        원본 ultralytics.data.dataset.YOLODataset.collate_fn의 로직을 기반으로,
        우리 멀티태스크 학습에 필요한 'custom_cls_label'을 처리하고,
        'batch_idx' 관련 shape 오류를 수정하는 커스텀 collate 함수.
        """
        new_batch = {}
        # 키가 누락되는 경우를 방지하기 위해 모든 샘플의 모든 키를 수집
        all_keys = set(k for d in batch for k in d.keys())

        for k in all_keys:
            # 현재 키를 가진 샘플들의 값만 추출
            values = [d[k] for d in batch if k in d]

            if not values:
                continue

            if k == 'img':
                # 샘플당 1개: [B, C, H, W]
                new_batch[k] = torch.stack(values, 0)
            elif k == 'custom_cls_label':
                # 샘플당 1개: [B]
                new_batch[k] = torch.stack(values, 0)
            elif k in {'cls', 'bboxes'}:  # batch_idx는 이제 여기서 처리하지 않음
                # 샘플당 N개: [Total_N, D]
                # 비어있는 텐서가 있을 수 있으므로, 필터링 후 연결
                valid_values = [v for v in values if v.numel() > 0]
                if valid_values:
                    new_batch[k] = torch.cat(valid_values, 0)
                else:
                    # 모든 샘플에 해당 키의 데이터가 없는 경우, 올바른 shape의 비어있는 텐서 생성
                    if k == 'cls':
                        new_batch[k] = torch.empty(0, 1, dtype=torch.long) # cls는 long 타입이어야 함
                    elif k == 'bboxes':
                        new_batch[k] = torch.empty(0, 4, dtype=torch.float32)
            else:
                # im_file, ori_shape, batch_idx 등 메타데이터는 리스트로 유지
                new_batch[k] = values

        # ultralytics 원본 collate_fn과 유사하게, 마지막에 batch_idx를 처리
        if 'batch_idx' in new_batch:
            # batch_idx는 리스트 안에 텐서들이 들어있는 형태이므로 torch.cat으로 합침
            new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)

        return new_batch
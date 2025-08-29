# mtl_loader.py (캐시 경로 관리 기능 추가)

import os
from pathlib import Path
from itertools import repeat
from multiprocessing.pool import ThreadPool
import random

import numpy as np
import torch
import cv2
from ultralytics.data.dataset import YOLODataset, DATASET_CACHE_VERSION, segments2boxes
from ultralytics.utils import DEFAULT_CFG, LOGGER, TQDM, NUM_THREADS, RANK
from ultralytics.data.utils import get_hash, HELP_URL, save_dataset_cache_file, load_dataset_cache_file, img2label_paths


class MultiTaskDataset(YOLODataset):
    """
    YOLODataset을 상속받아 멀티태스크(detection + classification) 학습을 지원하는 커스텀 데이터셋.
    데이터셋 구성에 따라 캐시 파일 경로를 동적으로 변경하는 기능을 포함합니다.
    """
    
    def __init__(self, *args, data=None, data_config_path=None, **kwargs):
        if data is None:
            raise ValueError("The 'data' dictionary must be provided to MultiTaskDataset.")
        
        self.det_label_dir = data.get('det_label_dir')
        self.cls_label_dir = data.get('cls_label_dir')
        self.data_config_path = data_config_path # data.yaml 또는 data_merged.yaml 경로
        
        if not self.det_label_dir or not self.cls_label_dir:
            raise ValueError("data.yaml에 'det_label_dir'와 'cls_label_dir'를 반드시 설정해야 합니다.")
        
        super().__init__(*args, data=data, **kwargs)

    def get_labels(self):
        """
        get_labels를 오버라이드하여 캐시 경로를 동적으로 생성합니다.
        - data.yaml -> .../images/cache/original/train.cache
        - data_merged.yaml -> .../images/cache/merged/train.cache
        """
        img_path_component = f'{os.sep}images{os.sep}'
        det_path_component = f'{os.sep}{self.det_label_dir}{os.sep}'
        
        self.label_files = [
            p.replace(img_path_component, det_path_component).replace(Path(p).suffix, '.txt')
            for p in self.im_files
        ]
        
        self.cls_label_files = [
            p.replace(f'{os.sep}{self.det_label_dir}{os.sep}', f'{os.sep}{self.cls_label_dir}{os.sep}')
            for p in self.label_files
        ]

        if not self.im_files:
             return []

        # --- 캐시 경로 생성 로직 ---
        # data_config_path를 기반으로 캐시 하위 디렉토리 이름 결정
        if self.data_config_path and 'merged' in str(self.data_config_path):
            cache_name = 'merged'
        else:
            cache_name = 'original'

        # 캐시 디렉토리 생성
        # self.img_path는 .../datasets/images/train 과 같은 형태이므로, 부모로 가서 images 폴더를 찾음
        cache_dir = Path(self.img_path).parent / 'cache' / cache_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        # 최종 캐시 파일 경로 설정
        # self.img_path는 train/val 같은 split 이름을 포함하므로 Path(self.img_path).name으로 split 이름을 가져옴
        cache_path = cache_dir / f'{Path(self.img_path).name}.cache'
        LOGGER.info(f"Using cache path: {cache_path}")
        # --- 캐시 경로 생성 로직 끝 ---

        try:
            cache, exists = load_dataset_cache_file(cache_path), True
            assert cache['version'] == DATASET_CACHE_VERSION
            assert cache['hash'] == get_hash(self.im_files + self.label_files)
        except (FileNotFoundError, AssertionError, AttributeError, TypeError):
            cache, exists = self.cache_labels(cache_path), False

        nf, nm, ne, nc, n = cache.pop("results")
        if exists and RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm} missing, {ne} empty, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)
        if nf == 0:
            LOGGER.warning(f'WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. See {HELP_URL}')

        cache.pop('hash', None)
        cache.pop('version')
        
        self.labels = cache.pop('labels')
        self.im_files = [lb['im_file'] for lb in self.labels]

        return self.labels

    def __getitem__(self, index):
        """
        YOLODataset.__getitem__을 기반으로 하여, custom_cls_label을 추가하도록 수정.
        super()를 호출하여 ratio_pad와 같은 모든 필수 키들이 보존되도록 합니다.
        """
        # YOLODataset의 __getitem__ 로직을 그대로 호출하여
        # 'ratio_pad'를 포함한 모든 표준 변환 및 키 추가를 처리합니다.
        label = super().__getitem__(index)

        # 사용자 정의 분류 라벨을 결과 딕셔너리에 추가합니다.
        # 이 라벨은 augmentation의 영향을 받지 않아야 하므로 원본 값을 사용합니다.
        if 'custom_cls_label' in self.labels[index]:
            label['custom_cls_label'] = self.labels[index]['custom_cls_label']
        
        return label

    def cache_labels(self, path=Path('./labels.cache')):
        x = {'labels': []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []
        desc = f"""{self.prefix}Scanning '{path.parent / path.stem}' for images and labels..."""
        
        with ThreadPool(NUM_THREADS) as pool:
            pbar = TQDM(pool.imap(self._verify_image_and_labels, zip(self.im_files, self.label_files, self.cls_label_files)), desc=desc, total=len(self.im_files))
            for im_file, det_lb, cls_lb, shape, segments, keypoints, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if nf_f > 0:
                    x['labels'].append(dict(
                        im_file=im_file,
                        shape=shape, # shape는 tuple 그대로 저장
                        cls=det_lb[:, 0:1] if det_lb.size else np.empty((0, 1)),
                        bboxes=det_lb[:, 1:] if det_lb.size else np.empty((0, 4)),
                        segments=segments if segments is not None else [],
                        keypoints=keypoints,
                        normalized=True,
                        bbox_format='xywh',
                        custom_cls_label=cls_lb
                    ))
                if msg:
                    msgs.append(msg)
                pbar.desc = f"""{desc}{nf} images, {nm} missing, {ne} empty, {nc} corrupt"""
        
        pbar.close()
        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{self.prefix}WARNING ⚠️ No labels found in {path}.')
        
        x['hash'] = get_hash(self.im_files + self.label_files)
        x['results'] = nf, nm, ne, nc, len(self.im_files)
        x['msgs'] = msgs
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def _verify_image_and_labels(self, args):
        im_file, det_lbl_path, cls_lbl_path = args
        nm, nf, ne, nc = 0, 0, 0, 0
        msg = ''
        
        try:
            im = cv2.imread(im_file)
            if im is None:
                raise FileNotFoundError(f"Image not found {im_file}")
            shape = im.shape[:2]
        except Exception as e:
            nc = 1
            msg = f'WARNING ⚠️ Ignoring corrupted image {im_file}: {e}'
            return None, None, None, None, None, None, nm, nf, ne, nc, msg

        det_lb = np.zeros((0, 5), dtype=np.float32)
        segments = []
        keypoints = None
        
        if not os.path.exists(det_lbl_path):
            nm = 1
        else:
            try:
                with open(det_lbl_path) as f:
                    lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if not lb:
                    ne = 1
                else:
                    if any(len(x) > 5 for x in lb):
                        classes = np.array([x[0] for x in lb], dtype=np.float32)
                        segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]
                        lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)
                    det_lb = np.array(lb, dtype=np.float32)
                    nf = 1
            except Exception as e:
                nc = 1
                msg = f'WARNING ⚠️ Ignoring corrupted det label {det_lbl_path}: {e}'

        cls_lb = -1
        if os.path.exists(cls_lbl_path):
            try:
                with open(cls_lbl_path, 'r') as f:
                    cls_lb = int(f.read().strip())
            except Exception as e:
                nc = 1
                msg = f'WARNING ⚠️ Ignoring corrupted cls label {cls_lbl_path}: {e}'
        
        return im_file, det_lb, cls_lb, shape, segments, keypoints, nm, nf, ne, nc, msg

    @staticmethod
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return {}

        custom_cls_labels = [b.pop('custom_cls_label') for b in batch]
        new_batch = YOLODataset.collate_fn(batch)
        new_batch['custom_cls_label'] = torch.tensor(custom_cls_labels, dtype=torch.long)
            
        return new_batch

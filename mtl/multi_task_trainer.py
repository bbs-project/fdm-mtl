# multi_task_trainer.py

import torch
from tqdm import tqdm
import numpy as np
import os
from collections import Counter
from pathlib import Path

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import RANK, LOGGER, DEFAULT_CFG
from ultralytics.utils.torch_utils import de_parallel

# Our custom modules
from mtl.mtl_loader import MultiTaskDataset
from mtl.multi_task_model import MultiTaskModel
from mtl.multi_task_validator import MultiTaskValidator


class MultiTaskTrainer(DetectionTrainer):
    """
    A custom trainer for multi-task learning that integrates a custom model, data loader,
    and validator, while leveraging the base trainer's training loop.
    It also calculates and applies class weights to the classification loss.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the MultiTaskTrainer, calculates class weights for the classification task.
        """
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self.cls_weights = None
        if self.args.task == 'detect': # In our setup, this indicates multi-task
            self.cls_weights = self._calculate_class_weights()
            if self.cls_weights is not None:
                LOGGER.info(f"Classification class weights calculated and will be applied: {self.cls_weights}")

    def _calculate_class_weights(self):
        """
        Calculates class weights based on the inverse number of samples for the classification task.
        Weights are normalized to sum to the number of classes. This method safely reads configuration
        from the model and data YAML files.
        """
        try:
            LOGGER.info("Calculating classification class weights...")

            # 1. Get nc_cls from the model configuration file (safe method)
            model_yaml_path = self.args.model
            if not model_yaml_path or not Path(model_yaml_path).exists():
                LOGGER.warning(f"⚠️ Model YAML path not specified or file not found at '{model_yaml_path}'. Skipping weight calculation.")
                return None
            
            from ultralytics.utils import YAML  # Local import
            model_cfg = YAML.load(model_yaml_path)
            num_classes = model_cfg.get('nc_cls')
            if not num_classes:
                LOGGER.warning(f"⚠️ 'nc_cls' not found in '{model_yaml_path}'. Skipping weight calculation.")
                return None

            # 2. Get necessary paths from data config
            data_cfg = self.data
            if 'path' not in data_cfg or 'cls_label_dir' not in data_cfg:
                LOGGER.warning("⚠️ 'path' or 'cls_label_dir' missing from data.yaml. Skipping weight calculation.")
                return None

            base_path = Path(data_cfg['path'])
            cls_label_path = base_path / data_cfg['cls_label_dir'] / 'train'

            if not cls_label_path.exists():
                LOGGER.warning(f"⚠️ Classification label directory not found, skipping weight calculation: {cls_label_path}")
                return None

            # 3. Count samples per class
            counts = Counter()
            label_files = list(cls_label_path.glob('*.txt'))
            if not label_files:
                LOGGER.warning(f"⚠️ No label files found in {cls_label_path}, skipping weight calculation.")
                return None

            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        class_index = int(f.read().strip())
                        if 0 <= class_index < num_classes:
                            counts[class_index] += 1
                except Exception:
                    pass  # Ignore parsing errors

            # 4. Calculate weights using Inverse Number of Samples
            weights = np.zeros(num_classes)
            for i in range(num_classes):
                weights[i] = 1.0 / (counts.get(i, 0) + 1e-6) if counts.get(i, 0) > 0 else 1.0

            # 5. Normalize weights
            total_weight = np.sum(weights)
            if total_weight == 0:
                LOGGER.warning("⚠️ Total weight is zero, cannot normalize. Returning uniform weights.")
                return torch.ones(num_classes, dtype=torch.float).to(self.device)
                
            normalized_weights = (weights / total_weight) * num_classes
            
            return torch.from_numpy(normalized_weights).float().to(self.device)

        except Exception as e:
            LOGGER.error(f"❌ Error calculating class weights: {e}")
            return None

    def get_validator(self):
        """
        Returns a MultiTaskValidator instance for validation.
        """
        if self.validator is None:
            self.validator = MultiTaskValidator(self.test_loader, save_dir=self.save_dir, args=self.args)
        return self.validator

    def get_model(self, cfg=None, weights=None, verbose=True, deactivate_weights=True):
        """
        Returns a MultiTaskModel instance, passing the calculated class weights.
        The model's config (cfg) is derived from self.args.model.
        """
        # The `cfg` argument from the parent call is typically the model's YAML path string.
        # We ensure our MultiTaskModel gets this path to load its specific architecture.
        model_config_path = self.args.model
        
        model = MultiTaskModel(
            cfg=model_config_path,
            ch=3, 
            nc=self.data['nc'], 
            # nc_cls is now read by the model from its own YAML, so we don't pass it here.
            verbose=verbose and RANK == -1, 
            args=self.args,
            class_weights=None # Pass the calculated weights
            #class_weights=self.cls_weights
        )
        if weights:
            model.load(weights)
        
        return model

    def build_dataset(self, img_path, mode='train', batch=None):
        """
        Builds and returns a MultiTaskDataset, passing the data config path for dynamic cache handling.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        dataset = MultiTaskDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == 'train',
            hyp=self.args,
            rect=mode=='val',
            data=self.data,
            # Pass the data config path to enable dynamic cache directory selection
            data_config_path=self.args.data,
            prefix=f'{mode}: '
        )
        return dataset

    def preprocess_batch(self, batch):
        """
        Preprocesses a batch by moving 'custom_cls_label' to the device.
        """
        # First, let the base class handle the standard items like 'img'.
        batch = super().preprocess_batch(batch)
        
        # Now, move our custom data to the correct device.
        if 'custom_cls_label' in batch:
            batch['custom_cls_label'] = batch['custom_cls_label'].to(self.device)
        
        return batch

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """
        Returns a dataloader for the MultiTaskDataset.
        """
        assert mode in {'train', 'val'}
        from ultralytics.data.build import build_dataloader
        from ultralytics.utils.torch_utils import torch_distributed_zero_first

        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)

        shuffle = mode == 'train'
        if getattr(dataset, 'rect', False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == 'train' else self.args.workers * 2
        
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)

    def set_model_attributes(self):
        """
        Sets model attributes including custom loss names from our MultiTaskModel.
        """
        super().set_model_attributes()
        # Ensure loss_names are correctly set from the model, which might now include weighted loss
        if hasattr(self.model, 'loss_names'):
             self.loss_names = self.model.loss_names

    def progress_string(self):
        """Returns a formatted string describing training progress."""
        n = 2 + len(self.loss_names) + 2  # Epoch, GPU_mem, losses, Instances, Size
        return ('\n' + '%11s' * n) % ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

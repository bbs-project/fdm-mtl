import os
from copy import deepcopy

import torch
import torch.nn as nn

from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.head import Detect, Classify
from ultralytics.utils import LOGGER
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.nn.tasks import initialize_weights

class MultiTaskModel(DetectionModel):
    """
    A custom YOLO-based model for multi-task learning (detection + classification).
    This model is designed to return outputs from both the detection and classification heads.
    """

    def __init__(self, cfg="halibut-mtl.yaml", ch=3, nc=None, nc_cls=None, verbose=True, args=None, class_weights=None):
        # args는 여기서 저장하지 않습니다. Trainer가 주입해 줄 것입니다.
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=False)
        
        if nc_cls is None:
            nc_cls = self.yaml.get('nc_cls')
        if not nc_cls:
            raise ValueError("'nc_cls' must be defined in the model YAML or passed as an argument.")

        # --- Head 인덱스 찾기 및 수정 ---
        self.det_head_idx = -1
        self.cls_head_idx = -1
        for i, m in enumerate(self.model):
            if isinstance(m, Detect):
                self.det_head_idx = i
            if isinstance(m, Classify):
                self.cls_head_idx = i
                if hasattr(m, 'linear') and m.linear.out_features != nc_cls:
                    LOGGER.info(f"Correcting Classify head output from {m.linear.out_features} to {nc_cls}")
                    m.linear = nn.Linear(m.linear.in_features, nc_cls)

        if self.det_head_idx == -1 or self.cls_head_idx == -1:
            raise ValueError("Both Detect and Classify heads must be in the model.")

        # --- 손실 함수 지연 초기화(Lazy Initialization) ---
        self.det_criterion = None
        # Pass weights to CrossEntropyLoss if they are provided
        self.cls_criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        self.loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'classify_loss']

        initialize_weights(self.model[self.cls_head_idx])

        if verbose:
            self.info()
            LOGGER.info("")

    def forward(self, x, *args, **kwargs):
        """
        Handles both the initial stride calculation call during __init__ and
        the regular forward passes during training/inference.
        """
        batch = x if isinstance(x, dict) else None
        x = x["img"] if isinstance(x, dict) else x
        
        # Check if the model is fully initialized (i.e., not in the middle of stride calculation)
        # This is the most robust way to handle the special __init__-time forward call.
        is_initialized = hasattr(self, 'det_head_idx') and hasattr(self, 'cls_head_idx')

        det_output = None
        cls_output = None
        y = []

        for i, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            
            x = m(x)  # run the module
            
            y.append(x if m.i in self.save else None)  # save output if specified

            # Only capture specific head outputs if the model is fully initialized
            if is_initialized:
                if i == self.det_head_idx:
                    det_output = x
                if i == self.cls_head_idx:
                    cls_output = x
        
        # --- Execution path branching ---

        if not is_initialized:
            # Scenario A: Called during DetectionModel.__init__ for stride calculation.
            # The final 'x' is the output of the Detect head (a list of tensors),
            # which is exactly what the stride calculation code expects.
            return x
        else:
            # Scenario B: Normal operation (training or inference).
            if self.training:
                return self.loss(batch, (det_output, cls_output))
            else:
                return det_output, cls_output

    def loss(self, batch, preds=None):
        # --- 지연 초기화: det_criterion이 없을 때만 생성 ---
        if self.det_criterion is None:
            # 이 시점에는 Trainer가 self.args를 설정해 주었으므로 안전합니다.
            self.det_criterion = v8DetectionLoss(self)

        if preds is None:
            # This path is taken when loss is called directly, e.g. during validation
            preds = self.forward(batch)

        det_output, cls_output = preds
        device = batch['img'].device

        # --- 1. Detection Loss ---
        loss_det, loss_items_det = self.det_criterion(det_output, batch)

        # --- 2. Classification Loss ---
        cls_labels = batch.get('custom_cls_label')
        if cls_labels is not None:
            # During validation, cls_output is a tuple (probabilities, logits)
            # We need the logits (second element) for cross_entropy
            cls_input = cls_output[1] if isinstance(cls_output, tuple) else cls_output
            
            # Move cls_labels to the same device as cls_input to prevent device mismatch errors during validation.
            cls_labels = cls_labels.to(cls_input.device)

            loss_cls = self.cls_criterion(cls_input, cls_labels.long())
        else:
            loss_cls = torch.tensor(0.0, device=device)

        # --- 3. Combine Losses ---
        total_loss = 0.5*loss_det + 0.5*loss_cls
        # Ensure loss_items has 4 elements to match loss_names
        loss_items = torch.cat((loss_items_det, loss_cls.detach().unsqueeze(0)))
        
        return total_loss, loss_items

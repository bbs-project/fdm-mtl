# multi_task_validator.py

import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import ClassifyMetrics, ConfusionMatrix
from ultralytics.utils import LOGGER, ops

class MultiTaskValidator(DetectionValidator):
    """
    MultiTaskValidator는 DetectionValidator를 상속받아 다음과 같은 기능을 추가합니다:
    1. 분류(Classification) 성능 지표(Top-1, Top-5 Accuracy)를 계산합니다.
    2. 탐지(Detection)와 분류(Classification) 성능을 모두 포함하는 종합적인 통계 및 fitness 점수를 반환합니다.
    3. 최종 검증 결과 출력에 분류 성능을 함께 표시합니다.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, args, _callbacks)
        # 분류 성능 측정을 위한 metrics 객체 초기화
        self.cls_metrics = ClassifyMetrics()
        self.cls_preds = []
        self.cls_targets = []
        # self.names_cls는 self.data가 설정된 후인 init_metrics에서 초기화합니다.

    def init_metrics(self, model):
        super().init_metrics(model)
        # self.data는 __call__ 메서드에서 trainer로부터 설정되므로, 이 시점에서는 안전합니다.
        self.names_cls = self.data.get('names_cls', self.data['names'])
        # 분류 작업에 대한 Confusion Matrix 초기화. `names` 인자를 사용합니다.
        self.cls_confusion_matrix = ConfusionMatrix(names=self.names_cls, task='classify')

        # =====👇 여기를 추가해야 합니다. =====
        # 매 검증 실행 시, 이전 결과를 저장하던 리스트를 초기화하여 메모리 누수를 방지합니다.
        self.cls_preds = []
        self.cls_targets = []
        # =====👆 여기까지 추가 =====

    def postprocess(self, preds):
        """
        모델의 출력을 후처리합니다.
        (det_output, cls_output) 튜플을 받아 det_output만 NMS에 전달하고,
        cls_output은 나중에 사용하기 위해 저장합니다.
        """
        det_output, cls_output = preds
        self.batch_cls_preds = cls_output  # update_metrics에서 사용하기 위해 저장
        
        # 부모 클래스의 postprocess는 NMS를 수행하여 탐지 결과만 반환
        # DetectionValidator의 postprocess는 텐서 하나만 받으므로 det_output만 전달
        return super().postprocess(det_output)

    def update_metrics(self, preds, batch):
        """탐지 및 분류 메트릭을 모두 업데이트합니다."""
        # 1. Detection Metrics 업데이트 (부모 클래스 로직 재사용)
        super().update_metrics(preds, batch)

        # 2. Classification Metrics 업데이트
        cls_labels = batch.get('custom_cls_label')
        if cls_labels is not None and hasattr(self, 'batch_cls_preds') and self.batch_cls_preds is not None:
            # 검증 모드에서 Classify 헤드는 (확률, 로짓) 튜플을 반환합니다.
            if isinstance(self.batch_cls_preds, tuple):
                cls_probs = self.batch_cls_preds[0]
                cls_logits = self.batch_cls_preds[1]
            else:  # 예외적인 경우 (예: 학습 모드)
                cls_probs = self.batch_cls_preds.softmax(1)
                cls_logits = self.batch_cls_preds

            n5 = min(len(self.names_cls), 5)
            self.cls_preds.append(cls_logits.argsort(1, descending=True)[:, :n5].cpu())
            self.cls_targets.append(cls_labels.cpu())
            
            # Confusion matrix는 로짓으로 계산
            for p, t in zip(cls_logits.argmax(1).cpu().numpy(), cls_labels.cpu().numpy()):
                self.cls_confusion_matrix.matrix[p, t] += 1

    def get_stats(self):
        """탐지 및 분류 통계를 결합하여 반환합니다."""
        # 1. Detection 통계 가져오기
        stats = super().get_stats()

        # 2. Classification 통계 계산 및 추가
        if self.cls_preds and self.cls_targets:
            self.cls_metrics.process(targets=self.cls_targets, pred=self.cls_preds)
            stats['top1_acc'] = self.cls_metrics.top1
            stats['top5_acc'] = self.cls_metrics.top5
            
            # F1-Score 계산 (from confusion matrix)
            matrix = self.cls_confusion_matrix.matrix
            tp = matrix.diagonal()
            fp = matrix.sum(1) - tp
            fn = matrix.sum(0) - tp
            
            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
            
            stats['f1_score'] = f1.mean() if f1 is not None else 0.0

        else:
            stats['top1_acc'] = 0.0
            stats['top5_acc'] = 0.0
            stats['f1_score'] = 0.0

        # 3. Fitness 점수 재계산 (탐지 성능 80% + 분류 정확도 10% + F1 10%)
        det_fitness = stats.get('fitness', 0.0)
        cls_acc_fitness = stats.get('top1_acc', 0.0)
        cls_f1_fitness = stats.get('f1_score', 0.0)
        stats['fitness'] = det_fitness * 0.8 + cls_acc_fitness * 0.1 + cls_f1_fitness * 0.1
        
        # print_results가 참조할 수 있도록 계산된 최종 stats를 self.stats에 저장합니다.
        self.stats = stats
        return stats

    def print_results(self):
        """탐지 및 분류 결과를 모두 출력합니다."""
        super().print_results()  # 탐지 결과 테이블을 먼저 출력

        # 분류 결과를 새로운 테이블로 명확하게 출력
        if self.cls_preds and self.cls_targets:
            LOGGER.info('')  # 시각적 구분을 위한 빈 줄 추가
            
            # 분류 테이블 헤더 생성 및 출력
            header_format = '%22s' + '%11s' * 3
            LOGGER.info(header_format % ('Class', 'Top-1 Acc', 'Top-5 Acc', 'F1-Score'))
            
            # 일관성을 위해 모든 값을 self.stats에서 가져옵니다.
            top1_acc = self.stats.get('top1_acc', 0.0)
            top5_acc = self.stats.get('top5_acc', 0.0)
            f1_score = self.stats.get('f1_score', 0.0)
            values_format = '%22s' + '%11.3g' * 3
            LOGGER.info(values_format % ('all', top1_acc, top5_acc, f1_score))
            
            if self.args.plots:
                # 분류 클래스 이름이 있는지 확인하고 전달
                cls_names = getattr(self, 'names_cls', {})
                self.cls_confusion_matrix.plot(save_dir=self.save_dir, names=cls_names, on_plot=self.on_plot)

    @property
    def metric_keys(self):
        """Return the metric keys used for plotting."""
        keys = super().metric_keys
        keys.extend(['metrics/top1_acc', 'metrics/top5_acc', 'metrics/f1_score'])
        return keys

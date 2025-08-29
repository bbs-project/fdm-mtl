import torch
import time
import argparse
import cv2
import numpy as np

from ultralytics.utils.torch_utils import select_device, get_flops
from mtl.multi_task_model import MultiTaskModel

def preprocess_image(image_path, size=(640, 640)):
    """
    실제 이미지를 읽고 모델 입력에 맞게 전처리합니다.
    (BGR -> RGB, 리사이즈, 정규화, 텐서 변환)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    
    # YOLOv8의 표준 전처리 방식 (Letterbox)을 간단히 모방
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    
    # HWC to CHW, (0, 255) to (0.0, 1.0)
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float()
    img /= 255.0
    
    # 배치 차원 추가 (1, 3, 640, 640)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
        
    return img

def main(args):
    """
    메인 실행 함수
    """
    # --- 설정 ---
    IMAGE_PATH = args.image
    MODEL_YAML_PATH = args.model
    INPUT_SIZE = (args.imgsz, args.imgsz)
    # 3번 GPU를 명시적으로 사용
    DEVICE = torch.device('cuda:3')

    print(f"실행 설정:")
    print(f"  - 모델 YAML: {MODEL_YAML_PATH}")
    print(f"  - 입력 이미지: {IMAGE_PATH}")
    print(f"  - 입력 크기: {INPUT_SIZE}")
    print(f"  - 사용 디바이스: {DEVICE}\n")

    try:
        # 1. 모델 로드
        print("1. 모델 로딩 중...")
        model = MultiTaskModel(cfg=MODEL_YAML_PATH, verbose=False).to(DEVICE)
        model.eval()
        print("   ...완료")

        # 2. 이미지 전처리
        print("2. 입력 이미지 전처리 중...")
        input_tensor = preprocess_image(IMAGE_PATH, size=INPUT_SIZE).to(DEVICE)
        print("   ...완료")

        # 3. 모델 크기 (파라미터, 연산량) 측정
        print("3. 모델 크기 및 연산량 측정 중...")
        
        # 파라미터 직접 계산
        params = sum(p.numel() for p in model.parameters())
        
        # GFLOPs는 get_flops 유틸리티 사용
        flops = get_flops(model, args.imgsz)
        
        print(f"  - 파라미터 (Parameters): {params / 1e6:.2f} M")
        print(f"  - 연산량 (GFLOPs): {flops:.2f} G")
        print("   ...완료")

        # 4. 추론 시간 (Latency) 측정
        print("4. 추론 시간 측정 중...")
        warmup_iterations = 10
        inference_iterations = 50

        # 워밍업
        with torch.no_grad():
            for _ in range(warmup_iterations):
                model(input_tensor)
        
        # 실제 측정
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(inference_iterations):
                model(input_tensor)
        torch.cuda.synchronize()
        end_time = time.time()

        avg_latency_ms = ((end_time - start_time) / inference_iterations) * 1000
        print(f"  - 평균 추론 시간 (Latency): {avg_latency_ms:.3f} ms")
        print("   ...완료")

    except Exception as e:
        print(f"\n오류 발생: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO Multi-Task 모델의 크기와 추론 시간을 측정합니다.")
    parser.add_argument('--model', type=str, required=True, help='모델 구조가 정의된 YAML 파일의 경로')
    parser.add_argument('--image', type=str, default='result/F04_U01_O0902_D2022-08-05_L360_W0374_S2_R01_B01_I00000613.jpg', help='추론에 사용할 이미지 파일의 경로')
    parser.add_argument('--imgsz', type=int, default=640, help='모델에 입력될 이미지의 크기 (정사각형)')
    
    args = parser.parse_args()
    main(args)

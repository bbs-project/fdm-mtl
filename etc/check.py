import sys
import torch
import subprocess

print("--- 1. 시스템 및 파이썬 정보 ---")
print(f"Python Version: {sys.version}")
print("-" * 30)

print("\n--- 2. NVIDIA GPU 및 CUDA 드라이버 정보 (nvidia-smi) ---")
try:
    # nvidia-smi 명령어 실행
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
    print(result.stdout)
except (FileNotFoundError, subprocess.CalledProcessError) as e:
    print("NVIDIA-SMI를 실행할 수 없습니다. GPU 드라이버가 설치되지 않았거나 경로 문제가 있을 수 있습니다.")
    print(f"에러: {e}")
print("-" * 30)


print("\n--- 3. PyTorch 정보 ---")
print(f"PyTorch Version: {torch.__version__}")
# PyTorch가 어떤 버전의 CUDA를 사용하여 빌드되었는지 확인
print(f"PyTorch built with CUDA Version: {torch.version.cuda}")
print("-" * 30)


print("\n--- 4. PyTorch의 GPU 사용 가능 여부 ---")
# PyTorch가 현재 시스템의 GPU를 사용할 수 있는지 확인
is_available = torch.cuda.is_available()
print(f"GPU Available: {is_available}")

if is_available:
    # 사용 가능한 GPU 개수
    print(f"Available GPUs: {torch.cuda.device_count()}")
    # 현재 사용 중인 GPU의 이름
    print(f"Current GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("PyTorch가 GPU를 찾을 수 없습니다. PyTorch를 GPU 버전으로 설치했는지 확인하세요.")

print("-" * 30)
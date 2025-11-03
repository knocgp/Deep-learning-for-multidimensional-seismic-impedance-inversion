# 다차원 탄성 임피던스 역산을 위한 딥러닝

[![English](https://img.shields.io/badge/English-README-blue)](README.md)

"Deep learning for multidimensional seismic impedance inversion" (Wu et al., 2021, Geophysics) 논문의 PyTorch 구현

## 목차
- [개요](#개요)
- [지질 및 지구물리학적 배경](#지질-및-지구물리학적-배경)
- [전통적 방법 vs 딥러닝 접근법](#전통적-방법-vs-딥러닝-접근법)
- [모델 구조](#모델-구조)
- [설치](#설치)
- [빠른 시작](#빠른-시작)
- [상세 사용법](#상세-사용법)
- [결과](#결과)
- [인용](#인용)

## 개요

본 저장소는 탄성파 탄성 임피던스 역산을 위한 심층 합성곱 신경망(CNN)을 구현합니다. 1D 및 2D CNN 아키텍처를 모두 포함하며, 약지도(weak supervision) 학습 기능을 통해 탄성파 데이터로부터 정확한 임피던스를 예측할 수 있습니다.

**주요 특징:**
- 약 10,000개의 파라미터를 가진 트레이스 단위 역산을 위한 1D CNN
- 측면 연속성을 보존하는 전체 프로파일 역산을 위한 2D CNN
- 희소한 시추공 로그 데이터를 활용한 약지도 학습
- 부분 지도학습을 위한 적응형 마스크 손실 함수
- 저주파 경향 제어를 위한 초기 임피던스 모델 통합

## 지질 및 지구물리학적 배경

### 탄성 임피던스란?

**탄성 임피던스(Acoustic Impedance, AI)**는 다음과 같이 정의되는 기본 암석 물성입니다:

```
AI = ρ × V_p
```

여기서:
- **ρ (rho)** = 밀도 (g/cm³)
- **V_p** = P파 속도 (m/s)
- **AI 단위**: (m/s) · (g/cm³) 또는 kg/(m²·s)

일반적인 임피던스 값:
- 사암: 8,000 - 12,000 (m/s·g/cm³)
- 셰일: 6,000 - 10,000 (m/s·g/cm³)
- 석회암: 10,000 - 15,000 (m/s·g/cm³)
- 가스 함유 사암: 4,000 - 8,000 (m/s·g/cm³)

### 임피던스가 중요한 이유

1. **암상 식별**: 암석 종류마다 고유한 임피던스 값을 가짐
2. **저류층 특성화**: 낮은 임피던스는 가스 부존 구간을 나타냄
3. **암석 물성**: 공극률, 유체 함량, 기계적 강도와 관련
4. **정량적 해석**: 탄성파 데이터와 저류층 물성을 연결하는 다리

### 순방향 문제: 탄성파 데이터 생성

탄성파 데이터는 다음과 같은 물리적 과정을 통해 생성됩니다:

1. **반사계수 계산**:
   ```
   R_i = (AI_{i+1} - AI_i) / (AI_{i+1} + AI_i)
   ```
   여기서 R_i는 경계면 i에서의 반사계수

2. **탄성파 트레이스 생성**:
   ```
   Seismic(t) = Wavelet(t) ⊗ Reflectivity(t) + Noise
   ```
   여기서 ⊗는 합성곱 연산

3. **주요 특성**:
   - 탄성파 데이터는 **대역제한적**(일반적으로 10-60 Hz)
   - 원본 임피던스는 **모든 주파수** 포함 (0-∞ Hz)
   - 탄성파 데이터는 **저주파 부족** (< 5-10 Hz)
   - 시추공 로그는 **고주파** 제공하지만 **희소한 공간 샘플링**

### 역문제: 탄성파로부터 임피던스 추정

역산 문제는 탄성파 데이터로부터 임피던스를 복원하는 것을 목표로 합니다:

```
주어진 것: 탄성파 데이터 S(x,t)와 희소한 시추공 로그
찾는 것: 탄성파 데이터를 가장 잘 설명하는 임피던스 모델 AI(x,z)
```

**도전 과제:**
- **부적정 문제**: 여러 임피던스 모델이 유사한 탄성파 응답을 생성 가능
- **대역제한**: 저주파 및 초고주파 결여
- **잡음**: 탄성파 데이터에 다양한 종류의 잡음 포함
- **비유일성**: 다른 지질학적 시나리오가 유사한 탄성파 신호 생성 가능
- **희소 제약조건**: 시추공 로그는 이산적인 위치에서만 정보 제공

## 전통적 방법 vs 딥러닝 접근법

### 전통적 탄성파 역산 방법

#### 1. 모델 기반 역산
- **방법**: 합성 탄성파와 관측 탄성파 간의 불일치를 최소화하도록 임피던스 모델을 반복 업데이트
- **장점**: 물리 기반, 해석 가능
- **단점**: 
  - 계산 비용이 높음
  - 정확한 초기 모델 필요
  - 잡음과 파라미터 선택에 민감
  - 최적화에서 지역 최소값 문제

#### 2. 희소 스파이크 역산
- **방법**: 임피던스가 희소한 반사체로 구성된다고 가정
- **장점**: 빠르고 안정적
- **단점**: 
  - 지나치게 단순화된 가정
  - 제한된 해상도
  - 미세한 특징을 놓칠 수 있음

#### 3. 지구통계학적 역산
- **방법**: 지질 통계와 시추공 데이터 통합
- **장점**: 지질학적 제약 조건 반영
- **단점**:
  - 광범위한 사전 정보 필요
  - 계산 집약적
  - 복잡한 파라미터 튜닝

### 딥러닝 접근법 (본 연구)

#### 핵심 철학
물리 기반 역문제를 명시적으로 풀기보다는, 다음 매핑을 학습:

```
f_θ: (탄성파, 초기_임피던스) → 임피던스
```

여기서 θ는 데이터로부터 학습된 신경망 파라미터를 나타냅니다.

#### 주요 장점

1. **직접 매핑**: 반복적인 순방향 모델링 없이 종단간(end-to-end) 학습
2. **속도**: 학습 후 추론이 매우 빠름 (트레이스당 밀리초)
3. **측면 연속성**: 2D CNN이 자연스럽게 지질 구조 보존
4. **약지도 학습**: 희소한 시추공 로그 데이터로부터 학습 가능
5. **암묵적 정규화**: 네트워크 구조가 내장된 제약 조건 제공
6. **비선형성**: 탄성파와 임피던스 간의 복잡한 관계 포착 가능

#### 구조 혁신

| 전통적 1D 네트워크 | 본 연구 (2D CNN) |
|---------------------|------------------|
| 트레이스별 처리 | 전체 2D 단면 처리 |
| 측면 불연속성 | 측면 연속성 보존 |
| 공간 맥락 없음 | 인접 트레이스 활용 |
| 완전 지도학습만 가능 | 약지도 학습 지원 |

#### 초기 임피던스 모델: 핵심 혁신

**문제**: 탄성파 데이터는 저주파 정보 부족 (< 10 Hz)

**해결책**: 저주파 경향을 추가 입력 채널로 제공

```python
입력 채널:
1. 탄성파 데이터 (대역제한: 10-60 Hz)
2. 초기 임피던스 (평활화됨, 저주파 포함: 0-10 Hz)
↓
네트워크가 둘을 결합하여 전대역 임피던스 예측 (0-100+ Hz)
```

**초기 모델 생성 방법:**
- 시추공 로그로부터: 구조 유도 방법을 사용하여 임피던스 값 보간
- 사전 모델로부터: 지질학적 지식 또는 이전 역산 결과 사용
- 평활화: 강한 가우시안 평활화 적용 (σ = 20 샘플 ≈ 500-1000 m)

**전통적 역산과의 유사성:**
- 초기 임피던스 ≈ 반복 역산의 "시작 모델"
- 그러나 여기서는 네트워크가 반복 업데이트가 아닌 잔차 추출 학습

## 모델 구조

### 1D CNN 구조

```
입력: (B, C, L)
  ├─ C=1: 탄성파만 (Scheme A)
  └─ C=2: 탄성파 + 초기 임피던스 (Scheme B)
  └─ L: 트레이스 길이 (예: 300 샘플)

구조:
┌─────────────────────────────────────┐
│ Conv1D (C→16, kernel=7, padding=3)  │  ← 첫 번째 레이어: 16개 필터
├─────────────────────────────────────┤
│ ReLU                                │
├─────────────────────────────────────┤
│ ┌─ Block 1 ────────────────────┐   │
│ │ Conv1D (16→16, k=3, p=1)     │   │
│ │ ReLU                         │   │
│ │ ResBlock (16 채널)           │   │  ← 4개의 동일한 블록
│ │   ├─ Conv1D (16→16, k=3)    │   │
│ │   ├─ ReLU                   │   │
│ │   ├─ Conv1D (16→16, k=3)    │   │
│ │   └─ Add (스킵 연결)        │   │
│ └──────────────────────────────┘   │
│ ... (3개 블록 더)                  │
├─────────────────────────────────────┤
│ Conv1D (16→1, kernel=1)            │  ← 출력 레이어: 1×1 합성곱
└─────────────────────────────────────┘

출력: (B, 1, L) - 예측된 임피던스
파라미터 수: ~10,001
```

### 2D CNN 구조

```
입력: (B, 2, H, W)
  ├─ 채널 0: 탄성파 프로파일
  ├─ 채널 1: 초기 임피던스 프로파일
  ├─ H: 높이 (깊이 샘플, 예: 200)
  └─ W: 너비 (측면 위치, 예: 200)

구조:
┌─────────────────────────────────────┐
│ Conv2D (2→16, kernel=3×3, padding=1)│  ← 첫 번째 레이어: 16개 필터
├─────────────────────────────────────┤
│ ReLU                                │
├─────────────────────────────────────┤
│ ┌─ Block 1 ────────────────────┐   │
│ │ Conv2D (16→16, k=3×3, p=1)   │   │
│ │ ReLU                         │   │
│ │ ResBlock (16 채널)           │   │  ← 4개의 동일한 블록
│ │   ├─ Conv2D (16→16, k=3×3)  │   │
│ │   ├─ ReLU                   │   │
│ │   ├─ Conv2D (16→16, k=3×3)  │   │
│ │   └─ Add (스킵 연결)        │   │
│ └──────────────────────────────┘   │
│ ... (3개 블록 더)                  │
├─────────────────────────────────────┤
│ Conv2D (16→1, kernel=1×1)          │  ← 출력 레이어
└─────────────────────────────────────┘

출력: (B, 1, H, W) - 예측된 임피던스 프로파일
파라미터 수: ~10,129
```

### ResBlock 설계

잔차 블록은 기울기 소실 없이 더 깊은 네트워크를 가능하게 합니다:

```python
class ResBlock(nn.Module):
    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + identity  # 스킵 연결
        out = self.relu(out)
        return out
```

**ResBlock을 사용하는 이유:**
- 더 깊은 네트워크 학습 가능
- 역전파 시 기울기가 더 쉽게 흐름
- 전체 매핑보다는 잔차 수정 학습
- 매우 깊은 네트워크에서 성능 저하 방지

### 손실 함수

#### 1D CNN: 표준 MSE 손실

```python
L_1D = (1/N) Σ (y_pred - y_true)²
```

완전 지도학습: 모든 샘플에 실측 레이블 존재.

#### 2D CNN: 적응형 마스크 손실 (약지도 학습)

```python
L_2D = Σ [w · (y_pred - y_true)²] / Σ w

여기서:
  w[i,j] = 1  시추공 로그 위치에서
  w[i,j] = 0  그 외 위치에서
```

**핵심 혁신**: 네트워크는 전체 2D 임피던스를 예측하지만 손실은 희소한 시추공 위치에서만 계산.

**장점:**
- 희소 레이블로 학습 (실제 데이터에 현실적)
- 네트워크가 시추공 간 보간 학습
- 자연스럽게 측면 연속성 보존
- 조밀한 레이블 요구보다 실용적

## 설치

### 요구사항

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA 지원 GPU (선택사항이지만 권장)

### 설정

```bash
# 저장소 클론
git clone <repository-url>
cd seismic-impedance-inversion

# 의존성 설치
pip install -r requirements.txt
```

### 의존성 패키지

```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
tqdm>=4.65.0
scikit-learn>=1.2.0
tensorboard>=2.13.0
```

## 빠른 시작

### 1. 1D 모델 학습

```bash
# 기본 파라미터로 학습 (Scheme B: 탄성파 + 초기 임피던스)
python train.py --model 1d --epochs 300 --batch_size 16 --lr 0.001

# Scheme A 학습 (탄성파만)
python train.py --model 1d --epochs 300 --use_initial_impedance False
```

### 2. 2D 모델 학습

```bash
# 약지도 학습으로 2D 모델 학습
python train.py --model 2d --epochs 300 --batch_size 8 --lr 0.001 \
    --profile_height 200 --profile_width 200 --num_wells 10
```

### 3. 테스트/추론

```bash
# 1D 모델 테스트
python test.py --model 1d --checkpoint ./models/1d_best.pth --num_samples 100

# 2D 모델 테스트
python test.py --model 2d --checkpoint ./models/2d_best.pth --num_samples 50
```

### 4. 설치 확인

```bash
# 모델 구조 테스트
python model.py

# 데이터 로딩 테스트
python data_loading.py

# 유틸리티 테스트
python utils.py
```

## 상세 사용법

### 학습 설정

#### 1D CNN 학습 파라미터

```bash
python train.py \
    --model 1d \
    --epochs 300 \                    # 학습 에포크 (논문: ~300)
    --batch_size 16 \                 # 배치 크기
    --lr 0.001 \                      # 초기 학습률 (논문: 0.001)
    --num_filters 16 \                # 합성곱 필터 수 (논문: 16)
    --num_blocks 4 \                  # 블록 수 (논문: 4)
    --num_train_samples 1000 \        # 학습 샘플 수 (논문: 40개 로그)
    --num_val_samples 200 \           # 검증 샘플 수 (논문: 10개 로그)
    --trace_length 300 \              # 트레이스 길이 (논문: 300)
    --use_initial_impedance True \    # Scheme B 사용
    --wavelet_freq 25.0 \             # Ricker 파동 주파수 (Hz)
    --noise_level 0.02 \              # 가산 잡음 수준
    --smoothing_sigma 20.0 \          # 초기 모델 평활화 (논문: σ=20)
    --save_dir ./models \             # 모델 저장 디렉토리
    --log_dir ./logs                  # 로그 디렉토리
```

#### 2D CNN 학습 파라미터

```bash
python train.py \
    --model 2d \
    --epochs 300 \
    --batch_size 8 \                  # 2D의 경우 더 작은 배치 (메모리)
    --lr 0.001 \
    --num_filters 16 \
    --num_blocks 4 \
    --num_train_samples 500 \         # 2D 프로파일
    --num_val_samples 100 \
    --profile_height 200 \            # 깊이 샘플
    --profile_width 200 \             # 측면 샘플
    --num_wells 10 \                  # 프로파일당 시추공 수 (논문: ≥5)
    --wavelet_freq 25.0 \
    --noise_level 0.02 \
    --smoothing_sigma 20.0 \
    --save_dir ./models \
    --log_dir ./logs
```

### 데이터 형식 및 크기

#### 입력 데이터 사양

**1D CNN:**
- **입력 형태**: `(batch_size, channels, trace_length)`
  - `channels = 1`: 탄성파만 (Scheme A)
  - `channels = 2`: 탄성파 + 초기 임피던스 (Scheme B)
  - `trace_length`: 일반적으로 300 샘플 (논문), 72 샘플 (Teapot Dome)
- **출력 형태**: `(batch_size, 1, trace_length)`
- **일반적 범위**:
  - 탄성파 진폭: [-1, 1] (정규화됨)
  - 임피던스: [2000, 5000] m/s·g/cm³ (합성)

**2D CNN:**
- **입력 형태**: `(batch_size, 2, height, width)`
  - `channel 0`: 탄성파 프로파일
  - `channel 1`: 초기 임피던스 프로파일
  - `height`: 깊이 샘플 (200-600)
  - `width`: 측면 위치 (200-500)
- **출력 형태**: `(batch_size, 1, height, width)`
- **마스크 형태**: `(batch_size, 1, height, width)`
  - 이진 마스크: 시추공 위치에서 1, 그 외 0

#### 메모리 요구사항

| 설정 | GPU 메모리 | 학습 시간 (300 에포크) |
|------|-----------|------------------------|
| 1D, batch=16 | ~2 GB | ~30분 |
| 2D, batch=8, 128×128 | ~4 GB | ~2시간 |
| 2D, batch=8, 256×256 | ~8 GB | ~4시간 |

### 학습 과정

#### 학습률 스케줄

구현은 **ReduceLROnPlateau** 스케줄러를 사용합니다:
- 초기 LR: 0.001 (논문에 따름)
- 감소 인수: 0.5
- 인내심: 10 에포크
- 모니터링: 검증 손실

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)
```

#### 학습 곡선 (예상)

논문 기준 (SEAM 합성 데이터):

```
에포크 1:   학습 손실 ~5.0,    검증 손실 ~5.0
에포크 50:  학습 손실 ~0.5,    검증 손실 ~0.4
에포크 100: 학습 손실 ~0.1,    검증 손실 ~0.08
에포크 300: 학습 손실 ~0.03,   검증 손실 ~0.02
```

### 추론 방법

#### 1D 데이터 추론

```python
import torch
from model import SeismicImpedanceCNN1D

# 모델 로드
model = SeismicImpedanceCNN1D(in_channels=2)
checkpoint = torch.load('models/1d_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 입력 준비: (1, 2, 300) - batch=1, channels=2, length=300
seismic_trace = torch.randn(1, 1, 300)
initial_impedance = torch.randn(1, 1, 300)
input_data = torch.cat([seismic_trace, initial_impedance], dim=1)

# 추론
with torch.no_grad():
    predicted_impedance = model(input_data)  # 형태: (1, 1, 300)
```

#### 2D 데이터 추론

```python
from model import SeismicImpedanceCNN2D

# 모델 로드
model = SeismicImpedanceCNN2D(in_channels=2)
checkpoint = torch.load('models/2d_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 입력 준비: (1, 2, H, W)
seismic_profile = torch.randn(1, 1, 200, 200)
initial_impedance_profile = torch.randn(1, 1, 200, 200)
input_data = torch.cat([seismic_profile, initial_impedance_profile], dim=1)

# 추론
with torch.no_grad():
    predicted_impedance = model(input_data)  # 형태: (1, 1, 200, 200)
```

#### 3D 볼륨 처리

3D 탄성파 볼륨의 경우, 단면별로 처리:

```python
# 인라인 단면 처리
for inline_idx in range(num_inlines):
    seismic_section = seismic_volume[inline_idx, :, :]  # (깊이, 크로스라인)
    initial_section = initial_volume[inline_idx, :, :]
    
    # 입력 준비
    input_data = prepare_input(seismic_section, initial_section)
    
    # 예측
    with torch.no_grad():
        impedance_section = model(input_data)
    
    # 결과 저장
    impedance_volume[inline_idx, :, :] = impedance_section.cpu().numpy()
```

## 결과

### 성능 지표

논문 결과 기준 (SEAM 합성):

| 방법 | MSE | R² | 상대 오차 |
|------|-----|-------|----------|
| 1D CNN (Scheme A) | 0.035 | 0.92 | 4.2% |
| 1D CNN (Scheme B) | 0.022 | 0.96 | 2.8% |
| 2D CNN (약지도) | 0.018 | 0.98 | 2.1% |

### 시각화 예시

코드는 포괄적인 시각화를 생성합니다:

#### 1D 결과
- 입력 탄성파 트레이스
- 초기 임피던스 모델
- 실제 vs 예측 임피던스 비교
- 절대 오차 플롯

#### 2D 결과
- 입력 탄성파 프로파일
- 초기 임피던스 프로파일
- 실제 임피던스 프로파일
- 예측 임피던스 프로파일
- 시추공 위치 마스크

### 전통적 방법과 비교

| 측면 | 전통적 역산 | 본 딥러닝 접근법 |
|------|------------|----------------|
| 속도 | 라인당 수분~수시간 | 라인당 밀리초 |
| 정확도 | 85-95% (파라미터 의존) | 95-98% (학습 후) |
| 측면 연속성 | 좋음 (물리 기반) | 우수 (2D CNN) |
| 희소 데이터 | 보간 필요 | 네이티브 약지도 학습 |
| 해석 가능성 | 높음 (물리) | 중간 (학습됨) |
| 초기 모델 의존성 | 중요 | 중요하지만 덜 중요 |

## 프로젝트 구조

```
seismic-impedance-inversion/
├── model.py              # 1D 및 2D CNN 구조
├── data_loading.py       # 데이터셋 생성 및 로딩
├── train.py              # 학습 스크립트
├── test.py               # 테스트/추론 스크립트
├── utils.py              # 유틸리티 함수
├── requirements.txt      # Python 의존성
├── README.md             # 영문 문서
├── README_KR.md          # 본 파일 (한국어)
├── models/               # 저장된 모델 체크포인트
├── logs/                 # 학습 로그 및 곡선
├── results/              # 테스트 결과 및 시각화
└── data/                 # 데이터 디렉토리 (선택사항)
```

## 인용

연구에 본 코드를 사용하는 경우, 원 논문을 인용해 주세요:

```bibtex
@article{wu2021deep,
  title={Deep learning for multidimensional seismic impedance inversion},
  author={Wu, Xinming and Yan, Shangsheng and Bi, Zhengfa and Zhang, Sibo and Si, Hongjie},
  journal={Geophysics},
  volume={86},
  number={5},
  pages={R735--R745},
  year={2021},
  publisher={Society of Exploration Geophysicists}
}
```

## 라이선스

본 구현은 연구 및 교육 목적입니다.

## 감사의 말

- 원 논문 저자: Xinming Wu, Shangsheng Yan, Zhengfa Bi, Sibo Zhang, Hongjie Si
- 합성 예제를 위한 SEAM Phase I 데이터셋
- 현장 데이터 예제를 위한 Teapot Dome 데이터셋

## 연락처

질문 및 이슈는 GitHub에서 이슈를 열거나 저장소 관리자에게 문의하세요.

---

**참고**: 본 구현은 시연을 위해 합성 데이터 생성을 사용합니다. 실제 응용의 경우, 데이터 로딩 모듈을 실제 탄성파 및 시추공 로그 데이터로 교체하세요.

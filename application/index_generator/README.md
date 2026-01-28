# TSUI 지수 산출 패키지

트럼프 지수(TSUI, Trump Statement Uncertainty Index) 산출을 위한 모듈화된 Python 패키지입니다.

## 설치

```bash
# 저장소 클론
git clone <repository-url>
cd index_generator

# 의존성 설치
pip install -r requirements.txt
```

## 요구사항

- Python 3.8+
- numpy >= 1.24.0
- pandas >= 2.0.0

**참고**: `tools/track1_sentiment_score.py`와 `tools/track2_sector_classification.py`를 실행하려면 추가로 PyTorch와 transformers가 필요합니다. 하지만 지수 계산 자체는 numpy와 pandas만으로 충분합니다.

## 패키지 구조

```
index_generator/
├── __init__.py              # 패키지 초기화 및 export (run_analysis)
├── main.py                  # 메인 실행 스크립트
├── pipeline.py              # 전체 공정 오케스트레이션
├── utils.py                 # 공통 유틸리티 함수
├── run_tsui.py              # 실행 예제 스크립트
├── data/                    # [Data Layer]
│   ├── __init__.py
│   ├── constants.py         # 상수 정의
│   └── data_loader.py       # 데이터 로딩 및 캐싱
├── engine/                  # [Engine Layer]
│   ├── __init__.py
│   ├── math_engines.py      # 순수 수학 계산 함수
│   └── index_calculator.py  # TSUI 지수 계산 함수
├── logic/                   # [Logic Layer]
│   ├── __init__.py
│   └── logic_helpers.py     # 데이터 가공 및 그룹화 함수
├── tools/                   # 별도 실행 스크립트
│   ├── track1_sentiment_score.py      # Track 1 감성 분석
│   └── track2_sector_classification.py # Track 2 섹터 분류
├── data/                    # 데이터 폴더
└── docs/                    # 문서 폴더
```

## 모듈 설명

### `data/` 패키지 (Data Layer)

#### `data/constants.py`
- 트럼프 재임 기간 정의
- 잔상 상수 (α) - 강도별
- 시간 가중치 설정
- 키워드 가중치 및 키워드 사전 (1기/2기 분리)
- Track 2 섹터명 매핑
- TSUI 지수 계산 상수 (Vol_Proxy, 로그 스케일링, EMA 등)

#### `data/data_loader.py`
- `load_track1_data()`: Track 1 데이터 로드
- `load_track2_data()`: Track 2 데이터 로드
- `load_etf_metadata()`: ETF 메타데이터 로드
- `load_all_etf_data()`: 모든 ETF 데이터 로드
- `load_dxy_data()`: DXY 데이터 로드
- `get_volatility_data()`: Volatility 데이터 조회
- `extract_sector_from_label_short()`: 섹터명 추출
- `extract_intensity_from_label_short()`: 강도 레벨 추출
- `get_daily_sector()`: 일별 섹터 정보 추출
- `initialize_data()`: 모든 데이터 로드 및 캐싱
- `get_track1_data()`, `get_track2_data()` 등: 캐시된 데이터 조회

### `utils.py` (루트)
- `parse_datetime()`: 날짜 문자열 파싱
- `calculate_time_weight()`: 시간 가중치 계산
- `calculate_keyword_weight()`: 키워드 가중치 계산
- `calculate_parkinson_volatility()`: Parkinson Volatility 계산
- `get_residue_alpha()`: 잔상 상수 반환

### `engine/` 패키지 (Engine Layer)

#### `engine/math_engines.py`
- `calculate_cumulative_energy()`: 누적 에너지 계산
- `calculate_log_scaled_values()`: 로그 스케일링 적용
- `normalize_to_baseline()`: Baseline 기준 정규화
- `apply_ema_smoothing()`: EMA 적용
- `renormalize_after_ema()`: EMA 후 재정규화

#### `engine/index_calculator.py`
- `calculate_tsui_index()`: 최종 TSUI 지수 계산

### `logic/` 패키지 (Logic Layer)

#### `logic/logic_helpers.py`
- `calculate_daily_raw_energy()`: 일일 원천 에너지 계산
- `calculate_daily_energies()`: 전체 기간 일일 에너지 계산
- `group_posts_by_date()`: 일별 포스트 그룹화
- `generate_all_dates()`: 전체 기간 날짜 생성
- `extract_daily_sectors()`: 일별 섹터 정보 추출
- `calculate_all_dates_vol_proxy()`: 모든 날짜 Vol_Proxy 계산
- `determine_baseline_scaled()`: Dynamic Baseline 설정

### `pipeline.py` (루트)
- `run_analysis()`: 전체 지수 산출 프로세스 조율 (구 `process_index_calculation`)

### `tools/` 패키지

별도로 실행되는 전처리 스크립트들:
- `track1_sentiment_score.py`: Track 1 감성 분석 (FinBERT 사용)
- `track2_sector_classification.py`: Track 2 섹터 분류 (BART-large-MNLI 사용) - **참고용**

**참고**: Track 2 전처리는 별도로 수행되며, 이 저장소에는 포함되지 않습니다. 지수 계산에는 이미 전처리된 Track 2 데이터가 필요합니다.

## 사용법

### 스크립트로 직접 실행

```bash
# 방법 1: 모듈로 실행
cd application
python -m index_generator.main

# 방법 2: 직접 실행
cd application/index_generator
python main.py
```

### 모듈로 import

```python
from index_generator import run_analysis
from pathlib import Path

run_analysis(
    track1_csv=Path("data/track_01/trump_posts_term1_2017_2021_with_sentiment.csv"),
    track2_csv=Path("data/track_02/trump_posts_36categori_final.csv"),
    etf_dir=Path("data/etf_sector"),
    output_csv=Path("data/tsui_index_term1_2017-2021.csv"),
    term="term1"
)
```

### 패키지에서 함수 import

```python
from index_generator import run_analysis
from index_generator.logic.logic_helpers import calculate_daily_raw_energy
from index_generator.data.data_loader import get_track1_data, initialize_data
from index_generator.utils import parse_datetime
```

## 데이터 구조

지수 계산을 위해서는 다음 데이터가 필요합니다:

```
data/
├── track_01/                    # Track 1 데이터 (감성 분석 결과)
│   ├── trump_posts_term1_2017_2021_with_sentiment.csv
│   └── trump_posts_term2_2025_with_sentiment.csv
├── track_02/                    # Track 2 데이터 (섹터 분류 결과)
│   └── trump_posts_36categori_final.csv
├── etf_sector/                  # ETF 가격 데이터
│   ├── etf_metadata.json
│   ├── etf_XLY_2017-2021.csv
│   └── ...
└── DXY.csv                      # 달러 인덱스 데이터
```

**참고**: 실제 데이터 파일은 저장소에 포함되지 않습니다. 샘플 데이터 구조만 참고하세요.

## 설치 (패키지로 설치)

```bash
# 개발 모드로 설치
pip install -e .

# 또는 일반 설치
pip install .
```

## 주요 개선사항

1. **계층적 구조**: 논리적 계층으로 모듈 분리
   - **Data Layer** (`data/`): 상수 정의 및 데이터 로딩/캐싱
   - **Engine Layer** (`engine/`): 순수 수학 계산 및 지수 산출
   - **Logic Layer** (`logic/`): 데이터 가공 및 그룹화
2. **책임 분리 (SRP 준수)**: 
   - `data/constants.py`: 모든 상수 정의
   - `data/data_loader.py`: 데이터 로딩 및 전역 캐싱
   - `utils.py`: 공통 유틸리티 함수
   - `logic/logic_helpers.py`: 데이터 가공 및 그룹화
   - `engine/math_engines.py`: 순수 수학 계산
   - `engine/index_calculator.py`: TSUI 지수 계산
   - `pipeline.py`: 전체 프로세스 오케스트레이션
3. **경로 의존성 제거**: 함수들이 전역 캐시에서 데이터를 가져와 경로 인자 불필요
4. **재사용성**: 개별 함수를 독립적으로 사용 가능
5. **유지보수성**: 코드 구조가 명확하여 수정이 용이
6. **패키징**: `__init__.py`를 통한 깔끔한 패키지 구조 및 `setup.py` 제공

## 라이선스

이 프로젝트는 개인 프로젝트입니다. 사용 시 출처를 명시해주세요.

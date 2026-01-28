"""
트럼프 소셜미디어 포스트 감성점수 분석 스크립트 (Track 1)

이 스크립트는 전처리된 트럼프 포스트 데이터에 대해 FinBERT 모델을 사용하여
감성분석을 수행하고, 기존 CSV 파일에 감성점수 컬럼을 추가합니다.

사용법:
    python track1_sentiment_score.py [입력파일경로]
    
    예시:
    python track1_sentiment_score.py data/trump_posts_preprocessed_term1_2017_2021.csv
    
    입력파일경로를 지정하지 않으면 기본값 사용:
    data/trump_posts_preprocessed_term2_2025.csv

주요 기능:
1. 전처리된 CSV 파일 읽기
2. FinBERT 모델로 각 포스트의 감성 분석
3. 감성점수 정규화 (-1.0 ~ +1.0)
4. 기존 CSV 파일에 결과 컬럼 추가 (원본 파일 백업 자동 생성)

출력 컬럼 설명:
- sentiment_label: 감성 레이블
  * 'positive': 긍정적 감성
  * 'negative': 부정적 감성
  * 'neutral': 중립적 감성

- sentiment_confidence: 모델의 예측 신뢰도 (0.0 ~ 1.0)
  * FinBERT 모델이 예측한 레이블에 대한 확률
  * 1.0에 가까울수록 모델이 해당 예측에 확신이 있음
  * 예: 0.9996 = 99.96% 확신

- sentiment_score: 정규화된 감성점수 (-1.0 ~ +1.0)
  * positive: 0.0 ~ +1.0 (긍정적일수록 높음)
  * negative: -1.0 ~ 0.0 (부정적일수록 낮음)
  * neutral: 0.0

주의사항:
- 모델 다운로드: 첫 실행 시 Hugging Face에서 모델을 자동 다운로드합니다.
- 메모리: GPU(MPS/CUDA) 사용 시 더 빠르게 처리됩니다.
- 백업: 원본 파일은 자동으로 타임스탬프가 포함된 백업 파일로 저장됩니다.
"""

import csv
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================================
# 상수 정의
# ============================================================================

# FinBERT 모델명
# Hugging Face에서 제공하는 금융 도메인 특화 감성분석 모델
# 첫 실행 시 자동으로 다운로드됩니다.
FINBERT_MODEL_NAME = "yiyanghkust/finbert-tone"

# 배치 처리 크기
# 한 번에 처리할 텍스트 개수 (메모리와 속도의 균형)
# GPU 메모리가 부족하면 이 값을 줄이세요 (예: 16, 8)
# CPU 사용 시에도 메모리에 따라 조정 가능
BATCH_SIZE = 32

# 최대 시퀀스 길이
# 모델이 처리할 수 있는 최대 토큰 수
# 이 길이를 초과하는 텍스트는 자동으로 잘립니다 (truncation)
MAX_LENGTH = 512


# ============================================================================
# 감성점수 계산 함수
# ============================================================================

def normalize_sentiment_score(label: str, score: float) -> float:
    """
    FinBERT-tone 출력을 -1.0 ~ +1.0 범위로 정규화
    
    FinBERT-tone 모델은 3가지 레이블을 출력하며, 각 레이블에 대한 확률을 제공합니다.
    이 함수는 레이블과 확률을 받아서 하나의 정규화된 점수로 변환합니다.
    
    변환 규칙:
    - positive + confidence(0.9) → +0.9 (긍정적, 높은 확신)
    - negative + confidence(0.9) → -0.9 (부정적, 높은 확신)
    - neutral + confidence(0.8) → 0.0 (중립, 확률과 무관)
    
    Args:
        label: 감성 레이블 ('positive', 'negative', 'neutral')
        score: 신뢰도 점수 (0.0 ~ 1.0), 모델이 예측한 레이블에 대한 확률
    
    Returns:
        float: 정규화된 감성점수
            - positive: 0.0 ~ +1.0 (긍정적일수록 높음)
            - negative: -1.0 ~ 0.0 (부정적일수록 낮음)
            - neutral: 0.0 (항상 0)
    
    Examples:
        >>> normalize_sentiment_score('positive', 0.95)
        0.95
        >>> normalize_sentiment_score('negative', 0.85)
        -0.85
        >>> normalize_sentiment_score('neutral', 0.90)
        0.0
    """
    if label == 'positive':
        return score  # 0.0 ~ +1.0
    elif label == 'negative':
        return -score  # -1.0 ~ 0.0
    else:  # neutral
        return 0.0


def analyze_sentiment_batch(
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    texts: List[str],
    device: torch.device
) -> List[Dict[str, Any]]:
    """
    텍스트 배치에 대해 감성분석 수행
    
    이 함수는 여러 텍스트를 한 번에 처리하여 효율성을 높입니다.
    각 텍스트에 대해 다음 정보를 반환합니다:
    - label: 예측된 감성 레이블 (positive/negative/neutral)
    - score: 예측 신뢰도 (0.0 ~ 1.0)
    - sentiment_score: 정규화된 감성점수 (-1.0 ~ +1.0)
    
    처리 과정:
    1. 텍스트를 토큰화 (최대 512 토큰)
    2. 모델에 입력하여 logits 출력
    3. Softmax로 확률 변환
    4. 가장 높은 확률의 레이블 선택
    5. 정규화된 점수 계산
    
    Args:
        tokenizer: FinBERT 토크나이저 (텍스트 → 토큰 변환)
        model: FinBERT 모델 (토큰 → 감성 예측)
        texts: 분석할 텍스트 리스트
        device: 계산 장치 (CPU/MPS/CUDA)
    
    Returns:
        List[Dict]: 각 텍스트의 감성분석 결과 리스트
            각 Dict는 다음 키를 포함:
            - 'label': str - 예측된 레이블
            - 'score': float - 신뢰도 (0.0 ~ 1.0)
            - 'sentiment_score': float - 정규화된 점수 (-1.0 ~ +1.0)
    """
    # 토큰화
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    ).to(device)
    
    # 모델 추론
    model.eval()
    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
    
    # 확률 변환: logits를 확률 분포로 변환 (합이 1.0이 되도록)
    probs = F.softmax(logits, dim=-1)
    
    # 레이블 매핑
    # FinBERT-tone 모델의 출력 인덱스 → 레이블 매핑
    # LABEL_0 = neutral, LABEL_1 = positive, LABEL_2 = negative
    label_map = {0: 'neutral', 1: 'positive', 2: 'negative'}
    
    results = []
    for prob in probs:
        # 가장 높은 확률의 레이블 선택 (argmax)
        predicted_idx = prob.argmax().item()
        predicted_label = label_map[predicted_idx]
        # 해당 레이블의 확률 = confidence
        predicted_score = prob[predicted_idx].item()
        
        # 정규화된 감성점수 계산 (positive → +, negative → -, neutral → 0)
        sentiment_score = normalize_sentiment_score(predicted_label, predicted_score)
        
        results.append({
            'label': predicted_label,           # 예측된 레이블
            'score': predicted_score,            # 신뢰도 (confidence)
            'sentiment_score': sentiment_score   # 정규화된 점수
        })
    
    return results


def process_sentiment_analysis(
    input_csv_path: Path,
    output_csv_path: Path,
    model_name: str = FINBERT_MODEL_NAME,
    batch_size: int = BATCH_SIZE
) -> None:
    """
    전처리된 CSV 파일에 대해 감성분석 수행하고 결과 컬럼 추가
    
    Args:
        input_csv_path: 입력 CSV 파일 경로
        output_csv_path: 출력 CSV 파일 경로 (기존 파일에 컬럼 추가)
        model_name: 사용할 모델명
        batch_size: 배치 처리 크기
    
    Raises:
        FileNotFoundError: 입력 파일이 존재하지 않을 때
        KeyError: CSV에 'text' 컬럼이 없을 때
    """
    # 입력 파일 존재 확인
    if not input_csv_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_csv_path}")
    
    print(f"FinBERT-tone 모델 로딩 중: {model_name}")
    print("(첫 실행 시 모델 다운로드로 인해 시간이 걸릴 수 있습니다)")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    except Exception as e:
        raise RuntimeError(f"모델 로딩 실패: {e}\n인터넷 연결을 확인하거나 모델명을 확인해주세요.") from e
    
    # Device 설정 (성능 우선순위: MPS > CUDA > CPU)
    # MPS: Apple Silicon (M1/M2 등) GPU 가속
    # CUDA: NVIDIA GPU 가속
    # CPU: 기본 처리 (가장 느림)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("사용 장치: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("사용 장치: CUDA (NVIDIA GPU)")
    else:
        device = torch.device('cpu')
        print("사용 장치: CPU (GPU를 사용할 수 없어 느릴 수 있습니다)")
    
    model = model.to(device)
    
    # CSV 파일 읽기
    print(f"\nCSV 파일 읽기: {input_csv_path}")
    rows_data = []
    texts = []
    
    try:
        with open(input_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            if 'text' not in fieldnames:
                raise KeyError(f"CSV 파일에 'text' 컬럼이 없습니다. 사용 가능한 컬럼: {fieldnames}")
            
            for row in reader:
                rows_data.append(row)
                texts.append(row['text'])
    except Exception as e:
        raise RuntimeError(f"CSV 파일 읽기 실패: {e}") from e
    
    total_count = len(texts)
    if total_count == 0:
        raise ValueError("CSV 파일에 데이터가 없습니다.")
    
    print(f"총 {total_count:,}개 포스트 읽기 완료")
    
    # 배치 단위로 감성분석 수행
    print(f"\n감성분석 진행 중 (배치 크기: {batch_size})...")
    all_results = []
    
    for i in range(0, total_count, batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_count + batch_size - 1) // batch_size
        
        print(f"배치 {batch_num}/{total_batches} 처리 중... ({i+1}~{min(i+batch_size, total_count)}/{total_count})")
        
        batch_results = analyze_sentiment_batch(
            tokenizer, model, batch_texts, device
        )
        all_results.extend(batch_results)
    
    # 결과를 기존 데이터에 추가하여 저장
    print(f"\n결과 저장 중: {output_csv_path}")
    
    # 기존 컬럼 + 새로운 컬럼
    output_fieldnames = list(fieldnames) + [
        'sentiment_label',
        'sentiment_confidence',
        'sentiment_score'
    ]
    
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        
        for row_data, result in zip(rows_data, all_results):
            # 기존 데이터 복사
            output_row = row_data.copy()
            
            # 감성분석 결과 추가
            output_row['sentiment_label'] = result['label']
            output_row['sentiment_confidence'] = f"{result['score']:.4f}"
            output_row['sentiment_score'] = f"{result['sentiment_score']:.4f}"
            
            writer.writerow(output_row)
    
    # 통계 출력
    print(f"\n{'='*60}")
    print(f"감성분석 완료!")
    print(f"{'='*60}")
    print(f"처리된 포스트: {total_count:,}개")
    print(f"출력 파일: {output_csv_path.name}")
    print(f"추가된 컬럼: sentiment_label, sentiment_confidence, sentiment_score")
    
    # 감성 분포 통계
    label_counts = {}
    for result in all_results:
        label = result['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\n【감성 레이블 분포】")
    label_names = {'positive': '긍정', 'negative': '부정', 'neutral': '중립'}
    for label in ['positive', 'negative', 'neutral']:
        count = label_counts.get(label, 0)
        percentage = (count / total_count) * 100
        label_kr = label_names.get(label, label)
        print(f"  {label_kr:4s} ({label:8s}): {count:5,}개 ({percentage:5.2f}%)")
    
    # 감성점수 통계
    sentiment_scores = [r['sentiment_score'] for r in all_results]
    avg_score = sum(sentiment_scores) / len(sentiment_scores)
    min_score = min(sentiment_scores)
    max_score = max(sentiment_scores)
    
    print(f"\n【감성점수 통계 (sentiment_score)】")
    print(f"  평균: {avg_score:+.4f}")
    print(f"  최소: {min_score:+.4f}")
    print(f"  최대: {max_score:+.4f}")
    
    # 신뢰도 통계
    confidences = [r['score'] for r in all_results]
    avg_confidence = sum(confidences) / len(confidences)
    min_confidence = min(confidences)
    max_confidence = max(confidences)
    
    print(f"\n【신뢰도 통계 (sentiment_confidence)】")
    print(f"  평균: {avg_confidence:.4f}")
    print(f"  최소: {min_confidence:.4f}")
    print(f"  최대: {max_confidence:.4f}")


# ============================================================================
# 메인 실행부
# ============================================================================

def create_backup_file(file_path: Path) -> Path:
    """
    원본 파일의 타임스탬프가 포함된 백업 생성
    
    원본 파일을 수정하기 전에 안전하게 백업합니다.
    백업 파일명 형식: {원본파일명}_backup_{YYYYMMDD_HHMMSS}.csv
    
    Args:
        file_path: 백업할 파일 경로
    
    Returns:
        Path: 생성된 백업 파일 경로
    
    Example:
        원본: trump_posts_preprocessed_term1_2017_2021.csv
        백업: trump_posts_preprocessed_term1_2017_2021_backup_20250126_143022.csv
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.parent / f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
    
    shutil.copy2(file_path, backup_path)
    print(f"원본 파일 백업 생성: {backup_path.name}")
    
    return backup_path


def main():
    """
    메인 실행 함수
    
    명령줄 인자로 입력 파일 경로를 받을 수 있습니다.
    인자가 없으면 기본 파일을 사용합니다.
    """
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    
    # 명령줄 인자에서 입력 파일 경로 가져오기
    if len(sys.argv) > 1:
        input_csv = Path(sys.argv[1])
        # 상대 경로인 경우 data 디렉토리 기준으로 변환
        if not input_csv.is_absolute():
            input_csv = data_dir / input_csv
    else:
        # 기본 파일 사용
        input_csv = data_dir / 'trump_posts_preprocessed_term2_2025.csv'
        print(f"입력 파일 경로를 지정하지 않아 기본 파일을 사용합니다: {input_csv.name}")
        print("다른 파일을 사용하려면: python track1_sentiment_score.py [파일경로]")
    
    # 입력 파일 존재 확인
    if not input_csv.exists():
        print(f"\n오류: 입력 파일을 찾을 수 없습니다: {input_csv}")
        print(f"현재 작업 디렉토리: {Path.cwd()}")
        print(f"스크립트 위치: {base_dir}")
        print(f"data 디렉토리: {data_dir}")
        if data_dir.exists():
            print(f"\ndata 디렉토리의 파일 목록:")
            for f in sorted(data_dir.glob("*.csv")):
                print(f"  - {f.name}")
        return
    
    # 원본 파일 백업 생성
    backup_path = create_backup_file(input_csv)
    
    # 임시 파일로 결과 저장
    temp_output = data_dir / f"{input_csv.stem}_temp{input_csv.suffix}"
    
    try:
        # 감성분석 수행 (임시 파일에 저장)
        process_sentiment_analysis(
            input_csv_path=input_csv,
            output_csv_path=temp_output
        )
        
        # 임시 파일을 원본 파일로 교체
        shutil.move(temp_output, input_csv)
        print(f"\n원본 파일 업데이트 완료: {input_csv}")
        print(f"백업 파일 위치: {backup_path}")
        
    except KeyboardInterrupt:
        # 사용자가 중단한 경우
        if temp_output.exists():
            temp_output.unlink()
        print(f"\n\n사용자에 의해 중단되었습니다.")
        print(f"원본 파일은 백업에서 복원 가능: {backup_path}")
        sys.exit(1)
    except Exception as e:
        # 오류 발생 시 임시 파일 삭제
        if temp_output.exists():
            temp_output.unlink()
        print(f"\n오류 발생: {type(e).__name__}: {e}")
        print(f"원본 파일은 백업에서 복원 가능: {backup_path}")
        raise


if __name__ == '__main__':
    main()

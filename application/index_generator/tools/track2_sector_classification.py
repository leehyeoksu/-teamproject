"""
트럼프 소셜미디어 포스트 통합점수 계산 스크립트 (Track 2)

이 스크립트는 전처리된 트럼프 포스트 데이터에 대해 BART-large-MNLI 모델을 사용하여
12개 ETF 섹터 × 3개 강도(high/medium/low) = 36개 라벨로 분류하고,
통합점수를 계산하여 기존 CSV 파일에 컬럼을 추가합니다.

사용법:
    python track2_sector_classification.py [입력파일경로]
    
    예시:
    python track2_sector_classification.py data/trump_posts_term1_2017_2021_with_sentiment.csv
    
    입력파일경로를 지정하지 않으면 기본값 사용:
    data/trump_posts_term2_2025_with_sentiment.csv

주요 기능:
1. 전처리된 CSV 파일 읽기
2. BART-large-MNLI 모델로 각 포스트를 36개 라벨로 분류 (12개 섹터 × 3개 강도)
3. 트럼프 지수용: 36개 중 최고 점수로 통합점수 계산
4. 대시보드용: 12개 섹터별 최고 점수 저장
5. 기존 CSV 파일에 결과 컬럼 추가 (원본 파일 백업 자동 생성)

ETF 섹터 (총 12개):
표준 S&P 500 섹터 (11개):
1. Energy (XLE) - 에너지
2. Materials (XLB) - 소재
3. Consumer Discretionary (XLY) - 경기소비재
4. Healthcare (XLV) - 헬스케어
5. Industrial (XLI) - 산업재
6. Technology (XLK) - 기술
7. Communication (XLC) - 통신
8. Financial (XLF) - 금융
9. Real Estate (XLRE) - 부동산
10. Utilities (XLU) - 유틸리티
11. Consumer Staples (XLP) - 필수소비재

추가 섹터 (1개):
12. Defense (ITA) - 방산 (표준 섹터 아님)

통합점수 계산:
- 공식: 통합점수 = (BART score × 0.8) + (강도 계수 × 0.2)
- 강도 계수: high=1.0, medium=0.7, low=0.4
- 36개 라벨 중 최고 점수를 사용하여 계산

출력 컬럼 설명:
- integrated_score: 통합점수 (0.0 ~ 1.0) - 트럼프 지수 계산에 사용
- sector_label: 최고 영향도 섹터 (36개 중 최고 점수 섹터)
- intensity_level: 강도 레벨 ("high", "medium", "low")
- bart_score: BART entailment score (36개 중 최고 점수)
- intensity_coefficient: 강도 계수 (high=1.0, medium=0.7, low=0.4)
- sector_{{sector}}_best_score: 각 섹터별 최고 점수 (12개 컬럼) - 대시보드용
- sector_{{sector}}_best_intensity: 각 섹터별 최고 점수의 강도 (12개 컬럼) - 대시보드용

주의사항:
- 모델 다운로드: 첫 실행 시 Hugging Face에서 모델을 자동 다운로드합니다.
- 처리 시간: BART-large-MNLI는 각 텍스트마다 36개 라벨을 모두 평가하므로 시간이 걸립니다.
- 메모리: GPU(MPS/CUDA) 사용 시 더 빠르게 처리됩니다.
- 백업: 원본 파일은 자동으로 타임스탬프가 포함된 백업 파일로 저장됩니다.

중요: Entailment Score vs 확률
- BART-large-MNLI는 확률(probability)이 아닌 entailment score를 반환합니다.
- Entailment score: 텍스트가 "This text is about [sector]"라는 가설과 얼마나 일치하는지
- 각 라벨에 대한 점수는 독립적으로 계산되며, 합이 1.0이 되지 않습니다.
- 점수가 높을수록 해당 섹터와의 관련성이 높다는 의미입니다.
"""

import csv
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import torch
from transformers import pipeline


# ============================================================================
# 상수 정의
# ============================================================================

# BART-large-MNLI 모델명
# Zero-shot classification을 위한 모델
# 첫 실행 시 자동으로 다운로드됩니다.
BART_MODEL_NAME = "facebook/bart-large-mnli"

# 배치 처리 크기
# 한 번에 처리할 텍스트 개수 (메모리와 속도의 균형)
# BART-large-MNLI는 각 텍스트마다 여러 레이블을 평가하므로 메모리를 많이 사용합니다.
# GPU 메모리가 부족하면 이 값을 줄이세요 (예: 8, 4)
BATCH_SIZE = 16

# ETF 섹터 정의
# 팀원 리스트 기준: 12개 섹터
# 표준 S&P 500 섹터: 11개 (GICS 분류)
# 추가 섹터: 1개 (Defense)
ETF_SECTORS = [
    # 표준 S&P 500 섹터 (11개)
    "energy",                        # XLE - Energy Select Sector SPDR Fund
    "materials",                     # XLB - Materials Select Sector SPDR Fund
    "consumer discretionary",         # XLY - Consumer Discretionary Select Sector SPDR Fund
    "healthcare",                    # XLV - Health Care Select Sector SPDR Fund
    "industrial",                    # XLI - Industrial Select Sector SPDR Fund
    "technology",                    # XLK - Technology Select Sector SPDR Fund
    "communication services",        # XLC - Communication Services Select Sector SPDR Fund
    "financial",                     # XLF - Financial Select Sector SPDR Fund
    "real estate",                   # XLRE - Real Estate Select Sector SPDR Fund
    "utilities",                     # XLU - Utilities Select Sector SPDR Fund
    "consumer staples",              # XLP - Consumer Staples Select Sector SPDR Fund
    # 추가 섹터 (1개)
    "defense"                        # ITA - iShares U.S. Aerospace & Defense ETF (표준 섹터 아님)
]

# 섹터 레이블 한글명 (출력용)
SECTOR_NAMES_KR = {
    "technology": "기술",
    "healthcare": "건강관리",
    "financial": "금융",
    "consumer discretionary": "소비재(선택)",
    "consumer staples": "소비재(필수)",
    "energy": "에너지",
    "industrial": "산업",
    "materials": "소재",
    "real estate": "부동산",
    "utilities": "유틸리티",
    "communication services": "통신서비스",
    "defense": "방산"
}

# 강도 계수 정의
INTENSITY_COEFFICIENTS = {
    "high": 1.0,
    "medium": 0.7,
    "low": 0.4
}

# 36개 candidate labels 생성 (12개 섹터 × 3개 강도)
def generate_candidate_labels() -> List[str]:
    """
    36개 candidate labels 생성
    
    Returns:
        List[str]: 36개 라벨 리스트
            예: ["high technology impact", "medium technology impact", ...]
    """
    intensities = ["high", "medium", "low"]
    labels = []
    
    for sector in ETF_SECTORS:
        for intensity in intensities:
            labels.append(f"{intensity} {sector} impact")
    
    return labels

# 36개 candidate labels
CANDIDATE_LABELS = generate_candidate_labels()


# ============================================================================
# 유틸리티 함수
# ============================================================================

def extract_sector_and_intensity(label: str) -> Tuple[str, str]:
    """
    라벨에서 섹터와 강도를 추출
    
    Args:
        label: BART가 선택한 라벨 (예: "high energy impact")
    
    Returns:
        tuple: (sector, intensity)
            - sector: 섹터명 (예: "energy")
            - intensity: 강도 (예: "high")
    """
    # "high energy impact" → ("energy", "high")
    parts = label.split()
    intensity = parts[0]  # "high", "medium", "low"
    sector = " ".join(parts[1:-1])  # "energy" (마지막 "impact" 제외)
    
    return sector, intensity


def select_best_per_sector(all_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    각 섹터별로 최고 점수를 가진 라벨 선택 (대시보드용)
    
    Args:
        all_results: BART가 반환한 36개 라벨의 점수 리스트 (점수 순으로 정렬됨)
    
    Returns:
        Dict: 각 섹터별 최고 점수 정보
            {
                'energy': {
                    'score': 0.85,
                    'intensity': 'high'
                },
                'technology': {
                    'score': 0.82,
                    'intensity': 'medium'
                },
                # ... (12개 섹터)
            }
    """
    sector_best = {}
    
    for result in all_results:
        sector, intensity = extract_sector_and_intensity(result['label'])
        
        # 섹터별로 최고 점수만 저장
        if sector not in sector_best or result['score'] > sector_best[sector]['score']:
            sector_best[sector] = {
                'score': result['score'],
                'intensity': intensity
            }
    
    return sector_best


def calculate_integrated_score(bart_score: float, intensity: str) -> float:
    """
    통합점수 계산 (트럼프 지수용)
    
    공식: 통합점수 = (BART score × 0.8) + (강도 계수 × 0.2)
    
    Args:
        bart_score: BART entailment score (0.0 ~ 1.0) - 36개 중 최고 점수
        intensity: 강도 ("high", "medium", "low") - 최고 점수 라벨에서 추출
    
    Returns:
        float: 통합점수 (0.0 ~ 1.0)
    
    예시:
        >>> calculate_integrated_score(0.90, "high")
        0.92  # (0.90 × 0.8) + (1.0 × 0.2) = 0.72 + 0.2 = 0.92
    """
    intensity_coefficient = INTENSITY_COEFFICIENTS[intensity]
    integrated_score = (bart_score * 0.8) + (intensity_coefficient * 0.2)
    return integrated_score


# ============================================================================
# 섹터 분류 함수
# ============================================================================

def classify_sector_batch(
    classifier: Any,
    texts: List[str],
    candidate_labels: List[str]
) -> List[Dict[str, Any]]:
    """
    텍스트 배치에 대해 섹터 분류 수행
    
    BART-large-MNLI는 zero-shot classification을 수행합니다.
    각 텍스트에 대해 모든 후보 레이블(36개)에 대한 entailment score를 계산하고,
    점수 순으로 정렬된 리스트를 반환합니다.
    
    주의: 이 점수는 확률(probability)이 아닌 entailment score입니다.
    - 확률: 모든 레이블의 합이 1.0이 되어야 함
    - Entailment score: 각 레이블에 대한 독립적인 점수, 합이 1.0이 되지 않음
    - 텍스트가 "This text is about [sector]"라는 가설과 얼마나 일치하는지를 나타냄
    
    Args:
        classifier: BART-large-MNLI 분류기 (pipeline)
        texts: 분석할 텍스트 리스트
        candidate_labels: 후보 라벨 리스트 (36개)
    
    Returns:
        List[Dict]: 각 텍스트의 분류 결과
            각 Dict는 다음 키를 포함:
            - 'all_results': List[Dict] - 36개 라벨의 점수 리스트 (점수 순으로 정렬)
            - 'best_overall': Dict - 36개 중 최고 점수 라벨
            - 'sector_best': Dict - 각 섹터별 최고 점수 (대시보드용)
    """
    results = []
    
    for text in texts:
        # BART-large-MNLI는 각 텍스트에 대해 모든 레이블을 평가
        # 결과는 딕셔너리 형태: {'sequence': text, 'labels': [...], 'scores': [...]}
        classification_result = classifier(text, candidate_labels)
        
        # 딕셔너리를 리스트 형태로 변환 (점수 순으로 정렬됨)
        results_list = [
            {'label': label, 'score': score}
            for label, score in zip(classification_result['labels'], classification_result['scores'])
        ]
        
        # 36개 중 최고 점수 라벨 (트럼프 지수용)
        best_overall = results_list[0]
        
        # 섹터별 최고 점수 선택 (대시보드용)
        sector_best = select_best_per_sector(results_list)
        
        results.append({
            'all_results': results_list,  # 36개 전체 결과 (리스트 형태)
            'best_overall': best_overall,  # 36개 중 최고
            'sector_best': sector_best  # 섹터별 최고
        })
    
    return results


def process_sector_classification(
    input_csv_path: Path,
    output_csv_path: Path,
    model_name: str = BART_MODEL_NAME,
    batch_size: int = BATCH_SIZE
) -> None:
    """
    전처리된 CSV 파일에 대해 섹터 분류 수행하고 결과 컬럼 추가
    
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
    
    print(f"BART-large-MNLI 모델 로딩 중: {model_name}")
    print("(첫 실행 시 모델 다운로드로 인해 시간이 걸릴 수 있습니다)")
    
    # Device 설정 (성능 우선순위: MPS > CUDA > CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "MPS (Apple Silicon GPU)"
    elif torch.cuda.is_available():
        device = 0  # 첫 번째 CUDA GPU
        device_name = "CUDA (NVIDIA GPU)"
    else:
        device = -1  # CPU
        device_name = "CPU (GPU를 사용할 수 없어 느릴 수 있습니다)"
    
    print(f"사용 장치: {device_name}")
    
    try:
        # Zero-shot classification pipeline 생성
        # device 파라미터: torch.device 객체, 정수(0=CUDA, -1=CPU), 또는 "mps"
        classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device
        )
    except Exception as e:
        raise RuntimeError(
            f"모델 로딩 실패: {e}\n"
            "인터넷 연결을 확인하거나 모델명을 확인해주세요."
        ) from e
    
    # CSV 파일 읽기
    print(f"\nCSV 파일 읽기: {input_csv_path}")
    rows_data = []
    texts = []
    
    try:
        with open(input_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            if 'text' not in fieldnames:
                raise KeyError(
                    f"CSV 파일에 'text' 컬럼이 없습니다. "
                    f"사용 가능한 컬럼: {fieldnames}"
                )
            
            for row in reader:
                rows_data.append(row)
                texts.append(row['text'])
    except Exception as e:
        raise RuntimeError(f"CSV 파일 읽기 실패: {e}") from e
    
    total_count = len(texts)
    if total_count == 0:
        raise ValueError("CSV 파일에 데이터가 없습니다.")
    
    print(f"총 {total_count:,}개 포스트 읽기 완료")
    print(f"분류할 라벨: {len(CANDIDATE_LABELS)}개 (12개 섹터 × 3개 강도)")
    print(f"  섹터: {', '.join(ETF_SECTORS[:6])}")
    print(f"        {', '.join(ETF_SECTORS[6:])}")
    print(f"  강도: high, medium, low")
    
    # 배치 단위로 섹터 분류 수행
    print(f"\n통합점수 계산 진행 중 (배치 크기: {batch_size})...")
    print("(각 포스트마다 36개 라벨을 모두 평가하므로 시간이 걸립니다)")
    all_results = []
    
    for i in range(0, total_count, batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_count + batch_size - 1) // batch_size
        
        print(
            f"배치 {batch_num}/{total_batches} 처리 중... "
            f"({i+1}~{min(i+batch_size, total_count)}/{total_count})"
        )
        
        batch_results = classify_sector_batch(
            classifier, batch_texts, CANDIDATE_LABELS
        )
        all_results.extend(batch_results)
    
    # 결과를 기존 데이터에 추가하여 저장
    print(f"\n결과 저장 중: {output_csv_path}")
    
    # 기존 컬럼 + 새로운 컬럼
    # 1. 트럼프 지수용 컬럼
    # 2. 대시보드용 컬럼 (섹터별 최고 점수)
    sector_best_score_columns = [f"sector_{sector}_best_score" for sector in ETF_SECTORS]
    sector_best_intensity_columns = [f"sector_{sector}_best_intensity" for sector in ETF_SECTORS]
    
    output_fieldnames = list(fieldnames) + [
        # 트럼프 지수용
        'integrated_score',       # 통합점수
        'sector_label',           # 최고 영향도 섹터 (36개 중 최고)
        'intensity_level',        # 강도 레벨
        'bart_score',            # BART entailment score (36개 중 최고)
        'intensity_coefficient'   # 강도 계수
    ] + sector_best_score_columns + sector_best_intensity_columns  # 대시보드용
    
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        
        for row_data, result in zip(rows_data, all_results):
            # 기존 데이터 복사
            output_row = row_data.copy()
            
            # 트럼프 지수용: 36개 중 최고 점수로 통합점수 계산
            best_overall = result['best_overall']
            sector, intensity = extract_sector_and_intensity(best_overall['label'])
            integrated_score = calculate_integrated_score(best_overall['score'], intensity)
            intensity_coefficient = INTENSITY_COEFFICIENTS[intensity]
            
            output_row['integrated_score'] = f"{integrated_score:.4f}"
            output_row['sector_label'] = sector
            output_row['intensity_level'] = intensity
            output_row['bart_score'] = f"{best_overall['score']:.4f}"
            output_row['intensity_coefficient'] = f"{intensity_coefficient:.4f}"
            
            # 대시보드용: 섹터별 최고 점수 저장
            sector_best = result['sector_best']
            for sector in ETF_SECTORS:
                if sector in sector_best:
                    output_row[f'sector_{sector}_best_score'] = f"{sector_best[sector]['score']:.4f}"
                    output_row[f'sector_{sector}_best_intensity'] = sector_best[sector]['intensity']
                else:
                    output_row[f'sector_{sector}_best_score'] = "0.0000"
                    output_row[f'sector_{sector}_best_intensity'] = ""
            
            writer.writerow(output_row)
    
    # 통계 출력
    print(f"\n{'='*60}")
    print(f"통합점수 계산 완료!")
    print(f"{'='*60}")
    print(f"처리된 포스트: {total_count:,}개")
    print(f"출력 파일: {output_csv_path.name}")
    
    # 처리된 텍스트 샘플 출력 (처음 3개)
    print(f"\n【처리된 텍스트 샘플】")
    for i, (row_data, result) in enumerate(zip(rows_data[:3], all_results[:3])):
        text = row_data.get('text', '')
        text_preview = text[:150] + "..." if len(text) > 150 else text
        best_overall = result['best_overall']
        sector, intensity = extract_sector_and_intensity(best_overall['label'])
        integrated_score = calculate_integrated_score(best_overall['score'], intensity)
        intensity_coefficient = INTENSITY_COEFFICIENTS[intensity]
        
        print(f"\n  [{i+1}] 텍스트: {text_preview}")
        print(f"      └─ 선택된 결과:")
        print(f"         섹터: {sector} ({SECTOR_NAMES_KR.get(sector, sector)})")
        print(f"         강도: {intensity} (계수: {intensity_coefficient:.1f})")
        print(f"         BART 점수: {best_overall['score']:.4f}")
        print(f"         통합점수: {integrated_score:.4f} = ({best_overall['score']:.4f} × 0.8) + ({intensity_coefficient:.1f} × 0.2)")
        
        # 상위 10개 라벨 점수 출력
        print(f"      └─ 상위 10개 라벨 점수 (전체 36개 중):")
        top_10 = result['all_results'][:10]
        for j, item in enumerate(top_10, 1):
            marker = "★" if j == 1 else " "
            print(f"         {marker} {j:2d}. {item['label']:35s} : {item['score']:.4f}")
        
        # 섹터별 최고 점수 출력
        print(f"      └─ 섹터별 최고 점수 (대시보드용, 상위 5개):")
        sector_best = result['sector_best']
        sorted_sectors = sorted(sector_best.items(), key=lambda x: x[1]['score'], reverse=True)[:5]
        for sector_name, sector_info in sorted_sectors:
            sector_kr = SECTOR_NAMES_KR.get(sector_name, sector_name)
            print(f"         - {sector_kr:12s} ({sector_name:20s}): {sector_info['score']:.4f} ({sector_info['intensity']})")
    
    print(f"\n추가된 컬럼:")
    print(f"【트럼프 지수용】")
    print(f"  - integrated_score: 통합점수 (0.0 ~ 1.0)")
    print(f"  - sector_label: 최고 영향도 섹터 (36개 중 최고)")
    print(f"  - intensity_level: 강도 레벨 (high/medium/low)")
    print(f"  - bart_score: BART entailment score (36개 중 최고)")
    print(f"  - intensity_coefficient: 강도 계수 (high=1.0, medium=0.7, low=0.4)")
    print(f"\n【대시보드용】")
    print(f"  - sector_{{sector}}_best_score: 각 섹터별 최고 점수 (12개 컬럼)")
    print(f"  - sector_{{sector}}_best_intensity: 각 섹터별 최고 점수의 강도 (12개 컬럼)")
    print(f"    예: sector_energy_best_score, sector_technology_best_score, ...")
    print(f"  (참고: 모든 점수는 확률이 아닌 entailment score입니다)")
    
    # 섹터 분포 통계
    sector_counts = {}
    intensity_counts = {"high": 0, "medium": 0, "low": 0}
    integrated_scores = []
    
    for result in all_results:
        best_overall = result['best_overall']
        sector, intensity = extract_sector_and_intensity(best_overall['label'])
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        intensity_counts[intensity] = intensity_counts[intensity] + 1
        integrated_score = calculate_integrated_score(best_overall['score'], intensity)
        integrated_scores.append(integrated_score)
    
    print(f"\n【섹터 분포 (36개 중 최고 점수)】")
    for sector in sorted(sector_counts.keys()):
        count = sector_counts[sector]
        percentage = (count / total_count) * 100
        sector_kr = SECTOR_NAMES_KR.get(sector, sector)
        print(f"  {sector_kr:12s} ({sector:20s}): {count:5,}개 ({percentage:5.2f}%)")
    
    print(f"\n【강도 분포】")
    for intensity in ["high", "medium", "low"]:
        count = intensity_counts[intensity]
        percentage = (count / total_count) * 100
        print(f"  {intensity:6s}: {count:5,}개 ({percentage:5.2f}%)")
    
    # 통합점수 통계
    avg_integrated = sum(integrated_scores) / len(integrated_scores)
    min_integrated = min(integrated_scores)
    max_integrated = max(integrated_scores)
    
    print(f"\n【통합점수 통계】")
    print(f"  평균: {avg_integrated:.4f}")
    print(f"  최소: {min_integrated:.4f}")
    print(f"  최대: {max_integrated:.4f}")
    
    # BART 점수 통계
    bart_scores = [r['best_overall']['score'] for r in all_results]
    avg_bart = sum(bart_scores) / len(bart_scores)
    min_bart = min(bart_scores)
    max_bart = max(bart_scores)
    
    print(f"\n【BART Entailment 점수 통계 (36개 중 최고)】")
    print(f"  (참고: 이 값은 확률이 아닌 entailment score입니다)")
    print(f"  평균: {avg_bart:.4f}")
    print(f"  최소: {min_bart:.4f}")
    print(f"  최대: {max_bart:.4f}")


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
        원본: trump_posts_term1_2017_2021_with_sentiment.csv
        백업: trump_posts_term1_2017_2021_with_sentiment_backup_20250126_143022.csv
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
        # 상대 경로인 경우 현재 작업 디렉토리 기준으로 해석
        if not input_csv.is_absolute():
            input_csv = Path.cwd() / input_csv
    else:
        # 기본 파일 사용
        input_csv = data_dir / 'trump_posts_term2_2025_with_sentiment.csv'
        print(
            f"입력 파일 경로를 지정하지 않아 기본 파일을 사용합니다: "
            f"{input_csv.name}"
        )
        print("다른 파일을 사용하려면: python track2_sector_classification.py [파일경로]")
    
    # 입력 파일 존재 확인
    if not input_csv.exists():
        print(f"\n오류: 입력 파일을 찾을 수 없습니다: {input_csv}")
        print(f"현재 작업 디렉토리: {Path.cwd()}")
        print(f"스크립트 위치: {base_dir}")
        print(f"data 디렉토리: {data_dir}")
        if data_dir.exists():
            print(f"\ndata 디렉토리의 파일 목록:")
            for f in sorted(data_dir.glob("*_with_sentiment.csv")):
                print(f"  - {f.name}")
        return
    
    # 원본 파일 백업 생성
    backup_path = create_backup_file(input_csv)
    
    # 임시 파일로 결과 저장
    temp_output = data_dir / f"{input_csv.stem}_temp{input_csv.suffix}"
    
    try:
        # 섹터 분류 수행 (임시 파일에 저장)
        process_sector_classification(
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

"""
TSUI 지수 산출 파이프라인

이 모듈은 지수 산출의 전체 공정을 오케스트레이션합니다.
"""

import csv
from pathlib import Path
from typing import List, Dict, Any, Optional

from .data.data_loader import (
    initialize_data,
    get_track1_data,
    get_track2_data
)
from .engine.index_calculator import (
    calculate_tsui_index
)
from .logic.logic_helpers import (
    group_posts_by_date,
    generate_all_dates,
    extract_daily_sectors,
    calculate_daily_energies
)
from .engine.math_engines import (
    calculate_cumulative_energy
)
from .data.constants import (
    BASELINE_DATE_TERM1,
    BASELINE_DATE_TERM2,
)


# ============================================================================
# Helper Functions (로컬 유지)
# ============================================================================

def _load_and_cache_data(
    track1_csv: Path,
    track2_csv: Optional[Path],
    etf_dir: Path,
    term: str
) -> tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """
    데이터 로드 및 캐싱
    
    Args:
        track1_csv: Track 1 CSV 파일 경로
        track2_csv: Track 2 CSV 파일 경로 (None 가능)
        etf_dir: ETF 데이터 디렉토리
        term: 기간 ("term1" 또는 "term2")
    
    Returns:
        tuple: (track1_data, track2_data, log_info)
        - log_info: 로그 출력용 정보 딕셔너리
    """
    period = "2017-2021" if term == "term1" else "2025-now"
    
    initialize_data(track1_csv, track2_csv, etf_dir, period)
    track1_data = get_track1_data()
    track2_data = get_track2_data() or {}
    
    # 로그 정보 수집
    log_info = {
        'track1_name': track1_csv.name,
        'track1_count': len(track1_data),
        'track2_name': track2_csv.name if track2_csv and track2_csv.exists() else None,
        'track2_count': len(track2_data) if track2_csv and track2_csv.exists() else 0,
        'has_track2': track2_csv is not None and track2_csv.exists()
    }
    
    return track1_data, track2_data, log_info


def _save_results_to_csv(
    output_csv: Path,
    tsui_results: Dict[str, Dict[str, Any]],
    daily_energies: Dict[str, float],
    daily_posts: Dict[str, List[Dict[str, Any]]]
) -> None:
    """
    결과를 CSV 파일로 저장
    
    Args:
        output_csv: 출력 CSV 파일 경로
        tsui_results: TSUI 계산 결과
        daily_energies: 일일 원천 에너지
        daily_posts: 일별 포스트 딕셔너리
    """
    output_fieldnames = [
        'date',
        'tsui_index',  # EMA 적용된 최종 지수
        'raw_tsui_index',  # EMA 적용 전 원본 지수
        'raw_energy',
        'cumulative_energy',
        'vol_proxy',  # 원본 vol_proxy (클리핑됨: 0.001~0.05)
        'raw_value',  # S_cumulative * Vol_Proxy (로그 스케일링 전)
        'scaled_value',  # ln(raw_value * 1000 + 1) (로그 스케일링 후)
        'baseline_scaled',  # Dynamic Baseline
        'etf_symbol',
        'sector',
        'post_count'
    ]
    
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        
        for date_str in sorted(tsui_results.keys()):
            result = tsui_results[date_str]
            row = {
                'date': result['date'],
                'tsui_index': f"{result['tsui_index']:.4f}",  # EMA 적용된 최종 지수
                'raw_tsui_index': f"{result.get('raw_tsui_index', result['tsui_index']):.4f}",  # EMA 적용 전 원본 지수
                'raw_energy': f"{daily_energies.get(date_str, 0.0):.4f}",
                'cumulative_energy': f"{result['cumulative_energy']:.4f}",
                'vol_proxy': f"{result['vol_proxy']:.4f}",
                'raw_value': f"{result.get('raw_value', 0.0):.4f}",
                'scaled_value': f"{result.get('scaled_value', 0.0):.4f}",
                'baseline_scaled': f"{result.get('baseline_scaled', 0.0):.4f}",
                'etf_symbol': result.get('etf_symbol', 'SPY'),
                'sector': result.get('sector', ''),
                'post_count': len(daily_posts.get(date_str, []))
            }
            writer.writerow(row)


def _print_statistics(tsui_results: Dict[str, Dict[str, Any]]) -> None:
    """
    통계 출력
    
    Args:
        tsui_results: TSUI 계산 결과
    """
    tsui_values = [r['tsui_index'] for r in tsui_results.values()]
    baseline_tsui = 100.0
    if tsui_values:
        print(f"  평균 TSUI: {sum(tsui_values) / len(tsui_values):.2f}")
        print(f"  최소 TSUI: {min(tsui_values):.2f}")
        print(f"  최대 TSUI: {max(tsui_values):.2f}")
        print(f"  기준점 TSUI: {baseline_tsui:.2f}")


# ============================================================================
# Main Function
# ============================================================================

def run_analysis(
    track1_csv: Path,
    track2_csv: Optional[Path],
    etf_dir: Path,
    output_csv: Path,
    term: str = "term1"
) -> None:
    """
    지수 산출 메인 처리 함수
    
    Args:
        track1_csv: Track 1 CSV 파일 경로
        track2_csv: Track 2 CSV 파일 경로 (None 가능)
        etf_dir: ETF 데이터 디렉토리
        output_csv: 출력 CSV 파일 경로
        term: 기간 ("term1" 또는 "term2")
    """
    print("=" * 60)
    print("TSUI 지수 산출 시작")
    print("=" * 60)
    
    # 1단계: 데이터 로드 및 캐싱
    print(f"\n[1단계] 데이터 로드 및 캐싱")
    track1_data, track2_data, load_log = _load_and_cache_data(track1_csv, track2_csv, etf_dir, term)
    print(f"  Track 1: {load_log['track1_name']}")
    print(f"    - {load_log['track1_count']:,}개 포스트 로드 완료")
    if load_log['has_track2']:
        print(f"  Track 2: {load_log['track2_name']}")
        print(f"    - {load_log['track2_count']:,}개 포스트 매칭 완료")
    else:
        print(f"  Track 2: 없음 (통합점수 1.0으로 가정)")
    
    # 기준일 설정 (기준일 이전 데이터 필터링용)
    baseline = BASELINE_DATE_TERM1 if term == "term1" else BASELINE_DATE_TERM2
    baseline_date_only = baseline.date()
    
    # 2단계: 일별 포스트 그룹화 및 전체 기간 생성
    print(f"\n[2단계] 일별 포스트 그룹화 (기준일: {baseline_date_only.isoformat()} 이후)")
    daily_posts, daily_intensity, start_date, end_date, group_log = group_posts_by_date(
        track1_data, baseline_date_only, track2_data
    )
    print(f"    - 기준일 이전 {group_log['filtered_count']:,}개 포스트 제외")
    print(f"    - 발언이 있는 날: {group_log['days_with_posts']}일")
    print(f"    - 전체 기간: {group_log['start_date']} ~ {group_log['end_date']} ({group_log['total_days']}일)")
    
    all_dates = generate_all_dates(start_date, end_date)
    
    # 2-1단계: 일별 섹터 정보 추출
    print(f"\n[2-1단계] 일별 섹터 정보 추출 (연속 일자 기반)")
    daily_sectors, sector_log = extract_daily_sectors(all_dates, daily_posts, track2_data)
    print(f"    - 섹터 특정: {sector_log['sector_specified_days']}일, 섹터 불분명/발언 없음: {sector_log['sector_unspecified_days']}일")
    
    # 3단계: 일일 원천 에너지 계산
    print(f"\n[3단계] 일일 원천 에너지 계산 (연속 일자 기반)")
    daily_energies, energy_log = calculate_daily_energies(all_dates, daily_posts, term)
    if energy_log['has_energy_data']:
        print(f"    - 발언이 있는 날 평균 에너지: {energy_log['avg_energy']:.4f}")
    print(f"    - 전체 기간: {energy_log['total_days']}일 (발언 없는 날 포함)")
    
    # 4단계: 누적 에너지 계산
    print(f"\n[4단계] 누적 에너지 계산 (연속 일자 기반)")
    cumulative_energies = calculate_cumulative_energy(daily_energies, daily_intensity)
    # 발언이 있는 날의 평균만 계산
    days_with_posts_cum = [e for date_str, e in cumulative_energies.items() if date_str in daily_posts]
    if days_with_posts_cum:
        print(f"    - 발언이 있는 날 평균 누적 에너지: {sum(days_with_posts_cum) / len(days_with_posts_cum):.4f}")
    print(f"    - 전체 기간 누적 에너지 계산 완료: {len(cumulative_energies)}일")
    
    # 5단계: 최종 TSUI 지수 계산
    print(f"\n[5단계] 최종 TSUI 지수 계산 (Vol_Proxy 동적 선택)")
    tsui_results = calculate_tsui_index(
        cumulative_energies, 
        daily_sectors,
        baseline
    )
    
    # 6단계: 결과 저장
    print(f"\n[6단계] 결과 저장: {output_csv.name}")
    _save_results_to_csv(output_csv, tsui_results, daily_energies, daily_posts)
    
    # 통계 출력
    print(f"\n{'='*60}")
    print(f"지수 산출 완료!")
    print(f"{'='*60}")
    _print_statistics(tsui_results)

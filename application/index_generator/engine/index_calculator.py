"""
TSUI 지수 산출 함수들

이 모듈은 최종 TSUI 지수를 계산하는 함수를 제공합니다.
"""

from typing import Dict, Any, Optional
from datetime import datetime

from ..data.data_loader import (
    load_etf_metadata,
    get_all_etf_data,
    get_dxy_data,
    get_etf_metadata,
    get_etf_dir,
    get_period
)
from ..logic.logic_helpers import (
    calculate_all_dates_vol_proxy,
    determine_baseline_scaled
)
from .math_engines import (
    calculate_log_scaled_values,
    normalize_to_baseline,
    apply_ema_smoothing,
    renormalize_after_ema
)


def calculate_tsui_index(
    cumulative_energies: Dict[str, float],
    daily_sectors: Dict[str, Optional[str]],
    baseline_date: datetime
) -> Dict[str, Dict[str, Any]]:
    """
    최종 TSUI 지수 산출
    
    공식: TSUI_t = S_cumulative_t × Vol_Proxy_t
    
    Vol_Proxy 데이터 소스 선택 로직:
    - 섹터 특정 시: Track 2에서 분류된 해당 섹터의 대표 ETF 사용
    - 섹터 불분명/전체 정책 시: S&P 500 (SPY) 또는 달러 인덱스 (DXY) 사용
    
    Args:
        cumulative_energies: {날짜(YYYY-MM-DD): 누적 에너지}
        daily_sectors: {날짜(YYYY-MM-DD): 섹터명} (없으면 None)
        baseline_date: 기준점 날짜 (지수 100으로 설정)
    
    Returns:
        Dict: {날짜: {tsui_index, raw_energy, cumulative_energy, vol_proxy, etf_symbol, ...}}
    """
    # 캐시에서 데이터 가져오기
    all_etf_data = get_all_etf_data() or {}
    dxy_data = get_dxy_data() or {}
    etf_dir = get_etf_dir()
    if etf_dir is None:
        raise ValueError("etf_dir가 설정되지 않았습니다. initialize_data()를 먼저 호출하세요.")
    period = get_period()
    if period is None:
        raise ValueError("period가 설정되지 않았습니다. initialize_data()를 먼저 호출하세요.")
    
    baseline_tsui = 100.0
    
    # ETF 메타데이터 로드 (섹터-ETF 매핑)
    sector_to_etf = get_etf_metadata() or load_etf_metadata(etf_dir)
    
    # 원래 기준일 저장 (결과에서 항상 100으로 설정하기 위해)
    init_date_str = baseline_date.date().isoformat()
    
    # 기준점 날짜의 누적 에너지 찾기
    baseline_date_str = init_date_str
    baseline_cumulative = cumulative_energies.get(baseline_date_str, 0.0)
    
    # 기준점이 0이면 첫 번째 0이 아닌 값 사용 (정규화 기준으로만 사용)
    base_date_str = baseline_date_str
    base_energy = baseline_cumulative
    if baseline_cumulative == 0.0:
        for date_str, energy in cumulative_energies.items():
            if energy > 0.0:
                base_energy = energy
                base_date_str = date_str
                break
    
    if base_energy == 0.0:
        base_energy = 1.0  # 기본값
    
    sorted_dates = sorted(cumulative_energies.keys())
    
    # 1단계: 모든 날짜에 대해 Vol_Proxy 계산 및 클리핑
    vol_map = calculate_all_dates_vol_proxy(
        sorted_dates,
        daily_sectors,
        sector_to_etf
    )
    
    # 2단계: 로그 스케일링
    scaled_values = calculate_log_scaled_values(
        sorted_dates,
        cumulative_energies,
        vol_map
    )
    
    # 3단계: Dynamic Baseline 설정
    baseline_scaled = determine_baseline_scaled(
        sorted_dates,
        init_date_str,
        scaled_values
    )
    
    init_scaled_val = scaled_values.get(init_date_str, {}).get('scaled', 0.0)
    
    # 4단계: Baseline 기준으로 정규화
    results = normalize_to_baseline(
        sorted_dates,
        init_date_str,
        baseline_tsui,
        baseline_scaled,
        init_scaled_val,
        scaled_values,
        base_energy
    )
    
    # 5단계: EMA 적용
    results = apply_ema_smoothing(results)
    
    # 6단계: EMA 계산 후 재정규화 및 최종 보정
    results = renormalize_after_ema(results, init_date_str, baseline_tsui)
    
    return results

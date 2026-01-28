"""
순수 수학 계산 엔진

로그 스케일링, EMA, 정규화 등 순수 수학적 계산 함수들을 제공합니다.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from ..data.constants import (
    BASELINE_SCALED_THRESHOLD,
    SCALED_VALUE_EPSILON,
    SECOND_DAY_OFFSET,
    EMA_SPAN,
    LOG_SCALING_AMPLIFIER
)
from ..utils import get_residue_alpha


def calculate_log_scaled_values(
    sorted_dates: List[str],
    cumulative_energies: Dict[str, float],
    vol_map: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    로그 스케일링 적용
    
    공식: scaled_value = ln(S_cumulative * Vol_Proxy * LOG_SCALING_AMPLIFIER + 1)
    
    Args:
        sorted_dates: 정렬된 날짜 리스트
        cumulative_energies: {날짜: 누적 에너지}
        vol_map: {날짜: {vol_proxy, etf_symbol, sector}}
    
    Returns:
        Dict: {날짜: {scaled, raw_value, cumulative_energy, vol_proxy, etf_symbol, sector}}
    """
    scaled_values = {}
    for date_str in sorted_dates:
        s_cumulative = cumulative_energies[date_str]
        vol_data = vol_map[date_str]
        vol_proxy = vol_data['vol_proxy']  # 원본 vol_proxy (정규화 안 함)
        etf_symbol = vol_data['etf_symbol']
        sector = vol_data['sector']
        
        # 로그 스케일링: ln(S_cumulative * Vol_Proxy * 증폭계수 + 1)
        # 정규화된 vol_proxy를 쓰지 않고 원본 vol_proxy 사용
        # 증폭 계수 LOG_SCALING_AMPLIFIER 적용하여 고점에서의 상승 기울기를 완만하게 조정
        raw_value = s_cumulative * vol_proxy
        scaled_value = np.log1p(raw_value * LOG_SCALING_AMPLIFIER)  # np.log1p(x) = ln(x + 1)
        
        scaled_values[date_str] = {
            'scaled': scaled_value,
            'raw_value': raw_value,  # 로그 스케일링 전 값
            'cumulative_energy': s_cumulative,
            'vol_proxy': vol_proxy,  # 원본 vol_proxy
            'etf_symbol': etf_symbol,
            'sector': sector
        }
    
    return scaled_values


def normalize_to_baseline(
    sorted_dates: List[str],
    init_date_str: str,
    baseline_tsui: float,
    baseline_scaled: float,
    init_scaled_val: float,
    scaled_values: Dict[str, Dict[str, Any]],
    base_energy: float
) -> Dict[str, Dict[str, Any]]:
    """
    Baseline 기준으로 정규화
    
    공식: raw_tsui_index = (scaled_value / baseline_scaled) * 100
    
    Args:
        sorted_dates: 정렬된 날짜 리스트
        init_date_str: 원래 기준일 (YYYY-MM-DD)
        baseline_tsui: 기준 TSUI 값 (100.0)
        baseline_scaled: Dynamic Baseline scaled 값
        init_scaled_val: 기준일의 원래 scaled 값
        scaled_values: {날짜: {scaled, raw_value, ...}}
        base_energy: 정규화 기준 누적 에너지
    
    Returns:
        Dict: {날짜: {date, raw_tsui_index, cumulative_energy, vol_proxy, raw_value, scaled_value, baseline_scaled, etf_symbol, sector, baseline_energy}}
    """
    results = {}
    
    for date_str in sorted_dates:
        scaled_data = scaled_values[date_str]
        scaled_value = scaled_data['scaled']
        
        # 기준일은 항상 100으로 고정
        if date_str == init_date_str:
            raw_tsui_index = baseline_tsui  # 기준일은 항상 100
        elif baseline_scaled == 0.0:
            raw_tsui_index = baseline_tsui  # 0으로 나누기 방지
        else:
            # 기준일의 scaled_value가 0이어서 baseline_scaled가 다른 날짜 값인 경우,
            # 둘째날이 baseline_scaled와 정확히 같으면 100이 되어 기준일과 같아짐
            # 이를 방지하기 위해 둘째날은 약간의 오프셋을 추가하거나, 
            # baseline_scaled를 기준일의 scaled_value로 사용하되 기준일만 100으로 고정
            if date_str > init_date_str and abs(scaled_value - baseline_scaled) < SCALED_VALUE_EPSILON:
                # 둘째날이 baseline_scaled와 거의 같으면 (부동소수점 오차 고려),
                # 기준일의 scaled_value(0)를 사용하여 계산하되, 0으로 나누기 방지
                if init_scaled_val > 0.0:
                    raw_tsui_index = (scaled_value / init_scaled_val) * baseline_tsui
                else:
                    # 기준일이 0이고 둘째날도 거의 같으면, 둘째날은 약간 높게 설정
                    raw_tsui_index = baseline_tsui + SECOND_DAY_OFFSET
            else:
                raw_tsui_index = (scaled_value / baseline_scaled) * baseline_tsui
        
        results[date_str] = {
            'date': date_str,
            'raw_tsui_index': raw_tsui_index,  # EMA 적용 전 원본 지수
            'cumulative_energy': scaled_data['cumulative_energy'],
            'vol_proxy': scaled_data['vol_proxy'],  # 원본 vol_proxy (클리핑됨)
            'raw_value': scaled_data['raw_value'],  # S_cumulative * Vol_Proxy (정규화 안 함)
            'scaled_value': scaled_data['scaled'],  # ln(raw_value * LOG_SCALING_AMPLIFIER + 1)
            'baseline_scaled': baseline_scaled,  # Dynamic Baseline
            'etf_symbol': scaled_data['etf_symbol'],
            'sector': scaled_data['sector'],
            'baseline_energy': base_energy
        }
    
    return results


def apply_ema_smoothing(
    results: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    EMA 적용 (노이즈 및 등락폭 최적화 - 스무딩 강화)
    
    Args:
        results: {날짜: {raw_tsui_index, ...}}
    
    Returns:
        Dict: {날짜: {tsui_index, ...}} (EMA 적용된 값 추가)
    """
    sorted_date_list = sorted(results.keys())
    raw_tsui_list = [results[date_str]['raw_tsui_index'] for date_str in sorted_date_list]
    
    if len(raw_tsui_list) > 0:
        # pandas Series로 변환하여 EMA 계산 (span=EMA_SPAN으로 스무딩 강화)
        raw_tsui_series = pd.Series(raw_tsui_list)
        ema_tsui_list = raw_tsui_series.ewm(span=EMA_SPAN, adjust=False).mean().tolist()
        
        # EMA 적용된 값을 results에 저장 (기준일도 EMA 값으로 저장)
        for i, date_str in enumerate(sorted_date_list):
            results[date_str]['tsui_index'] = ema_tsui_list[i]
    else:
        # 데이터가 없으면 raw_tsui_index를 그대로 사용
        for date_str in results:
            results[date_str]['tsui_index'] = results[date_str]['raw_tsui_index']
    
    return results


def renormalize_after_ema(
    results: Dict[str, Dict[str, Any]],
    init_date_str: str,
    baseline_tsui: float
) -> Dict[str, Dict[str, Any]]:
    """
    EMA 계산 후 기준일의 tsui_index로 재정규화
    
    기준일(1/20)이 무조건 100.0000이 되도록 보정
    
    Args:
        results: {날짜: {tsui_index, ...}}
        init_date_str: 원래 기준일 (YYYY-MM-DD)
        baseline_tsui: 기준 TSUI 값 (100.0)
    
    Returns:
        Dict: 재정규화된 results
    """
    if init_date_str in results:
        base_ema_idx = results[init_date_str]['tsui_index']
        if base_ema_idx != 0.0 and base_ema_idx != baseline_tsui:
            # 모든 날짜의 tsui_index를 (오늘의 tsui_index / 기준일의 tsui_index) * 100으로 재정규화
            for date_str in results:
                results[date_str]['tsui_index'] = (results[date_str]['tsui_index'] / base_ema_idx) * baseline_tsui
        else:
            # 기준일의 tsui_index가 이미 100이거나 0이면 기준일만 100으로 설정
            results[init_date_str]['tsui_index'] = baseline_tsui
    else:
        # 기준일이 results에 없으면 모든 값을 baseline_tsui로 설정
        for date_str in results:
            results[date_str]['tsui_index'] = baseline_tsui
    
    # 최종 보정 - 기준일은 무조건 100.0000으로 설정
    results[init_date_str]['tsui_index'] = baseline_tsui
    
    return results


def calculate_cumulative_energy(
    daily_energies: Dict[str, float],
    intensity_levels: Dict[str, Optional[str]]
) -> Dict[str, float]:
    """
    누적 에너지 산출 (연속 일자 기반)
    
    공식: S_cumulative_t = S_raw_t + (S_cumulative_t-1 × α)
    
    발언이 없는 날(s_raw_t = 0.0)에도 전날의 누적 에너지에 alpha를 곱한 잔상이 적용됩니다.
    
    Args:
        daily_energies: {날짜(YYYY-MM-DD): 일일 원천 에너지} (발언 없는 날은 0.0)
        intensity_levels: {날짜(YYYY-MM-DD): 강도 레벨} (전날 기준, 발언 없는 날은 None)
    
    Returns:
        Dict: {날짜: 누적 에너지}
    """
    cumulative_energies = {}
    sorted_dates = sorted(daily_energies.keys())
    
    # 이전 날짜의 강도 레벨 추적 (발언이 없는 날은 이전 날짜의 강도 사용)
    prev_intensity = None
    
    for i, date_str in enumerate(sorted_dates):
        s_raw_t = daily_energies[date_str]
        
        # 전날 누적 에너지와 잔상 상수
        if i > 0:
            prev_date = sorted_dates[i - 1]
            s_cumulative_t_minus_1 = cumulative_energies.get(prev_date, 0.0)
            
            # 강도 레벨 결정: 현재 날짜에 발언이 있으면 그 날의 강도, 없으면 이전 날짜의 강도 사용
            intensity = intensity_levels.get(date_str)
            if intensity is None:
                # 발언이 없는 날: 이전 날짜의 강도 사용 (잔상 효과 유지)
                intensity = prev_intensity
            
            alpha = get_residue_alpha(intensity)
            
            # 발언이 없는 날(s_raw_t = 0.0)에도 잔상 효과 적용
            s_cumulative = s_raw_t + (s_cumulative_t_minus_1 * alpha)
            
            # 현재 날짜의 강도 업데이트 (다음 날짜를 위해)
            if intensity is not None:
                prev_intensity = intensity
        else:
            # 첫 날은 잔상 없음
            s_cumulative = s_raw_t
            # 첫 날의 강도 저장
            prev_intensity = intensity_levels.get(date_str)
        
        cumulative_energies[date_str] = s_cumulative
    
    return cumulative_energies

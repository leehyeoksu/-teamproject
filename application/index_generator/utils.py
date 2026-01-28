"""
TSUI 지수 산출에 사용되는 유틸리티 함수들

이 모듈은 날짜 파싱, 시간 가중치 계산, 키워드 가중치 계산,
Parkinson Volatility 계산 등 유틸리티 함수를 제공합니다.
"""

from datetime import datetime, timezone
from typing import Optional
import math


def parse_datetime(date_str: str) -> datetime:
    """
    날짜 문자열을 datetime 객체로 변환
    
    Args:
        date_str: 날짜 문자열 (예: "2017-01-20 10:43:23+00:00")
    
    Returns:
        datetime: UTC 타임존을 포함한 datetime 객체
    """
    try:
        # ISO 형식 파싱
        dt = datetime.fromisoformat(date_str.replace('+00:00', ''))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        # 다른 형식 시도
        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        raise ValueError(f"날짜 형식을 파싱할 수 없습니다: {date_str}")


def calculate_time_weight(dt: datetime) -> float:
    """
    발언 시각에 따른 시간 가중치 계산
    
    Args:
        dt: 발언 시각 (UTC)
    
    Returns:
        float: 시간 가중치 (장중: 1.0, 장외: 0.7)
    """
    # 함수 내부에서 import하여 순환 참조 방지
    from .data.constants import (
        MARKET_HOURS_START_UTC,
        MARKET_HOURS_END_UTC,
        TIME_WEIGHT_IN_MARKET,
        TIME_WEIGHT_OUT_MARKET
    )
    
    hour_utc = dt.hour
    
    # 장중: 14:30~21:00 UTC (09:30~16:00 EST)
    if MARKET_HOURS_START_UTC <= hour_utc < MARKET_HOURS_END_UTC:
        return TIME_WEIGHT_IN_MARKET
    else:
        return TIME_WEIGHT_OUT_MARKET


def calculate_keyword_weight(text: str, term: str = "term1") -> float:
    """
    텍스트에 키워드가 포함되어 있는지 확인하고 가중치 반환
    Volfefe 지수 방법론 참조: 1기/2기별로 다른 키워드 사전 사용
    
    Args:
        text: 분석할 텍스트
        term: 기간 ("term1" 또는 "term2")
    
    Returns:
        float: 키워드 가중치 (키워드 포함 시 2.0, 없으면 1.0)
    """
    # 함수 내부에서 import하여 순환 참조 방지
    from .data.constants import (
        VOLFEFE_KEYWORD_WEIGHT,
        KEYWORD_MAP_TERM1,
        KEYWORD_MAP_TERM2
    )
    
    text_lower = text.lower()
    
    # 기간에 따른 키워드 사전 선택
    keyword_map = KEYWORD_MAP_TERM2 if term == "term2" else KEYWORD_MAP_TERM1
    
    # global 키워드 확인
    for keyword in keyword_map["global"]:
        if keyword.lower() in text_lower:
            return VOLFEFE_KEYWORD_WEIGHT
    
    # intensity 키워드 확인
    for keyword in keyword_map["intensity"]:
        if keyword.lower() in text_lower:
            return VOLFEFE_KEYWORD_WEIGHT
    
    return 1.0


def calculate_parkinson_volatility(high: float, low: float) -> float:
    """
    Parkinson Volatility 계산
    
    공식: Vol_Proxy = sqrt((1/(4*ln(2))) * (ln(High) - ln(Low))^2)
    
    Args:
        high: 일일 최고가
        low: 일일 최저가
    
    Returns:
        float: Parkinson Volatility (보통 0.00~0.05 범위, 데이터 없을 때는 None 반환)
    """
    if high <= 0 or low <= 0 or high < low:
        return None  # 데이터 오류 시 None 반환 (기본값 1.0 사용 안 함)
    
    # high == low인 경우 (변동성 없음)
    if high == low:
        return 0.0001  # 0으로 나누기 방지를 위해 매우 작은 값 반환
    
    log_high = math.log(high)
    log_low = math.log(low)
    log_range = log_high - log_low
    
    volatility = math.sqrt((1.0 / (4.0 * math.log(2.0))) * (log_range ** 2))
    
    # 0이 되는 것을 방지
    if volatility == 0.0:
        return 0.0001
    
    return volatility


def get_residue_alpha(intensity_level: Optional[str]) -> float:
    """
    강도 레벨에 따른 잔상 상수 반환
    
    Args:
        intensity_level: 강도 레벨 ("high", "medium", "low", "HIGH", "MED", "LOW") 또는 None
    
    Returns:
        float: 잔상 상수 (기본값: 0.5)
    """
    # 함수 내부에서 import하여 순환 참조 방지
    from .data.constants import RESIDUE_ALPHA
    
    if intensity_level is None:
        return 0.5  # 기본값 (medium)
    
    return RESIDUE_ALPHA.get(intensity_level, RESIDUE_ALPHA.get(intensity_level.lower(), 0.5))

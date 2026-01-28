"""
데이터 가공 및 그룹화 헬퍼 함수

날짜 생성, 섹터 추출, Vol 매칭 등 데이터 가공 및 그룹화 함수들을 제공합니다.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, date, timezone
from collections import defaultdict
import numpy as np
import math

from ..utils import parse_datetime
from ..data.data_loader import (
    get_all_etf_data,
    get_dxy_data,
    get_period,
    get_etf_dir,
    get_volatility_data,
    get_daily_sector,
    get_track2_row_by_id,
    get_track2_data,
    extract_intensity_from_label_short
)
from ..data.constants import (
    BASELINE_SCALED_THRESHOLD,
    VOL_PROXY_INITIAL,
    VOL_PROXY_MIN,
    VOL_PROXY_MAX,
    TIME_WEIGHT_OUT_MARKET
)


# ============================================================================
# Vol_Proxy 계산 함수들
# ============================================================================

def calculate_single_date_vol_proxy(
    date_str: str,
    daily_sectors: Dict[str, Optional[str]],
    sector_to_etf: Dict[str, Dict[str, str]],
    fallback_to_one: bool = False
) -> tuple[Optional[float], str]:
    """
    단일 날짜에 대한 Vol_Proxy 계산 (공통 함수)
    
    Args:
        date_str: 날짜 문자열 (YYYY-MM-DD)
        daily_sectors: {날짜: 섹터명}
        sector_to_etf: {섹터: {symbol, etf_name}}
        fallback_to_one: 값이 없을 때 1.0을 보장할지 여부 (True면 None/0일 때 1.0 반환)
    
    Returns:
        tuple: (vol_proxy, etf_symbol)
        - fallback_to_one=True: vol_proxy는 항상 float (None 없음, 0이면 1.0)
        - fallback_to_one=False: vol_proxy는 Optional[float] (None 가능)
    """
    # 캐시에서 데이터 가져오기
    all_etf_data = get_all_etf_data() or {}
    dxy_data = get_dxy_data() or {}
    period = get_period()
    if period is None:
        raise ValueError("period가 설정되지 않았습니다. initialize_data()를 먼저 호출하세요.")
    etf_dir = get_etf_dir()
    if etf_dir is None:
        raise ValueError("etf_dir가 설정되지 않았습니다. initialize_data()를 먼저 호출하세요.")
    
    date_obj = datetime.fromisoformat(date_str).date()
    vol_proxy = None  # 기본값 없음 (데이터가 없으면 None)
    etf_symbol = "SPY"  # 기본값 (전체 시장)
    
    # 섹터 정보 확인
    sector = daily_sectors.get(date_str)
    
    if sector and sector in sector_to_etf:
        # 섹터 특정 시: 해당 섹터의 ETF 사용
        etf_symbol = sector_to_etf[sector]['symbol']
        vol_data = get_volatility_data(
            date_obj, etf_symbol, all_etf_data, dxy_data
        )
        
        if vol_data and 'high' in vol_data and 'low' in vol_data:
            high = vol_data['high']
            low = vol_data['low']
            vol_proxy = calculate_parkinson_volatility(high, low)
    else:
        # 섹터 불분명/전체 정책 시: SPY 또는 DXY 사용
        for market_symbol in ["SPY", "DXY"]:
            vol_data = get_volatility_data(
                date_obj, market_symbol, all_etf_data, dxy_data
            )
            if vol_data and 'high' in vol_data and 'low' in vol_data:
                etf_symbol = market_symbol
                high = vol_data['high']
                low = vol_data['low']
                vol_proxy = calculate_parkinson_volatility(high, low)
                
                # fallback_to_one이면 첫 번째로 찾은 값 사용 후 종료
                # 일반 날짜는 None 체크 후 break하여 유효한 값만 사용
                if fallback_to_one:
                    break  # baseline은 기준점이므로 첫 번째로 찾은 값 사용 후 종료
                elif vol_proxy is not None:  # None 체크 후 break (연속성 유지)
                    break
    
    # fallback_to_one 처리: 0이면 1.0으로 fallback, None은 없음
    if fallback_to_one:
        if vol_proxy is None or vol_proxy == 0.0:
            vol_proxy = 1.0
        return vol_proxy, etf_symbol
    
    # 일반 날짜: None 그대로 반환 (호출자가 prev_vol_proxy 사용)
    return vol_proxy, etf_symbol


def calculate_baseline_vol_proxy(
    base_date_str: str,
    daily_sectors: Dict[str, Optional[str]],
    sector_to_etf: Dict[str, Dict[str, str]]
) -> tuple[float, str]:
    """
    Baseline용 Vol_Proxy 계산
    
    주의: baseline은 지수의 시작점(기준점)이므로 일반 날짜와 별개로 취급
    - 기본값 1.0으로 fallback 있음 (None 체크 불필요)
    - 한 번만 계산하므로 break 후 추가 처리 없음
    
    Args:
        base_date_str: 정규화 기준일 (YYYY-MM-DD)
        daily_sectors: {날짜: 섹터명}
        sector_to_etf: {섹터: {symbol, etf_name}}
    
    Returns:
        tuple: (baseline_vol_proxy, baseline_etf_symbol)
    """
    return calculate_single_date_vol_proxy(
        base_date_str,
        daily_sectors,
        sector_to_etf,
        fallback_to_one=True
    )


def calculate_all_dates_vol_proxy(
    sorted_dates: List[str],
    daily_sectors: Dict[str, Optional[str]],
    sector_to_etf: Dict[str, Dict[str, str]]
) -> Dict[str, Dict[str, Any]]:
    """
    모든 날짜에 대해 Vol_Proxy 계산 및 클리핑
    
    주의: 일반 날짜는 매일 계산하며 연속성 유지 필요
    - vol_proxy가 None이면 prev_vol_proxy 사용 (연속성 유지)
    - None 체크 후 break하여 유효한 값만 사용
    
    Args:
        sorted_dates: 정렬된 날짜 리스트
        daily_sectors: {날짜: 섹터명}
        sector_to_etf: {섹터: {symbol, etf_name}}
    
    Returns:
        Dict: {날짜: {vol_proxy, etf_symbol, sector}}
    """
    vol_map = {}
    prev_vol_proxy = VOL_PROXY_INITIAL  # 직전 거래일 변동성 (초기값)
    
    for date_str in sorted_dates:
        # Vol_Proxy 계산 (섹터별 ETF 선택)
        vol_proxy, etf_symbol = calculate_single_date_vol_proxy(
            date_str,
            daily_sectors,
            sector_to_etf,
            fallback_to_one=False
        )
        
        # vol_proxy가 None이면 직전 거래일 값 사용 (연속성 유지)
        if vol_proxy is None:
            vol_proxy = prev_vol_proxy
        else:
            prev_vol_proxy = vol_proxy
        
        # 변동성 상한선 처리: VOL_PROXY_MIN ~ VOL_PROXY_MAX 범위로 클리핑 (데이터 오류 원천 차단)
        vol_proxy = np.clip(vol_proxy, VOL_PROXY_MIN, VOL_PROXY_MAX)
        prev_vol_proxy = vol_proxy  # 클리핑된 값으로 업데이트
        
        sector = daily_sectors.get(date_str)
        vol_map[date_str] = {
            'vol_proxy': vol_proxy,
            'etf_symbol': etf_symbol,
            'sector': sector or ''
        }
    
    return vol_map


def determine_baseline_scaled(
    sorted_dates: List[str],
    init_date_str: str,
    scaled_values: Dict[str, Dict[str, Any]]
) -> float:
    """
    Dynamic Baseline 설정
    
    기준일의 scaled_value가 0이거나 너무 작으면, 그 이후 처음으로 유의미한 값을 찾음
    단, 기준일의 raw_tsui_index는 항상 100으로 고정하므로,
    baseline_scaled는 기준일 이후 날짜들의 계산에만 사용
    
    Args:
        sorted_dates: 정렬된 날짜 리스트
        init_date_str: 원래 기준일 (YYYY-MM-DD)
        scaled_values: {날짜: {scaled, ...}}
    
    Returns:
        float: baseline_scaled 값
    """
    init_scaled_val = scaled_values.get(init_date_str, {}).get('scaled', 0.0)
    
    baseline_scaled = init_scaled_val
    if init_scaled_val < BASELINE_SCALED_THRESHOLD:
        # 기준일 이후 날짜에서 유의미한 값 찾기 (기준일 제외)
        for date_str in sorted_dates:
            if date_str <= init_date_str:
                continue  # 기준일 이전/당일은 스킵
            candidate_scaled = scaled_values.get(date_str, {}).get('scaled', 0.0)
            if candidate_scaled >= BASELINE_SCALED_THRESHOLD:
                baseline_scaled = candidate_scaled
                break
        
        # 그래도 유의미한 값이 없으면 첫 번째 0이 아닌 값 사용 (기준일 제외)
        if baseline_scaled < BASELINE_SCALED_THRESHOLD:
            for date_str in sorted_dates:
                if date_str <= init_date_str:
                    continue
                candidate_scaled = scaled_values.get(date_str, {}).get('scaled', 0.0)
                if candidate_scaled > 0.0:
                    baseline_scaled = candidate_scaled
                    break
        
        # 정말 모든 값이 0이면 기본값 사용
        if baseline_scaled == 0.0:
            baseline_scaled = 1.0
    
    return baseline_scaled


# ============================================================================
# TSUI 도메인 특화 계산 함수들
# ============================================================================

def calculate_time_weight(dt: datetime) -> float:
    """
    발언 시각에 따른 시간 가중치 계산
    
    Args:
        dt: 발언 시각 (UTC)
    
    Returns:
        float: 시간 가중치 (장중: 1.0, 장외: 0.7)
    """
    # 함수 내부에서 import하여 순환 참조 방지
    from ..data.constants import (
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
    from ..data.constants import (
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
    from ..data.constants import RESIDUE_ALPHA
    
    if intensity_level is None:
        return 0.5  # 기본값 (medium)
    
    return RESIDUE_ALPHA.get(intensity_level, RESIDUE_ALPHA.get(intensity_level.lower(), 0.5))


# ============================================================================
# 날짜 및 데이터 그룹화 함수들
# ============================================================================

def generate_all_dates(start_date: date, end_date: date) -> List[str]:
    """
    전체 기간 날짜 리스트 생성
    
    Args:
        start_date: 시작일
        end_date: 종료일
    
    Returns:
        List[str]: 날짜 문자열 리스트 (YYYY-MM-DD)
    """
    all_dates = []
    current_date = start_date
    while current_date <= end_date:
        all_dates.append(current_date.isoformat())
        current_date += timedelta(days=1)
    return all_dates


def group_posts_by_date(
    track1_data: List[Dict[str, Any]],
    baseline_date_only: date,
    track2_data: Dict[str, List[Dict[str, Any]]]
) -> tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Optional[str]], date, date, Dict[str, Any]]:
    """
    일별로 포스트 그룹화 및 전체 기간 생성
    
    Args:
        track1_data: Track 1 데이터 리스트
        baseline_date_only: 기준일 (date 객체)
        track2_data: Track 2 데이터 딕셔너리
    
    Returns:
        tuple: (daily_posts, daily_intensity, start_date, end_date, log_info)
        - log_info: 로그 출력용 정보 딕셔너리
    """
    daily_posts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    daily_intensity: Dict[str, Optional[str]] = {}
    
    # 시작일과 종료일 찾기
    start_date = None
    end_date = None
    
    filtered_count = 0
    for post in track1_data:
        date_str = post.get('date', '')
        try:
            dt = parse_datetime(date_str)
            date_only = dt.date()
            
            # 기준일 이전 데이터는 제외
            if date_only < baseline_date_only:
                filtered_count += 1
                continue
            
            # 시작일/종료일 업데이트
            if start_date is None or date_only < start_date:
                start_date = date_only
            if end_date is None or date_only > end_date:
                end_date = date_only
            
            date_str_iso = date_only.isoformat()
            daily_posts[date_str_iso].append(post)
            
            # 전날 강도 레벨 저장 (누적 에너지 계산용)
            post_id = post.get('id', '')
            track2_row = get_track2_row_by_id(post_id)
            
            if track2_row:
                # 새로운 형식: label_short에서 강도 추출
                label_short = track2_row.get('label_short', '')
                if label_short:
                    intensity = extract_intensity_from_label_short(label_short)
                else:
                    # 기존 형식: intensity_level 사용
                    intensity = track2_row.get('intensity_level', '')
                    if intensity:
                        intensity = intensity.upper()  # HIGH/MED/LOW로 통일
                
                if intensity:
                    if date_str_iso not in daily_intensity:
                        daily_intensity[date_str_iso] = intensity
                    # 같은 날 여러 포스트 중 가장 높은 강도 사용
                    elif intensity in ["HIGH", "high"]:
                        daily_intensity[date_str_iso] = intensity
                    elif intensity in ["MED", "medium"] and daily_intensity[date_str_iso] not in ["HIGH", "high"]:
                        daily_intensity[date_str_iso] = intensity
        except Exception:
            continue
    
    # 전체 기간 날짜 리스트 생성 (연속된 일자 기반)
    if start_date is None or end_date is None:
        # 데이터가 없으면 기준일만 사용
        start_date = baseline_date_only
        end_date = baseline_date_only
    
    all_dates = generate_all_dates(start_date, end_date)
    
    # 로그 정보 수집
    log_info = {
        'filtered_count': filtered_count,
        'days_with_posts': len(daily_posts),
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'total_days': len(all_dates)
    }
    
    return daily_posts, daily_intensity, start_date, end_date, log_info


def extract_daily_sectors(
    all_dates: List[str],
    daily_posts: Dict[str, List[Dict[str, Any]]],
    track2_data: Dict[str, List[Dict[str, Any]]]
) -> tuple[Dict[str, Optional[str]], Dict[str, Any]]:
    """
    일별 섹터 정보 추출 (Vol_Proxy 계산용)
    
    Args:
        all_dates: 전체 기간 날짜 리스트
        daily_posts: 일별 포스트 딕셔너리
        track2_data: Track 2 데이터 딕셔너리
    
    Returns:
        tuple: (daily_sectors, log_info)
        - daily_sectors: {날짜: 섹터명} (발언 없는 날은 None)
        - log_info: 로그 출력용 정보 딕셔너리
    """
    daily_sectors: Dict[str, Optional[str]] = {}
    
    # 발언이 있는 날만 섹터 추출
    for date_str in sorted(daily_posts.keys()):
        sector = get_daily_sector(date_str, daily_posts, track2_data)
        daily_sectors[date_str] = sector
    
    # 발언이 없는 날은 None으로 설정 (Vol_Proxy는 prev_vol_proxy 사용)
    for date_str in all_dates:
        if date_str not in daily_sectors:
            daily_sectors[date_str] = None
    
    sector_count = sum(1 for s in daily_sectors.values() if s)
    
    # 로그 정보 수집
    log_info = {
        'sector_specified_days': sector_count,
        'sector_unspecified_days': len(daily_sectors) - sector_count
    }
    
    return daily_sectors, log_info


def calculate_daily_raw_energy(
    posts: List[Dict[str, Any]],
    term: str = "term1",
    track2_data: Optional[Dict[str, Dict[str, Any]]] = None
) -> float:
    """
    일일 원천 에너지 산출
    
    공식: S_raw_t = sum(S_tone × S_topic × W_kw × W_time)
    
    Args:
        posts: 해당 날짜의 모든 포스트 리스트
        term: 기간 ("term1" 또는 "term2") - 키워드 사전 선택용
        track2_data: Track 2 데이터 딕셔너리 {post_id: {...}} (None이면 캐시에서 가져옴)
    
    Returns:
        float: 일일 원천 에너지
    """
    # track2_data가 없으면 캐시에서 가져옴
    if track2_data is None:
        track2_data = get_track2_data() or {}
    
    daily_energy = 0.0
    
    for post in posts:
        # S_tone: Track 1 감성점수
        sentiment_score = float(post.get('sentiment_score', 0.0))
        # 절댓값 사용 (negative도 강도로 간주)
        s_tone = abs(sentiment_score)
        
        # S_topic: Track 2 통합점수 (없으면 1.0)
        post_id = post.get('id', '')
        # 중복 id 처리를 위해 get_track2_row_by_id 사용
        track2_row = get_track2_row_by_id(post_id)
        if track2_row:
            # 새로운 형식: score 컬럼 사용, 없으면 integrated_score
            integrated_score = float(track2_row.get('score', track2_row.get('integrated_score', 1.0)))
        else:
            integrated_score = 1.0  # Track 2 없을 때 기본값
        
        # W_kw: 키워드 가중치 (기간별 키워드 사전 사용)
        text = post.get('text', '')
        w_kw = calculate_keyword_weight(text, term=term)
        
        # W_time: 시간 가중치
        date_str = post.get('date', '')
        try:
            dt = parse_datetime(date_str)
            w_time = calculate_time_weight(dt)
        except Exception:
            w_time = TIME_WEIGHT_OUT_MARKET  # 기본값
        
        # 일일 원천 에너지 누적
        energy = s_tone * integrated_score * w_kw * w_time
        daily_energy += energy
    
    return daily_energy


def calculate_daily_energies(
    all_dates: List[str],
    daily_posts: Dict[str, List[Dict[str, Any]]],
    term: str
) -> tuple[Dict[str, float], Dict[str, Any]]:
    """
    일일 원천 에너지 계산 (전체 기간에 대해, 발언이 없는 날은 0.0)
    
    Args:
        all_dates: 전체 기간 날짜 리스트
        daily_posts: 일별 포스트 딕셔너리
        term: 기간 ("term1" 또는 "term2")
    
    Returns:
        tuple: (daily_energies, log_info)
        - daily_energies: {날짜: 일일 원천 에너지}
        - log_info: 로그 출력용 정보 딕셔너리
    """
    daily_energies = {}
    for date_str in all_dates:
        if date_str in daily_posts:
            # 발언이 있는 날: 실제 에너지 계산
            posts = daily_posts[date_str]
            energy = calculate_daily_raw_energy(posts, term=term)
            daily_energies[date_str] = energy
        else:
            # 발언이 없는 날: 0.0 할당 (잔상 효과는 누적 에너지 계산에서 처리)
            daily_energies[date_str] = 0.0
    
    # 발언이 있는 날의 평균만 계산
    days_with_posts = [e for date_str, e in daily_energies.items() if date_str in daily_posts]
    avg_energy = sum(days_with_posts) / len(days_with_posts) if days_with_posts else 0.0
    
    # 로그 정보 수집
    log_info = {
        'avg_energy': avg_energy,
        'total_days': len(all_dates),
        'has_energy_data': len(days_with_posts) > 0
    }
    
    return daily_energies, log_info

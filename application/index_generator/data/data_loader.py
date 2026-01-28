"""
TSUI 지수 산출에 필요한 데이터 로딩 함수들

이 모듈은 Track 1, Track 2, ETF, DXY 데이터를 로드하는 함수들을 제공합니다.

로컬 캐싱을 통해 한 번 로드한 데이터를 재사용할 수 있습니다.

"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..utils import parse_datetime
from .constants import TRACK2_SECTOR_MAP

# ============================================================================
# 모듈 레벨 캐시 (로컬 캐싱)
# ============================================================================

_track1_data: Optional[List[Dict[str, Any]]] = None
_track2_data: Optional[Dict[str, List[Dict[str, Any]]]] = None
_track2_usage_count: Dict[str, int] = {}  # 각 id별 사용 횟수 추적 (중복 id 처리용)
_all_etf_data: Optional[Dict[str, Dict[datetime, Dict[str, float]]]] = None
_dxy_data: Optional[Dict[datetime, Dict[str, float]]] = None
_etf_metadata: Optional[Dict[str, Dict[str, str]]] = None
_etf_dir: Optional[Path] = None
_period: Optional[str] = None


def load_track1_data(csv_path: Path) -> List[Dict[str, Any]]:
    """
    Track 1 데이터 로드 (감성점수 포함)
    
    Args:
        csv_path: Track 1 CSV 파일 경로
    
    Returns:
        List[Dict]: 각 포스트의 데이터 리스트
    """
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_track2_data(
    csv_path: Optional[Path], 
    track1_data: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Track 2 데이터 로드 (통합점수 포함)
    
    중복 id를 처리하기 위해 리스트 형태로 저장: {id: [row1, row2, ...]}
    
    Args:
        csv_path: Track 2 CSV 파일 경로 (None이면 빈 딕셔너리 반환)
        track1_data: Track 1 데이터 리스트 (id 매칭용, 선택사항)
    
    Returns:
        Dict: {post_id: [{score, label, label_short, ...}, ...]}
    """
    if csv_path is None or not csv_path.exists():
        return {}
    
    track2_data: Dict[str, List[Dict[str, Any]]] = {}
    track2_rows = []
    
    # Track 2 데이터 읽기
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            track2_rows.append(row)
    
    # id 컬럼이 있는지 확인
    if track2_rows and 'id' in track2_rows[0]:
        # id 컬럼이 있으면 id로 매칭 (리스트 형태로 저장)
        for row in track2_rows:
            post_id = row.get('id', '')
            if post_id:
                if post_id not in track2_data:
                    track2_data[post_id] = []
                track2_data[post_id].append(row)
    elif track1_data:
        # id 컬럼이 없으면 Track 1과 행 순서로 매칭
        for i, row in enumerate(track2_rows):
            if i < len(track1_data):
                post_id = track1_data[i].get('id', '')
                if post_id:
                    if post_id not in track2_data:
                        track2_data[post_id] = []
                    track2_data[post_id].append(row)
    else:
        # Track 1 데이터가 없으면 인덱스를 id로 사용
        for i, row in enumerate(track2_rows):
            idx_str = str(i)
            if idx_str not in track2_data:
                track2_data[idx_str] = []
            track2_data[idx_str].append(row)
    
    return track2_data


def load_etf_metadata(etf_dir: Path) -> Dict[str, Dict[str, str]]:
    """
    ETF 메타데이터 로드 (섹터-ETF 매핑)
    
    Args:
        etf_dir: ETF 데이터 디렉토리
    
    Returns:
        Dict: {sector: {symbol, etf_name}}
    """
    metadata_path = etf_dir / 'etf_metadata.json'
    
    if not metadata_path.exists():
        return {}
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # 섹터별로 매핑 변환
    sector_to_etf = {}
    for symbol, info in metadata.items():
        sector = info.get('sector', '')
        if sector:
            sector_to_etf[sector] = {
                'symbol': symbol,
                'etf_name': info.get('etf_name', '')
            }
    
    return sector_to_etf


def load_all_etf_data(
    etf_dir: Path, 
    period: str
) -> Dict[str, Dict[datetime, Dict[str, float]]]:
    """
    모든 ETF 섹터 데이터를 기간별로 일괄 로드
    
    Args:
        etf_dir: ETF 데이터 디렉토리
        period: 기간 ("2017-2021" 또는 "2025-now")
    
    Returns:
        Dict: {symbol: {date: {high, low, close, ...}}}
    """
    all_etf_data = {}
    
    # etf_sector 폴더의 모든 ETF 파일 찾기
    pattern = f"etf_*_{period}.csv"
    etf_files = list(etf_dir.glob(pattern))
    
    for csv_path in etf_files:
        # 파일명에서 심볼 추출 (예: "etf_XLE_2017-2021.csv" -> "XLE")
        symbol = csv_path.stem.split('_')[1]
        
        etf_data = {}
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                date_str = row['Date']
                try:
                    dt = parse_datetime(date_str)
                    date_only = dt.date()
                    etf_data[date_only] = {
                        'high': float(row['High']),
                        'low': float(row['Low']),
                        'close': float(row['Close']),
                        'open': float(row['Open'])
                    }
                except Exception:
                    continue
        
        if etf_data:
            all_etf_data[symbol] = etf_data
    
    return all_etf_data


def load_dxy_data(dxy_path: Path) -> Dict[datetime, Dict[str, float]]:
    """
    DXY (달러 인덱스) 데이터 로드
    
    Args:
        dxy_path: DXY CSV 파일 경로
    
    Returns:
        Dict: {date: {high, low, close, ...}}
    """
    if not dxy_path.exists():
        return {}
    
    dxy_data = {}
    with open(dxy_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            date_str = row.get('Date') or row.get('date_only', '')
            try:
                dt = parse_datetime(date_str)
                date_only = dt.date()
                dxy_data[date_only] = {
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'open': float(row['Open'])
                }
            except Exception:
                continue
    
    return dxy_data


def load_etf_data(
    etf_dir: Path, 
    symbol: str, 
    period: str
) -> Dict[datetime, Dict[str, float]]:
    """
    특정 ETF 데이터 로드 (하위 호환성 유지)
    
    Args:
        etf_dir: ETF 데이터 디렉토리
        symbol: ETF 심볼 (예: "XLE", "DXY")
        period: 기간 ("2017-2021" 또는 "2025-now", DXY는 무시)
    
    Returns:
        Dict: {date: {high, low, close, ...}}
    """
    # DXY는 별도 파일
    if symbol == "DXY":
        dxy_path = etf_dir.parent / "DXY.csv"
        return load_dxy_data(dxy_path)
    
    # ETF 섹터 파일
    csv_path = etf_dir / f"etf_{symbol}_{period}.csv"
    
    if not csv_path.exists():
        return {}
    
    etf_data = {}
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            date_str = row['Date']
            try:
                dt = parse_datetime(date_str)
                date_only = dt.date()
                etf_data[date_only] = {
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'open': float(row['Open'])
                }
            except Exception:
                continue
    
    return etf_data


def get_volatility_data(
    date_obj: datetime.date,
    symbol: str,
    all_etf_data: Dict[str, Dict[datetime, Dict[str, float]]],
    dxy_data: Dict[datetime, Dict[str, float]]
) -> Optional[Dict[str, float]]:
    """
    날짜와 심볼로 Volatility 계산에 필요한 데이터 조회
    
    Args:
        date_obj: 날짜
        symbol: ETF 심볼 (예: "XLE", "SPY", "DXY")
        all_etf_data: 모든 ETF 데이터 {symbol: {date: {...}}}
        dxy_data: DXY 데이터 {date: {...}}
    
    Returns:
        Optional[Dict]: {high, low, close, open} 또는 None
    """
    # DXY 처리
    if symbol == "DXY":
        return dxy_data.get(date_obj)
    
    # SPY 처리 (파일이 없으면 None 반환)
    if symbol == "SPY":
        spy_data = all_etf_data.get("SPY")
        if spy_data:
            return spy_data.get(date_obj)
        return None
    
    # 일반 ETF 섹터 처리
    etf_data = all_etf_data.get(symbol)
    if etf_data:
        return etf_data.get(date_obj)
    
    return None


def extract_sector_from_label_short(label_short: str) -> Optional[str]:
    """
    label_short에서 섹터명 추출
    
    Args:
        label_short: Track 2의 label_short (예: "MED_CONS_DISC", "HIGH_ENERGY")
    
    Returns:
        Optional[str]: 섹터명 (예: "consumer_discretionary", "energy")
    """
    if not label_short:
        return None
    
    # 형식: "MED_CONS_DISC" -> "CONS_DISC" 추출
    parts = label_short.split('_')
    if len(parts) < 2:
        return None
    
    # 첫 번째는 강도 (HIGH/MED/LOW), 나머지는 섹터
    sector_short = '_'.join(parts[1:])
    
    # short_label을 full sector name으로 변환
    return TRACK2_SECTOR_MAP.get(sector_short)


def extract_intensity_from_label_short(label_short: str) -> Optional[str]:
    """
    label_short에서 강도 레벨 추출
    
    Args:
        label_short: Track 2의 label_short (예: "MED_CONS_DISC", "HIGH_ENERGY")
    
    Returns:
        Optional[str]: 강도 레벨 ("HIGH", "MED", "LOW")
    """
    if not label_short:
        return None
    
    parts = label_short.split('_')
    if len(parts) < 1:
        return None
    
    intensity = parts[0]
    # 소문자로 들어올 수도 있으므로 대문자로 변환
    return intensity.upper() if intensity in ["HIGH", "MED", "LOW"] else None


def get_daily_sector(
    date_str: str,
    daily_posts: Dict[str, List[Dict[str, Any]]],
    track2_data: Optional[Dict[str, List[Dict[str, Any]]]] = None
) -> Optional[str]:
    """
    일별로 가장 높은 점수의 섹터 추출
    
    Track 2의 새로운 형식 지원:
    - label_short 컬럼 사용 (예: "MED_CONS_DISC")
    - 또는 기존 형식도 지원 (sector_label, integrated_score)
    
    중복 id 처리를 위해 get_track2_row_by_id를 사용합니다.
    
    Args:
        date_str: 날짜 (YYYY-MM-DD)
        daily_posts: 일별 포스트 딕셔너리
        track2_data: Track 2 데이터 딕셔너리 (사용하지 않음, 하위 호환성용)
    
    Returns:
        Optional[str]: 섹터명 (없으면 None)
    """
    posts = daily_posts.get(date_str, [])
    if not posts:
        return None
    
    # 해당 날짜의 모든 포스트에서 섹터 정보 추출
    sector_scores = {}
    
    for post in posts:
        post_id = post.get('id', '')
        track2_row = get_track2_row_by_id(post_id)
        
        if not track2_row:
            continue
        
        # 새로운 형식: label_short 사용
        label_short = track2_row.get('label_short', '')
        if label_short:
            sector = extract_sector_from_label_short(label_short)
            if sector:
                # score 또는 integrated_score 사용
                score = float(track2_row.get('score', track2_row.get('integrated_score', 0.0)))
                if sector not in sector_scores:
                    sector_scores[sector] = 0.0
                sector_scores[sector] += score
            continue
        
        # 기존 형식: sector_label 사용 (하위 호환성)
        sector_label = track2_row.get('sector_label', '')
        if sector_label:
            # 섹터명 추출 (예: "high energy impact" -> "energy")
            sector = sector_label
            if ' ' in sector_label:
                parts = sector_label.split()
                if len(parts) >= 2:
                    sector = parts[1]
            
            integrated_score = float(track2_row.get('integrated_score', 0.0))
            if sector not in sector_scores:
                sector_scores[sector] = 0.0
            sector_scores[sector] += integrated_score
    
    # 가장 높은 점수의 섹터 반환
    if sector_scores:
        return max(sector_scores.items(), key=lambda x: x[1])[0]
    
    return None


# ============================================================================
# 데이터 초기화 및 Getter 함수 (로컬 캐싱)
# ============================================================================

def initialize_data(
    track1_csv: Path,
    track2_csv: Optional[Path],
    etf_dir: Path,
    period: str
) -> None:
    """
    모든 데이터를 로드하여 모듈 레벨 캐시에 저장
    
    Args:
        track1_csv: Track 1 CSV 파일 경로
        track2_csv: Track 2 CSV 파일 경로 (None 가능)
        etf_dir: ETF 데이터 디렉토리
        period: 기간 ("2017-2021" 또는 "2025-now")
    """
    global _track1_data, _track2_data, _all_etf_data, _dxy_data, _etf_metadata, _etf_dir, _period, _track2_usage_count
    
    # Track 1 데이터 로드
    _track1_data = load_track1_data(track1_csv)
    
    # Track 2 데이터 로드 (Track 1 데이터를 사용하여 id 매칭)
    _track2_data = load_track2_data(track2_csv, track1_data=_track1_data) if track2_csv else {}
    # 사용 횟수 초기화 (중복 id 처리용)
    _track2_usage_count = {}
    
    # ETF 및 DXY 데이터 로드
    _all_etf_data = load_all_etf_data(etf_dir, period)
    dxy_path = etf_dir.parent / "DXY.csv"
    _dxy_data = load_dxy_data(dxy_path)
    
    # ETF 메타데이터 로드
    _etf_metadata = load_etf_metadata(etf_dir)
    
    # 설정 저장
    _etf_dir = etf_dir
    _period = period


def get_track1_data() -> Optional[List[Dict[str, Any]]]:
    """캐시된 Track 1 데이터 반환"""
    return _track1_data


def get_track2_data() -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """캐시된 Track 2 데이터 반환 (리스트 형태)"""
    return _track2_data


def get_track2_row_by_id(post_id: str) -> Optional[Dict[str, Any]]:
    """
    Track 2 데이터에서 id로 행을 가져옴 (중복 id 처리)
    
    중복 id가 있으면 순차적으로 반환합니다.
    
    Args:
        post_id: 포스트 id
    
    Returns:
        Dict: Track 2 행 데이터 (없으면 None)
    """
    global _track2_data, _track2_usage_count
    
    if not _track2_data or post_id not in _track2_data:
        return None
    
    rows = _track2_data[post_id]
    if not rows:
        return None
    
    # 사용 횟수 초기화 (없으면)
    if post_id not in _track2_usage_count:
        _track2_usage_count[post_id] = 0
    
    # 현재 인덱스
    current_index = _track2_usage_count[post_id]
    
    # 인덱스가 범위를 벗어나면 마지막 항목 반환
    if current_index >= len(rows):
        return rows[-1]
    
    # 현재 항목 반환하고 사용 횟수 증가
    result = rows[current_index]
    _track2_usage_count[post_id] = current_index + 1
    
    return result


def get_all_etf_data() -> Optional[Dict[str, Dict[datetime, Dict[str, float]]]]:
    """캐시된 모든 ETF 데이터 반환"""
    return _all_etf_data


def get_dxy_data() -> Optional[Dict[datetime, Dict[str, float]]]:
    """캐시된 DXY 데이터 반환"""
    return _dxy_data


def get_etf_metadata() -> Optional[Dict[str, Dict[str, str]]]:
    """캐시된 ETF 메타데이터 반환"""
    return _etf_metadata


def get_etf_dir() -> Optional[Path]:
    """캐시된 ETF 디렉토리 경로 반환"""
    return _etf_dir


def get_period() -> Optional[str]:
    """캐시된 기간 반환"""
    return _period


def clear_cache() -> None:
    """캐시 초기화 (테스트용)"""
    global _track1_data, _track2_data, _all_etf_data, _dxy_data, _etf_metadata, _etf_dir, _period, _track2_usage_count
    _track1_data = None
    _track2_data = None
    _all_etf_data = None
    _dxy_data = None
    _etf_metadata = None
    _etf_dir = None
    _period = None
    _track2_usage_count = {}

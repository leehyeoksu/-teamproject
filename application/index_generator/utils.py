"""
공통 유틸리티 함수

이 모듈은 도메인 독립적인 공통 유틸리티 함수를 제공합니다.
"""

from datetime import datetime, timezone


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

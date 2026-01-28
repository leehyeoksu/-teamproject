"""
TSUI 지수 산출에 사용되는 상수 정의

이 모듈은 트럼프 지수 산출에 필요한 모든 상수를 정의합니다.
"""

from datetime import datetime, timezone

# ============================================================================
# 트럼프 재임 기간
# ============================================================================

TRUMP_FIRST_TERM_START = datetime(2017, 1, 20, tzinfo=timezone.utc)
TRUMP_FIRST_TERM_END = datetime(2021, 1, 20, tzinfo=timezone.utc)
TRUMP_SECOND_TERM_START = datetime(2025, 1, 20, tzinfo=timezone.utc)

# 기준점 (지수 100으로 설정)
BASELINE_DATE_TERM1 = TRUMP_FIRST_TERM_START
BASELINE_DATE_TERM2 = TRUMP_SECOND_TERM_START

# ============================================================================
# 잔상 상수 (α) - 강도별
# ============================================================================
# Track 2의 새로운 명칭에 맞춰 수정: HIGH/MED/LOW
# 어제의 에너지가 오늘까지 남는 비율 (감쇄상수)
RESIDUE_ALPHA = {
    "high": 0.7,
    "medium": 0.5,
    "low": 0.2,
    "HIGH": 0.7,  # 새로운 형식
    "MED": 0.5,   # 새로운 형식
    "LOW": 0.2   # 새로운 형식
}

# ============================================================================
# 시간 가중치
# ============================================================================
# 장중: 09:30~16:00 EST (14:30~21:00 UTC)
# 장외: 그 외 시간
MARKET_HOURS_START_UTC = 14  # 09:30 EST = 14:30 UTC
MARKET_HOURS_END_UTC = 21    # 16:00 EST = 21:00 UTC
TIME_WEIGHT_IN_MARKET = 1.0
TIME_WEIGHT_OUT_MARKET = 0.7

# ============================================================================
# 키워드 가중치 (W_kw)
# ============================================================================
# Volfefe 지수 방법론 참조: JP모건의 Volfefe 지수에서 사용하는 키워드 가중치 방법론 계승
# 시장 전체를 흔드는 핵심 키워드 포함 시 x2.0 스파이크 처리
# 공식에서 사용: S_raw_t = sum(S_tone × S_topic × W_kw × W_time)
# 참조: trump_index_methodology.md 섹션 6 (키워드 가중치 전략)
VOLFEFE_KEYWORD_WEIGHT = 2.0

# ============================================================================
# 키워드 사전 (1기/2기 분리)
# ============================================================================
# 1기와 2기는 정책 테마가 다르므로 키워드도 구분하여 사용
# 참조: trump_index_methodology.md 섹션 6.1 (1기/2기 핵심 키워드)

KEYWORD_MAP_TERM1 = {
    "global": [
        # 시장 전체 키워드 (Volfefe 계승)
        "tariff", "trade war", "china", "mexico", "canada", "eu", "billion dollars",
        # 섹터별 키워드
        "steel", "aluminum", "section 232",  # 철강/알루미늄
        "chips", "tsmc", "export controls",  # 반도체
        "soybeans", "agriculture",  # 농업
        "auto", "nafta", "usmca"  # 자동차
    ],
    "intensity": [
        # 강도 부스터 키워드 (트럼프 특유의 화법)
        "massive", "huge", "disaster", "catastrophic"
    ]
}

KEYWORD_MAP_TERM2 = {
    "global": [
        # 신규 테마 키워드
        "ai", "artificial intelligence", "crypto", "digital assets", 
        "de-risking", "economic security",
        # 강화된 섹터 키워드
        "advanced ai chips", "nvidia", "h200", "tsmc",  # 고급 반도체
        "oil", "drill", "fracking", "lng",  # 에너지
        # 정책 수단
        "section 232", "ieepa"
    ],
    "intensity": [
        # 강도 부스터 키워드 (1기와 동일)
        "massive", "huge", "disaster", "catastrophic"
    ]
}

# ============================================================================
# Track 2 섹터명 매핑
# ============================================================================
# short_label -> full sector name
# methodology.md의 SECTOR_MAP 참조
TRACK2_SECTOR_MAP = {
    "ENERGY": "energy",
    "MATR": "materials",
    "CONS_DISC": "consumer_discretionary",
    "HEALTH": "healthcare",
    "INDU": "industrial",
    "TECH": "technology",
    "COMM": "communication",
    "FINANCE": "financial",
    "RE_EST": "real_estate",
    "UTIL": "utilities",
    "CONS_STAP": "consumer_staples",
    "AERO_DEF": "defense"  # "Aerospace & Defense" -> "defense"
}

# ============================================================================
# TSUI 지수 계산 상수
# ============================================================================

# Vol_Proxy 관련 상수
VOL_PROXY_INITIAL = 0.01  # 직전 거래일 변동성 초기값
VOL_PROXY_MIN = 0.001  # 변동성 하한선 (데이터 오류 원천 차단)
VOL_PROXY_MAX = 0.05  # 변동성 상한선 (데이터 오류 원천 차단)

# 로그 스케일링 관련 상수
LOG_SCALING_AMPLIFIER = 500  # 로그 스케일링 증폭 계수 (고점에서의 상승 기울기 완만하게 조정)

# Baseline 관련 상수
BASELINE_SCALED_THRESHOLD = 0.1  # baseline_scaled 유의미한 값 판단 임계값
SCALED_VALUE_EPSILON = 0.0001  # 부동소수점 오차 임계값 (baseline_scaled와의 차이 비교용)
SECOND_DAY_OFFSET = 0.01  # 기준일이 0이고 둘째날도 거의 같을 때 둘째날 오프셋

# EMA 관련 상수
EMA_SPAN = 5  # Exponential Moving Average span (스무딩 강화)

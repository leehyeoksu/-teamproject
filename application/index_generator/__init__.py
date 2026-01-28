"""
TSUI 지수 산출 패키지

이 패키지는 트럼프 지수(TSUI) 산출에 필요한 모든 모듈을 제공합니다.
"""

__version__ = "1.0.0"

# 파이프라인에서 메인 함수 export
from .pipeline import run_analysis

__all__ = [
    'run_analysis',
]

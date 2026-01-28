"""
트럼프 지수(TSUI) 산출 스크립트

이 스크립트는 Track 1 (감성점수)과 Track 2 (통합점수) 데이터를 기반으로
TSUI (Trump Statement Uncertainty Index) 지수를 산출합니다.

사용법:
    python -m index_generator.main
    또는
    cd index_generator && python main.py

지수 산출 공식:
1. 일일 원천 에너지: S_raw_t = sum(S_tone × S_topic × W_kw × W_time)
2. 누적 에너지: S_cumulative_t = S_raw_t + (S_raw_t-1 × α)
3. 최종 지수: TSUI_t = S_cumulative_t × Vol_Proxy_t

주의사항:
- Track 2 데이터가 없으면 S_topic을 1.0으로 가정하여 계산합니다.
- ETF 데이터가 없으면 Vol_Proxy를 1.0으로 가정하여 계산합니다.
"""

from pathlib import Path

from .pipeline import run_analysis


# ============================================================================
# 메인 실행부
# ============================================================================

def main():
    """
    메인 실행 함수
    """
    # 스크립트 위치: index_generator/main.py
    # 데이터 위치: data/src_data/ (index_generator의 하위 폴더)
    base_dir = Path(__file__).parent  # index_generator
    data_dir = base_dir / 'data' / 'src_data'
    etf_dir = data_dir / 'etf_sector'
    
    # ============================================================================
    # 경로 설정 (static)
    # ============================================================================
    
    # Track 1 데이터 경로
    TRACK1_TERM1 = data_dir / 'track_01' / 'trump_posts_term1_2017_2021_with_sentiment.csv'
    TRACK1_TERM2 = data_dir / 'track_01' / 'trump_posts_term2_2025_with_sentiment.csv'
    
    # Track 2 데이터 경로 (id 컬럼이 있는 final.csv 사용)
    TRACK2_FILE = data_dir / 'track_02' / 'trump_posts_36categori_final.csv'
    
    # 출력 파일 경로
    OUTPUT_TERM1 = data_dir / 'tsui_index_term1_2017-2021.csv'
    OUTPUT_TERM2 = data_dir / 'tsui_index_term2_2025-now.csv'
    
    # ============================================================================
    # 1기 지수 산출
    # ============================================================================
    print("=" * 60)
    print("1기 (2017-2021) TSUI 지수 산출")
    print("=" * 60)
    
    # Track 2 파일은 1기/2기 통합 파일 사용
    track2_csv = TRACK2_FILE if TRACK2_FILE.exists() else None
    
    try:
        run_analysis(
            track1_csv=TRACK1_TERM1,
            track2_csv=track2_csv,
            etf_dir=etf_dir,
            output_csv=OUTPUT_TERM1,
            term="term1"
        )
        print(f"\n✅ 1기 완료: {OUTPUT_TERM1}")
    except Exception as e:
        print(f"\n❌ 1기 오류 발생: {type(e).__name__}: {e}")
        raise
    
    # ============================================================================
    # 2기 지수 산출
    # ============================================================================
    print("\n" + "=" * 60)
    print("2기 (2025-now) TSUI 지수 산출")
    print("=" * 60)
    
    # Track 2 파일은 1기/2기 통합 파일 사용
    track2_csv = TRACK2_FILE if TRACK2_FILE.exists() else None
    
    try:
        run_analysis(
            track1_csv=TRACK1_TERM2,
            track2_csv=track2_csv,
            etf_dir=etf_dir,
            output_csv=OUTPUT_TERM2,
            term="term2"
        )
        print(f"\n✅ 2기 완료: {OUTPUT_TERM2}")
    except Exception as e:
        print(f"\n❌ 2기 오류 발생: {type(e).__name__}: {e}")
        raise
    
    print("\n" + "=" * 60)
    print("전체 지수 산출 완료!")
    print("=" * 60)


if __name__ == '__main__':
    main()

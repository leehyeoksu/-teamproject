# 📑 [Final Blueprint] 트럼프 지수(TSUI) 산출 및 검증 가이드

## 1. 프로젝트 개요
* **지수 명칭**: **TSUI** (Trump Statement Uncertainty Index)
* **산출 철학**: "트럼프의 발언 강도가 시장의 내재 변동성을 얼마나 자극했는가"를 수학적으로 도출.
* **기준점**: 2017.01.20 (1기 임기 시작일)을 지수 **100**으로 고정.

---

## 2. 데이터 아키텍처 및 분석 트랙

### **Track 1: 발언 강도 분석 (Intensity)**
* **모델**: `yiyanghkust/finbert-tone`
* **역할**: 발언의 매파적(Hawkish/Negative) 강도를 $0 \sim 1$ 사이의 값($S_{tone}$)으로 산출.

### **Track 2: 섹터 및 강도 통합 분석 (Targeting)**
* **모델**: `facebook/bart-large-mnli` (Zero-shot)
* **로직**: 12개 섹터 $\times$ 3개 강도(Low, Medium, High) = **36개 카테고리** 교차 분류.
```python
SECTORS = [
    "Energy", "Materials", "Consumer Discretionary", "Health Care",
    "Industrials", "Technology", "Communication Services", "Financials",
    "Real Estate", "Utilities", "Consumer Staples", "Aerospace & Defense"
]

# 모델이 '의도'를 파악하기 쉽게 더 구체적인 표현으로 변경
LEVELS = {
    "HIGH": "a direct, immediate, and concrete action against the {} sector",
    "MEDIUM": "a strategic warning or potential pressure on the {} sector",
    "LOW": "a minor mention or general opinion about the {} sector"
}
for sector in SECTORS:
    for k, phrase in LEVELS.items():
        label = phrase.format(sector) # {} 자리에 섹터명 주입
        key = f"{k}_{sector.replace(' ', '_').upper()}"
        MODEL_LABELS.append(label)
        LABEL_KEY_MAP[label] = key
        
  # 1. 섹터명 단축 맵 (가독성 최우선)
SECTOR_MAP = {
    "Energy": "ENERGY",
    "Materials": "MATR",
    "Consumer Discretionary": "CONS_DISC",
    "Health Care": "HEALTH",
    "Industrials": "INDU",
    "Technology": "TECH",
    "Communication Services": "COMM",
    "Financials": "FINANCE",
    "Real Estate": "RE_EST",
    "Utilities": "UTIL",
    "Consumer Staples": "CONS_STAP",
    "Aerospace & Defense": "AERO_DEF"
}

# 2. 강도 단축 맵
LEVEL_MAP = {
    "a direct, immediate, and concrete action against the": "HIGH",
    "a strategic warning or potential pressure on the": "MED",
    "a minor mention or general opinion about the": "LOW"
}

# 3. 36개 전체 라벨에 대한 매핑 딕셔너리 자동 생성
full_label_map = {}
for long_prefix, short_prefix in LEVEL_MAP.items():
    for long_sector, short_sector in SECTOR_MAP.items():
        # 원본 라벨 형태: "a strategic warning ... on the Energy sector"
        long_label = f"{long_prefix} {long_sector} sector"
        short_label = f"{short_prefix}_{short_sector}"
        full_label_map[long_label] = short_label
```        
* **강도 가중치 ($W_{int}$)**:
    - **HIGH (Direct/Concrete Action)**: 가중치 1.0 / 잔상상수 0.7
    - **MED (Strategic Warning)**: 가중치 0.6 / 잔상상수 0.5
    - **LOW (Minor Mention)**: 가중치 0.2 / 잔상상수 0.2
* **산출**: 36개 조합 중 가장 높은 확률값과 강도 가중치를 결합하여 섹터별 영향력($S_{topic}$) 도출.

---

## 3. 통계적 지수 산출 수식 ($TSUI_t$)

JP모건의 **Volfefe 방법론**과 **Parkinson 변동성 이론**에 **시간 감쇄 누적 모델**을 결합하여, 발언의 즉각적인 충격뿐만 아니라 시장에 머무는 잔상 효과까지 수치화합니다.

본 지수는 트럼프 대통령 발언의 단순 빈도가 아니라, 시장이 느끼는 **'실질적 공포의 총량'**을 정량화하는 데 목적이 있습니다.

### **Step 1: 일일 원천 에너지 산출 ($S_{raw, t}$)**
그날 발생한 모든 발언의 개별 점수를 합산하여 당일의 순수 에너지를 먼저 계산합니다.
* **공식**: $S_{raw, t} = \sum_{i=1}^{n} (S_{tone, i} \times S_{topic, i} \times W_{kw, i} \times W_{time, i})$

* **설계 근거 (Why?)**: 
    * **다각적 영향력 평가**: 단순 텍스트 분석을 넘어 감성(Tone), 주제 중요도(Topic), 핵심 키워드(Keyword)를 결합하여 발언 하나가 가진 **물리적인 충격량**을 계산합니다.
    * **시장 반응 시차 반영($W_{time}$)**: 미 증시 장중 발언은 즉각 반영되지만, 장외 발언은 익일 시가 변동성으로 전이됩니다. 이를 반영해 시간대별 가중치를 차등화하여 **현실적인 시장 반영도**를 구현했습니다.

* **데이터 정합성 (Sequential Cursor Matching)**: 
    - 동일 시간대 중복 ID 포스트의 누락을 방지하기 위해 **순차적 1:1 매칭(Cursor)** 방식을 채택함. 
    - 이를 통해 트럼프 특유의 연사형(連射) 발언 에너지를 데이터 손실 없이 100% 반영함.

### **Step 2: 누적 에너지 산출 ($S_{cumulative, t}$)**

당일 에너지에 과거 에너지가 복리로 누적된 잔상을 더하여 리스크의 '연속성'을 구현합니다.

* **공식**: $S_{cumulative, t} = S_{raw, t} + (S_{cumulative, t-1} \times \alpha)$

* **설계 근거 (Why?)**: 
    - **시계열적 복리 효과**: 오늘의 심리는 어제의 충격 위에 쌓인다는 전제하에, 단순 잔상이 아닌 **재귀적(Recursive) 누적** 방식을 채택함.
    - **공백기 연속성**: 발언이 없는 날($S_{raw,t}=0$)에도 전일 에너지가 $\alpha$ 비율로 지수 감쇄(Decay)하며 흐르게 하여, 리스크 지표의 시계열적 단절을 방지함.

### **Step 3: 최종 지수 확정 ($TSUI_t$)**
누적된 에너지에 시장의 실제 전도율인 **Parkinson Volatility**를 곱하여 최종 지수를 도출합니다.
* **공식**: $TSUI_t = S_{cumulative, t} \times \text{Vol\_Proxy}_t$

* **설계 근거 (Why?)**: 
    * **실제 시장 전도율 반영**: 발언이 아무리 강력해도 시장이 무덤덤하면 리스크가 낮고, 반대로 시장이 예민할 때 발언이 터지면 파괴력은 배가됩니다.
    * **Parkinson Volatility 채택**: 종가 기준 변동성과 달리 고가(High)와 저가(Low)를 활용하는 파킨슨 방식을 사용하여, 장중에 발생한 극심한 공포와 등락을 포착합니다. 이는 **트럼프 리스크 특유의 장중 변동성**을 측정하기에 최적의 도구입니다.

### **핵심 변수 정의**
1. **키워드 가중치 ($W_{kw}$)**: Volfefe 핵심 키워드(`Tariff`, `China`, `Fed` 등) 포함 시 **x2.0** 스파이크 처리.
2. **시간 가중치 ($W_{time}$)**: 발언 시각에 따른 시장 반영도 차등화.
    * **장중 (09:30~16:00 EST)**: 1.0 (즉각 반영).
    * **장외 (16:00~09:30 EST)**: 0.7 (익일 시가 변동성 전이 가중치).
3. **변동성 프록시 ($\text{Vol\_Proxy}_t$)**: **Parkinson Volatility** 활용.
    * 데이터 대상(Source):
        - 섹터 특정 시: Track 2에서 분류된 해당 섹터의 대표 ETF(예: $SOXX, XLE$ 등)의 $H/L$ 사용.
        - 섹터 불분명/전체 정책 시: 시장 전체 지수인 S&P 500($SPY$) 또는 **달러 인덱스($DXY$)**의 $H/L$ 사용.
    * 장중 고가(High)와 저가(Low)의 진폭을 측정하여 종가에 가려진 공포를 포착.
    $$\text{Vol\_Proxy} = \sqrt{\frac{1}{4\ln2}( \ln(High) - \ln(Low) )^2}$$


## 4. 지수 엔진 안정화 및 고도화 (Optimization Detail)

초기 산출 공식($S \times Vol$)을 실제 1,700여 개의 시계열 데이터에 적용한 결과, 이론적으로 예측하지 못한 세 가지 기술적 결함이 발견되었습니다. 이를 해결하기 위해 적용된 보정 로직의 근거는 다음과 같습니다.

### **1) 지수 폭주 방지: 로그 압축 및 해상도 최적화 (Log-Scaling with Sensitivity Tuning)**

* **발견된 문제 (Problem)**: 
    * 2020년 '코로나 쇼크'와 같이 에너지($S$)와 변동성($V$)이 동시 폭발하는 구간에서, 초기 산출 방식은 지수를 **30,000~50,000 포인트 이상**으로 무한정 발산시켰습니다.
    * 이로 인해 일상적인 변동 구간(100~300)의 데이터가 그래프 바닥에 붙어버려, 리스크 변화를 식별할 수 없는 **통계적 착시** 현상이 발생했습니다.변별력을 상실함.
* **보정 방법 (Solution)**: 
    * 지수의 비선형 압축을 위해 로그 함수를 도입하되, 데이터의 해상도 유지를 위한 증폭 계수를 결합했습니다.
    * 공식: $V_{scaled, t} = \ln( (S_{cumulative, t} \times \text{Vol\_Proxy}_t) \times 500 + 1 )$
* **채택 근거 및 500의 의미 (Why 500?)**: 
    * **데이터 해상도(Resolution) 확보**: 원본 에너지 값($S \times V$)은 평상시 0.001 단위의 매우 작은 소수점으로 움직입니다. 증폭 계수 없이 로그를 씌우면 지수가 거의 변하지 않는 '반응성 저하'가 발생하므로, **500배의 선제적 증폭**을 통해 미세한 변화를 유의미한 정수 단위 변동으로 끌어올렸습니다.
    * **체감형 지표 구현**: 수천 번의 백테스팅 결과, 500은 평상시 지수를 100~400대, 피크 시를 700대 내외로 안착시키는 **가장 직관적인 배율**임이 증명되었습니다.
    * **심리적 저항선 모델링**: 투자자의 공포는 선형적이 아닌 기하급수적으로 커집니다. 로그 스케일링은 이러한 인간의 심리적 내성을 반영하여, 지수가 무한정 튀지 않으면서도 **역대급 위기 상황(700선 이상)을 명확히 인지**할 수 있도록 설계된 공학적 선택입니다.

### **2) 시계열 연속성 확보: 5일 EMA 스무딩**

* **발견된 문제 (Problem)**: 
    * 트럼프 대통령이 발언을 쉬는 주말이나 공백기에 지수가 즉각 **0**으로 수직 낙하하는 '데이터 단절 현상'이 발생했습니다.
    * 시장에 잔존하는 긴장감과 정책적 불확실성은 하루아침에 사라지지 않음에도, 지수가 '점' 단위로만 존재하여 거시적인 리스크 흐름(Trend)을 파악하기 어려웠습니다.를 반영하지 못함.
* **보정 방법 (Solution)**: 
    * 일일 단위의 노이즈를 정제하고 데이터 간의 연결성을 부여하기 위해 **5일 지수 이동 평균(EMA, span=5)** 필터를 적용했습니다.
* **채택 근거 및 5일의 선정 이유 (Why 5 Days?)**: 
    * **주간 단위 거래일 정렬**: 금융 시장의 1주일 영업일은 **5일**입니다. span=5는 시장이 한 주 동안 받아들인 충격을 가장 균형 있게 요약해 주는 수치입니다.
    * **3일의 한계 (과도한 민감도)**: 초기 모델인 3일 EMA는 반응은 빠르나, 단순 오타나 사소한 해프닝성 발언에도 지수가 지나치게 요동치는 '잔진동(Noise)'을 다 걷어내지 못했습니다.
    * **10일의 위험성 (반응성 저하)**: 10일 이상의 장기 EMA를 적용할 경우, 리스크 지표의 생명인 **'즉각성'**이 훼손됩니다. 중대한 정책 발표가 있어도 지수가 너무 느리게 반응하여, 리스크 경보 지표로서의 실무적 가치가 상실되는 '지연 현상(Lag)'이 발생하기 때문입니다.
    * **결론**: 수백 번의 시뮬레이션 결과, **5일**은 시장의 긴장 상태를 '묵직하게' 유지하면서도, 새로운 충격에는 '민첩하게' 반응하는 **리스크 지표의 골든 타임**임이 증명되었습니다.

`
"리스크 지표는 너무 예민해서도(3일), 너무 둔해서도(10일) 안 됩니다. 우리는 주식 시장의 호흡인 5영업일을 기준으로 잡음으로써, 일시적인 소음은 걸러내고 **'현재 리스크가 고조되는 국면인지'**를 가장 정확하게 포착하도록 튜닝했습니다."
`

### **3) 데이터 오염 원천 차단: 실시간 클리핑 (Clipping: Why 0.05?)**

* **발견된 문제 (Problem)**: 
    * yfinance 등 외부 API를 통해 수집되는 변동성 데이터(`vol_proxy`)는 네트워크 오류나 데이터 제공사 측의 문제로 인해 간헐적으로 **1.0(100% 변동)** 혹은 **0(데이터 누락)**과 같은 극단적인 이상치가 유입될 수 있습니다.
    * 이러한 단 하나의 오염된 데이터가 지수에 반영될 경우, 전체 시계열의 스케일이 파괴되어 이후의 모든 지표가 신뢰성을 잃게 되는 **'연쇄 오염'** 현상이 발생했습니다.

* **보정 방법 (Solution)**: 
    * 연산 과정에서 `vol_proxy` 값이 특정 범위를 벗어나지 못하도록 강제 제한하는 **클리핑(Clipping)** 로직을 적용했습니다.
    * 공식: $Vol\_Proxy_{applied} = \text{clip}(Vol\_Proxy_{raw}, 0.001, 0.05)$

* **채택 근거 및 0.05의 선정 이유 (Why 0.05?)**: 
    * **현실적 최대 변동폭 반영**: 주식 시장에서 대표 지수(SPY 등)나 섹터 ETF의 일일 변동성(Parkinson Vol)이 **5%(0.05)**를 넘어서는 경우는 금융 위기 수준의 극단적인 상황입니다. 이를 상한선으로 설정함으로써, 통계적 오류는 걸러내고 **실제 시장의 강력한 쇼크는 그대로 수용**할 수 있는 최적의 임계치를 확보했습니다.
    * **하한선(0.001)의 역할**: 변동성이 0에 수렴할 경우 이후 로그($\ln$) 연산에서 수학적 오류(Runtime Warning)가 발생하거나 지수가 비정상적으로 소멸할 수 있습니다. 최소한의 하한선을 지지함으로써 **수학적 안정성**을 확보했습니다.
    * **결론**: 클리핑은 시스템의 **견고성(Robustness)**을 위한 '안전 퓨즈'입니다. 이를 통해 1,700여 개의 행 데이터 중 이상치에 의한 지수 파손 사례를 **0건**으로 완벽하게 제어했습니다.

`
"외부 데이터는 언제든 틀릴 수 있습니다. 하지만 우리 지수는 틀려선 안 됩니다. 0.05 클리핑은 '시장은 미쳐도 우리 지수는 이성을 유지하게 만드는' 최소한의 방어 장치입니다. 덕분에 API가 1.0이라는 말도 안 되는 값을 뱉어도 우리 지표는 오염되지 않고 정상적인 궤적을 유지합니다."
`

### **4) 지표 무결성 보장: 시작점 재정규화 (Re-normalization: Why 100.0000?)**

* **발견된 문제 (Problem)**: 
    * 지수 이동 평균(EMA)은 이전 데이터의 가중치를 포함하여 현재값을 산출하는 특성을 가집니다. 이로 인해 기준일($t=0$)의 초기 연산값이 이전 데이터의 부재로 인해 1기(2017)와 2기(2025)에서 각각 129.5, 40.2 등 제각각으로 산출되는 현상이 발생했습니다.
    * 시작점이 다르면 "과거 취임 초기보다 현재가 더 위험한가?"라는 핵심 질문에 대해 정량적인 비교가 불가능해집니다.

* **보정 방법 (Solution)**: 
    * 연산의 최후 단계에서 모든 시계열 데이터를 기준일($t=0$)의 산출값으로 나누어 눈금을 재조정하는 **역산 보정(Back-calculation)**을 적용했습니다.
    * 공식: $TSUI_{final, t} = \frac{TSUI_{ema, t}}{TSUI_{ema, baseline}} \times 100$

* **채택 근거 및 정규화의 의미 (Why 100.0000?)**: 
    * **상대적 지표의 표준화**: 지수는 절대값이 아니라 **'기준 대비 배율'**이 핵심입니다. 두 시기의 시작점을 **100.0000**으로 강제 고정함으로써, 지수가 200이 되면 "취임 당시보다 공포가 2배 커졌다"는 명확한 해석 권한을 사용자에게 부여합니다.
    * **비교의 공정성 확보**: 재정규화를 통해 1기(2017)와 2기(2025) 데이터를 동일한 저울 위에 올릴 수 있게 되었습니다. 이를 통해 "2기 평균 지수가 1기 대비 약 26% 높게 형성되어 있다"는 객관적인 비교 분석 결과를 도출할 수 있었습니다.
    * **결론**: 재정규화는 지표의 **'숫자적 가독성'**과 **'분석적 일관성'**을 완성하는 마침표입니다. 이를 통해 1,700여 개의 행 데이터가 하나의 논리적인 체계 안으로 통합되었습니다.

`
"데이터가 아무리 많아도 기준이 없으면 정보가 아니라 소음에 불과합니다. 우리는 시작점 재정규화를 통해 8년의 시간을 뛰어넘어 1기와 2기의 리스크를 동일한 잣대로 비교할 수 있는 **'공통 언어'**를 만들었습니다. 덕분에 우리 지표는 과거의 기록이자 현재의 나침반이 될 수 있습니다."
`

---

## 5. 시차 매칭 및 검증 방법론 (Validation)

### **4.1 시차 매칭 (Time-Lag Sync)**
* **로직**: `pd.merge_asof(direction='forward')`를 사용하여 발언 시점($T$) 이후 가장 가까운 시장 데이터($T+n$)를 동기화.
* **처리**: 주말 및 장외 발언은 익일 개장 데이터와 매칭하여 데이터 연속성 확보.

### **4.2 통계적 검증**
* **그랜저 인과관계(Granger Causality)**: $TSUI$ 시계열이 실제 `vix_delta`, `epu_delta`의 미래 값을 예측하는 선행 지표인지 검정.
* **비정상 수익률(AR)**: 지수 급등일 전후의 실제 ETF 수익률이 통계적 기대치를 벗어나는지 확인.

---

## 6. TSUI 레벨링 시스템 (Risk Leveling)

단순한 점수 나열이 아니라, 과거 1기(2017-2021) 전체 데이터의 통계적 분포를 기준으로 현재의 불확실성 강도를 5단계로 정의합니다.

### **5.1 Z-Score 기반 상대 평가**
지수 산출 결과값의 평균($\mu$)과 표준편차($\sigma$)를 활용하여 현재 지수의 위치를 판별합니다.

| 레벨 | 명칭 | 통계적 기준 (Z-Score) | 시장 의미 및 대응 가이드 |
| :--- | :--- | :--- | :--- |
| **LEVEL 1** | **Critical** | $Z > +2.0$ | **초비상.** 상위 2.5%의 역사적 패닉. 즉각적 리스크 관리 필요. |
| **LEVEL 2** | **High** | $+1.0 < Z \le +2.0$ | **경계.** 유의미한 시장 충격 발생. 특정 섹터 변동성 급증. |
| **LEVEL 3** | **Elevated** | $0 < Z \le +1.0$ | **주의.** 평소보다 매파적이나 시장이 감내 가능한 수준. |
| **LEVEL 4** | **Neutral** | $-1.0 < Z \le 0$ | **정상.** 시장 영향력이 미미한 일상적 발언 상태. |
| **LEVEL 5** | **Safe** | $Z \le -1.0$ | **안정.** 비둘기파적 발언 우세로 불확실성 해소 국면. |

* **효과**: 시장이 트럼프 발언에 무뎌지는 '내성'을 통계적으로 자동 반영하여 경보의 정확도를 유지합니다.

---

## 7. 키워드 가중치 전략 (Hybrid Keyword Weighting)

Volfefe의 핵심 엔진을 유지하면서, 우리 프로젝트의 특성인 '12개 섹터 분석'에 최적화된 하이브리드 키워드 사전을 운용합니다.

### **7.1 가중치 카테고리**
1. **Market-Wide (Volfefe 계승)**: 시장 전체를 흔드는 키워드. (x2.0 가중치)
   * 예: `Tariff`, `China`, `Fed`, `Trade War`, `Billion Dollars`
2. **Sector-Specific (자체 지정)**: Track 2의 섹터 확신도를 보정하는 키워드.
   * 반도체: `Chips`, `TSMC`, `Nvidia`, `Export Controls`
   * 에너지: `Oil`, `Drill`, `Fracking`, `LNG`
   * 방산: `NATO`, `Defense Budget`, `Space Force`
3. **Intensity Boost (형용사 가중치)**: 발언의 강도를 격상시키는 트럼프 특유의 화법.
   * 예: `Massive`, `Huge`, `Disaster`, `Unprecedented`


### **📌 트럼프 1기 (2017–2021) 핵심 키워드**

| 카테고리 | 키워드 | 근거 자료 |
| --- | --- | --- |
| **시장 전체** | `Tariff`, `Trade War`, `China`, `Mexico`, `Canada`, `EU`, `Billion Dollars` | USCC 타임라인 + 무역 전쟁 보고서 [www.uscc.gov](https://www.uscc.gov/sites/default/files/2021-04/Timeline_of_Executive_Actions_on_China-2017_to_2021.pdf), [taxfoundation.org](https://taxfoundation.org/research/all/federal/trump-tariffs-trade-war/) |
| **섹터별** | • **철강/알루미늄**: `Steel`, `Aluminum`, `Section 232`
• **반도체**: `Chips`, `TSMC`, `Export Controls`
• **농업**: `Soybeans`, `Agriculture`
• **자동차**: `Auto`, `NAFTA`, `USMCA` | 라보뱅크 섹터 분석 + USCC 문서 [www.rabobank.com](https://www.rabobank.com/knowledge/d011318389-us-china-trade-war-which-sectors-are-most-vulnerable-in-the-global-value-chain), [aflcio.org](https://aflcio.org/testimonies/united-states-mexico-canada-agreement-likely-impact-us-economy-and-specific-industry) |
| **강도 부스터** | `Massive`, `Huge`, `Disaster`, `Catastrophic` | 트럼프 트윗 어조 분석 논문 (Ross, 2020) [www.sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S0271530919302617) |

### **📌 트럼프 2기 (2025–현재) 핵심 키워드**

| 카테고리 | 키워드 | 근거 자료 |
| --- | --- | --- |
| **신규 테마** | `AI`, `Artificial Intelligence`, `Crypto`, `Digital Assets`, `De-risking`, `Economic Security` | 2025년 1월 암호화폐 행정 명령 + AI 액션 플랜 [www.grantthornton.com](https://www.grantthornton.com/insights/articles/advisory/2025/crypto-policy-outlook), [www.whitehouse.gov](https://www.whitehouse.gov/wp-content/uploads/2025/07/Americas-AI-Action-Plan.pdf) |
| **강화된 섹터** | • **고급 반도체**: `Advanced AI Chips`, `Nvidia`, `H200`, `TSMC`
• **에너지**: `Oil`, `Drill`, `Fracking`, `LNG` | 2026년 1월 반도체 관세 프로클레이션 [www.thomsonreuters.com](https://www.thomsonreuters.com/en-us/help/onesource-global-trade/regulatory-insights/2026/january-15th/trump-administration-imposes-tariffs-on-advanced-a), [www.whitehouse.gov](https://www.whitehouse.gov/fact-sheets/2026/01/fact-sheet-president-donald-j-trump-takes-action-on-certain-advanced-computing-chips-to-protect-americas-economic-and-national-security/) |
| **정책 수단** | `Section 232`, `IEEPA` (International Emergency Economic Powers Act) | 백악관 팩트 시트 + 무역 전문가 분석 [www.whitehouse.gov](https://www.whitehouse.gov/fact-sheets/2026/01/fact-sheet-president-donald-j-trump-takes-action-on-certain-advanced-computing-chips-to-protect-americas-economic-and-national-security/), [taxfoundation.org](https://taxfoundation.org/research/all/federal/trump-tariffs-trade-war/) |

### **🔍 공식 학술 자료 (키워드 방법론 검증용)**

| 자료명 | 출처 | 핵심 내용 | 링크 |
| --- | --- | --- | --- |
| **Measuring Economic Policy Uncertainty** (Baker et al., 2016) | *Quarterly Journal of Economics* (피인용 16,000+) | `Economy` + `Policy` + `Uncertainty` 3가지 키워드 조합 기반 뉴스 카운팅 방법론 제시 | [policyuncertainty.com](https://www.policyuncertainty.com/) [academic.oup.com](https://academic.oup.com/qje/article-abstract/131/4/1593/2468873) |
| **US Daily Trade Policy Uncertainty Index** | Baker-Bloom-Davis 공식 사이트 | 무역 정책 불확실성 측정을 위한 카테고리별 키워드 사전 제공 | [Trade Uncertainty 페이지](https://www.policyuncertainty.com/trade_uncertainty.html) [www.policyuncertainty.com](https://www.policyuncertainty.com/trade_uncertainty.html) |
| **Timeline of Executive Actions on China (2017–2021)** | USCC (US-China Economic and Security Review Commission) | 1기 트럼프 행정부의 중국 관련 8개 행정 명령 및 구체적 조치 목록 | [USCC 공식 문서](https://www.uscc.gov/sites/default/files/2021-04/Timeline_of_Executive_Actions_on_China-2017_to_2021.pdf) [www.uscc.gov](https://www.uscc.gov/sites/default/files/2021-04/Timeline_of_Executive_Actions_on_China-2017_to_2021.pdf) |

### **7.2 구현 방식**
코드 내에 하드코딩하지 않고 `Weight Dictionary` 형태의 외부 설정 파일로 관리하여, 시장 상황에 따라 유연하게 키워드를 업데이트합니다.

```python
# 가중치 사전 예시 구조
KEYWORD_MAP = {
    "global": {"tariff": 2.0, "fed": 2.0},
    "sector": {"semiconductor": ["chips", "export"], "energy": ["oil", "drill"]},
    "intensity": ["massive", "huge", "disaster"]
}
```

## 8. 1기-2기 데이터 연속성 및 공백기 처리 전략

트럼프 1기(과거)와 2기(현재) 사이의 4년(2021~2024) 공백으로 인한 데이터 왜곡을 방지하기 위해 **세션 분리 및 적응형 레벨링** 체계를 도입합니다.

### **8.1 세션별 독립 지수 관리 (Segmented Indexing)**
* **1기 세션 (Benchmark)**: 2017.01.20을 지수 100으로 설정하여 임기 마지막 날까지 누적 산출. 이를 통해 우리 모델의 통계적 기준값($\mu, \sigma$)을 확립합니다.
* **공백기 처리 (Freezing)**: 2021.01.21 ~ 2025.01.19 기간은 데이터 '결측치(NaN)'로 처리하여 정책적 불연속성을 명시합니다.
* **2기 세션 (Live)**: 2025.01.20을 **새로운 기준점(100)**으로 재설정하여 시작합니다. 4년 사이 변화한 시장 체급과 섹터 비중(AI 등)을 반영하기 위한 조치입니다.

### **8.2 적응형 레벨링 시스템 (Adaptive Leveling)**
* **초기 단계 (벤치마킹)**: 2기 데이터가 충분히 쌓이기 전까지는 1기에서 도출된 통계 분포($\mu_{1st}, \sigma_{1st}$)를 기준으로 LEVEL 1~5 경보를 생성합니다.
* **전환 단계 (재보정)**: 2기 발언 데이터가 일정 수치(예: 100건) 이상 누적되면, **2기 자체 롤링 윈도우(Rolling Window)** 기반의 평균과 표준편차로 자동 전환합니다.
* **효과**: 트럼프의 강화된 화법이나 시장의 변화된 민감도(내성)를 실시간으로 지수에 반영할 수 있습니다.

### **8.3 2기 특화 키워드 확장**
* 1기의 전통적 무역 키워드(`Tariff`, `Steel`)에 더해, 2기 핵심 테마인 `AI`, `Economic Security`, `De-risking`, `Crypto` 등을 키워드 가중치 사전에 추가하여 분석의 시의성을 확보합니다.

---
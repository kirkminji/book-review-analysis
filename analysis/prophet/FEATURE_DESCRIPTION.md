# books_ml_dataset_with_prophet.csv 피처 설명서

## 개요
- **파일 위치**: `analysis/prophet/books_ml_dataset_with_prophet.csv`
- **레코드 수**: 8,364개
- **컬럼 수**: 33개
- **단위**: 도서(product_code) × 주차(ymw)

---

## 1. 기본 키

| 컬럼 | 타입 | 설명 | 예시 |
|------|------|------|------|
| `product_code` | string | 도서 고유 코드 | S000000696161 |
| `ymw` | string | 주차 코드 (년+월+주) | 2025012 |

---

## 2. 카테고리 비율 피처 (10개)

| 컬럼 | 설명 |
|------|------|
| `category_1` ~ `category_10` | 해당 도서가 각 카테고리에 속하는 비율 (0~1) |

### 카테고리 매핑
| 번호 | 카테고리명 (한글) | 영문 코드 |
|------|------------------|-----------|
| 1 | 부동산/실물자산 | real_estate |
| 2 | 재테크/개인금융 | personal_finance |
| 3 | 주식투자/트레이딩 | stock_trading |
| 4 | 지정학/국제정세 | geopolitics |
| 5 | 테크/스타트업 | tech_startup |
| 6 | 거시경제/금융정책 | macro_economy |
| 7 | 기업경영/리더십 | business |
| 8 | 경제이론/학술 | econ_theory |
| 9 | 투자철학/대가 | invest_philosophy |
| 10 | 금융시스템/위기 | financial_crisis |

### 출처
- `books` 테이블의 태그 기반 분류
- **Join 기준**: `product_code`

### 예시
```
category_4 = 0.6 → 해당 도서가 "지정학/국제정세" 카테고리에 60% 속함
```

---

## 3. 카테고리 × 바이럴 교차 피처 (10개)

| 컬럼 | 계산식 |
|------|--------|
| `category_1_x_viral_index` | category_1 × viral_index_1 |
| `category_2_x_viral_index` | category_2 × viral_index_2 |
| ... | ... |
| `category_10_x_viral_index` | category_10 × viral_index_10 |

### 의미
- 해당 도서가 속한 카테고리의 **뉴스 바이럴 영향력**을 반영
- 도서의 카테고리 비율이 높을수록, 해당 카테고리의 뉴스 영향을 많이 받음

### 출처
- `weekly_news_viral_index_revised.csv`의 `viral_index_smoothed` 값 사용
- **Join 기준**: `ymw`

### 예시
```
category_4 = 0.6
viral_index_4 = -59.5
category_4_x_viral_index = 0.6 × (-59.5) = -35.7
```

---

## 4. Prophet 예측 피처 (10개)

| 컬럼 | 카테고리 | 최적 Lag |
|------|----------|---------|
| `prophet_forecast_macro_economy` | 거시경제/금융정책 | 3주 |
| `prophet_forecast_econ_theory` | 경제이론/학술 | 4주 |
| `prophet_forecast_financial_crisis` | 금융시스템/위기 | 2주 |
| `prophet_forecast_business` | 기업경영/리더십 | 4주 |
| `prophet_forecast_real_estate` | 부동산/실물자산 | 4주 |
| `prophet_forecast_personal_finance` | 재테크/개인금융 | 3주 |
| `prophet_forecast_stock_trading` | 주식투자/트레이딩 | 2주 |
| `prophet_forecast_geopolitics` | 지정학/국제정세 | 2주 |
| `prophet_forecast_tech_startup` | 테크/스타트업 | 4주 |
| `prophet_forecast_invest_philosophy` | 투자철학/대가 | 3주 |

### 의미
- 각 카테고리별 **Decay Score 기반 판매점수**를 Prophet 모델로 예측한 값
- **최적 시차(Lag)가 카테고리별로 다르게 적용됨**

### 출처
- `prophet_walkforward_decay_predictions.csv`
- **Join 기준**: `ymw`

### 예측 방법
- Prophet Walkforward Validation (min_train=5주)
- Regressor: `viral_index_smoothed` (시차 적용)

### 결측치 처리
- 초반 주차(학습 기간)는 **카테고리별 평균값**으로 채움

### 예시
```
prophet_forecast_geopolitics (2025042 주차)
= T-2주(2025040) viral_index로 예측한 "2025042 주차 판매점수"
```

---

## 5. 타겟 변수 (1개)

| 컬럼 | 설명 |
|------|------|
| `y_sales_score` | 해당 도서의 해당 주차 Decay Score |

### Decay Score 계산 방식
```
if 차트인 (1~20위):
    score = 21 - rank  # 1위=20점, 20위=1점
else (차트아웃):
    score = 전주_score × 0.5  # 점진적 감소
    if score < 0.1:
        score = 0
```

### 출처
- DB `weekly_bestsellers` 테이블에서 계산
- **Join 기준**: `product_code` + `ymw`

### Decay Score 도입 이유
- 기존 방식: 차트아웃 시 즉시 0점 (급격한 단절)
- Decay 방식: 차트아웃 후에도 영향력이 점진적으로 소멸

---

## 데이터 흐름도

```
┌─────────────────────────────────────────────────────────────────┐
│                         books 테이블                             │
│                      (product_code 기준)                         │
├─────────────────────────────────────────────────────────────────┤
│  → category_1 ~ category_10 (도서의 카테고리 비율)               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              weekly_news_viral_index_revised.csv                 │
│                        (ymw 기준 Join)                           │
├─────────────────────────────────────────────────────────────────┤
│  → viral_index_smoothed                                          │
│  → category_X_x_viral_index = category_X × viral_index_X         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│           prophet_walkforward_decay_predictions.csv              │
│                        (ymw 기준 Join)                           │
├─────────────────────────────────────────────────────────────────┤
│  → prophet_forecast_X (카테고리별 피벗, 최적 Lag 적용)           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  weekly_bestsellers 테이블                       │
│                  (product_code + ymw 기준 Join)                  │
├─────────────────────────────────────────────────────────────────┤
│  → y_sales_score (Decay Score 계산)                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 관련 파일 목록

| 파일 | 설명 |
|------|------|
| `books_ml_dataset_with_prophet.csv` | 최종 ML 데이터셋 |
| `prophet_walkforward_decay_predictions.csv` | Prophet 예측 결과 (전체) |
| `prophet_walkforward_decay_summary.csv` | 카테고리별 성능 요약 |
| `weekly_bestseller_scores_decay.csv` | 카테고리별 주차별 Decay Score 합계 |
| `weekly_news_viral_index_revised.csv` | 주간 뉴스 바이럴 지수 |

---

## Prophet 모델 성능 요약

| 카테고리 | 최적 Lag | MAE | RMSE | 방향성 정확도 |
|----------|---------|-----|------|--------------|
| econ_theory | 4주 | 4.69 | 5.79 | 50.0% |
| tech_startup | 4주 | 6.28 | 8.38 | 50.0% |
| real_estate | 4주 | 6.80 | 9.61 | 67.5% |
| geopolitics | 2주 | 6.91 | 8.54 | 71.4% |
| financial_crisis | 2주 | 7.62 | 11.11 | 61.9% |
| personal_finance | 3주 | 8.31 | 10.19 | 48.8% |
| invest_philosophy | 3주 | 8.38 | 10.69 | 56.1% |
| macro_economy | 3주 | 10.35 | 12.95 | 48.8% |
| stock_trading | 2주 | 10.56 | 13.51 | 45.2% |
| business | 4주 | 11.70 | 15.20 | 52.5% |
| **평균** | - | **8.16** | **10.63** | **55.2%** |

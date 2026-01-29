# 경제/경영 베스트셀러 예측 분석 프로젝트

뉴스 바이럴 지수와 경제 지표를 활용하여 교보문고 경제/경영 분야 베스트셀러를 예측하는 프로젝트

- **학습 데이터**: 2025년 1월 ~ 12월 (48주, 141권)
- **검증 데이터**: 2026년 1월 1주차 실제 베스트셀러

> 핵심 질문: "뉴스 바이럴 트렌드로 베스트셀러를 예측할 수 있는가?"

---

## 분석 파이프라인

```
데이터 수집 → EDA → 상관분석 → Prophet 시계열 → ML 예측 → Validation
```

---

## 주요 결과

### 상관분석
- 뉴스 바이럴 지수와 베스트셀러 간 1~3주 시차 존재
- 카테고리별 상관계수 0.31~0.42

### Prophet
- Walk-Forward 검증, 평균 MAPE 15~25%
- 급격한 트렌드 변화 예측에 한계

### ML 예측 (LightGBM)

| 조건 | R² | F1 |
|------|-----|-----|
| 전체 43개 피처 | 0.21 | 0.36 |
| Feature Selection (8개) | 0.41 | 0.49 |
| + y_lag1 (전주 판매점수) | 0.86 | 0.97 |

- Feature Selection: 다중공선성 제거 + SHAP 60% 기준 8개 선택
- y_lag1 하나가 43개 피처를 압도 (베스트셀러 유지율 92.3%)

### Validation (2026년 1월 1주차)

| 지표 | 값 |
|------|-----|
| Top 20 일치율 | 50% (10/20) |
| 순위 상관계수 | 0.713 |
| 신규 진입 (예측 불가) | 10권 |

- 기존 책 순위 예측: 양호 (r=0.71)
- 신규 진입 10권: 학습 데이터에 없는 완전 신규 도서로 예측 불가

---

## 결론

> **"기존 베스트셀러의 순위 유지는 예측 가능하나, 신규 진입(특히 신간)은 예측 불가"**

- 바이럴 지수 단독 예측력은 미미 (r < 0.1)
- 전주 판매점수(y_lag1)가 지배적 — 이는 플랫폼 노출, 마케팅, 진열 등 숨겨진 변수의 대리변수
- 과적합 아님: Naive Baseline(전주값 그대로)도 R²=0.88, 데이터 자체의 특성

---

## 파일 구조

```
book-review-analysis/
├── README.md
├── bestseller_crawler/           # 교보문고 크롤러
├── raw_biz_news_data/            # 한경 뉴스 원본 데이터
│
└── analysis/
    ├── EDA/                      # 탐색적 데이터 분석
    ├── correlation/              # 바이럴-판매 상관분석, 뉴스 카테고라이징
    ├── viral_index/              # 주간 바이럴 지수 산출
    ├── prophet/                  # Prophet 시계열 예측
    ├── my_prophet/               # Walk-Forward Prophet 분석
    ├── prediction/               # ML 예측 및 Validation
    │   ├── ml_lag_feature_test_v4.ipynb - ML 분석 (메인)
    │   ├── books_ml_dataset_v4.csv      - 통합 학습 데이터셋
    │   ├── ml_image_v4/                 - 분석 시각화 결과물
    │   ├── models/                      - 학습된 모델 및 데모 앱
    │   │   ├── app.py                   - Gradio 기반 예측 데모 앱
    │   │   └── best_reg_model.pkl       - 최종 예측 모델
    │   └── validation/                  - 2026년 1월 데이터 검증
    │       ├── validation_2026_jan.py   - 검증 데이터 생성 및 예측
    │       └── validation_2026_jan.ipynb
    ├── prediction_legacy/        # 이전 버전 예측
    ├── review_issues/            # 리뷰 분석
    └── market_analytics/         # 시장 분석
```

---

## 사용 기술

Python 3.13 | LightGBM, XGBoost, scikit-learn | Prophet | SHAP | matplotlib, plotly | Supabase

---

*마지막 업데이트: 2026년 1월 29일*

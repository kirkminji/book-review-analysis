# ecommerce-product-review-analysis

This project uses `uv` for dependency management.

## Prerequisites
- **uv**: [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

## Setup
To install all dependencies and set up the virtual environment, run:

```bash
uv sync
```

## Running the Crawler
To run the Playwright crawler:

```bash
# First time setup for Playwright browsers
uv run playwright install

# Run the script
uv run kyobo_chroller_sample.py
```

## Dependencies
Dependencies are managed in `pyproject.toml` and locked in `uv.lock`.
Start adding new dependencies with:
```bash
uv add <package_name>
```

---

# 뉴스-베스트셀러 상관관계 분석

## 분석 개요

경제/경영 분야 **뉴스 트렌드**와 **교보문고 베스트셀러** 간의 상관관계를 분석하여, 뉴스가 도서 구매에 미치는 영향을 정량적으로 파악하는 프로젝트입니다.

### 연구 질문
- 특정 주제의 뉴스가 급증하면 관련 도서 판매도 증가하는가?
- 뉴스 트렌드와 베스트셀러 순위 사이에 시간 차이(시차)가 존재하는가?
- 어떤 카테고리가 뉴스 트렌드에 가장 민감하게 반응하는가?

## 데이터

| 데이터셋 | 기간 | 건수 | 설명 |
|---------|------|------|------|
| 뉴스 데이터 | 2025년 1월~12월 | 63,423건 | BigKinds 경제/경영 분야 기사 |
| 베스트셀러 데이터 | 2025년 1월~12월 | 2,313건 | 교보문고 경제/경영 월별 베스트셀러 (20위×12개월) |
| 리뷰 데이터 | 2025년 1월~12월 | - | 베스트셀러 도서별 사용자 리뷰 |

---

### 데이터 수집 방법

#### 1. 뉴스 데이터 (BigKinds)
- **출처**: [BigKinds](https://www.bigkinds.or.kr/) (한국언론진흥재단 뉴스 빅데이터 분석 서비스)
- **수집 방법**: BigKinds 웹사이트에서 경제/경영 분야 기사 검색 후 CSV 다운로드
- **검색 조건**: 2025년 1월~12월, 경제 분야

#### 2. 베스트셀러 데이터 (교보문고)
- **출처**: 교보문고 경제/경영 카테고리 월별 베스트셀러
- **수집 방법**: Playwright 기반 웹 크롤링 (`bestseller-crawler/kyobo_bestseller_crawler.py`)
- **수집 대상**: 월별 상위 20위 도서의 상세 정보
- **실행 명령**:
  ```bash
  uv run python bestseller-crawler/kyobo_bestseller_crawler.py --year 2025 --months 1 2 3 4 5 6 7 8 9 10 11 12
  ```

#### 3. 리뷰 데이터 (교보문고)
- **출처**: 교보문고 도서 상세페이지 리뷰
- **수집 방법**: 교보문고 API 호출 (`kyobo_review_crawler/kyobo_review_crawler.py`)
- **수집 대상**: 베스트셀러 도서별 사용자 리뷰 (도서당 최대 300개)
- **실행 명령**:
  ```bash
  uv run python kyobo_review_crawler/kyobo_review_crawler.py
  ```

---

### 데이터 스키마

#### 뉴스 데이터 (`news_2025_categorized.csv`)
| 컬럼명 | 설명 | 예시 |
|--------|------|------|
| 뉴스 식별자 | BigKinds 고유 ID | 2100601.20250331 |
| 일자 | 기사 발행일 | 2025-03-31 |
| 언론사 | 기사 출처 | 한국경제 |
| 제목 | 기사 제목 | DN솔루션즈, 해외 기관 영업 시작... |
| 통합 분류1~3 | BigKinds 자동 분류 | 경제>증권_증시 |
| 키워드 | 추출 키워드 | dn솔루션즈,투자자,lg... |
| 본문 | 기사 전문 | (텍스트) |
| URL | 기사 원문 링크 | https://... |
| 년월 | 기사 발행 년월 | 2025-03 |
| **카테고리** | 분석용 카테고리 (10개) | 주식투자/트레이딩 |

#### 베스트셀러 데이터 (`kyobo_bestseller_*.csv`)
| 컬럼명 | 설명 | 예시 |
|--------|------|------|
| bestseller_month | 베스트셀러 기준 월 | 2025-01 |
| rank | 순위 (1~20) | 1 |
| title | 도서 제목 | 트렌드 코리아 2026 |
| author | 저자 | 김난도, 전미영... |
| translator | 번역자 | (null) |
| publisher | 출판사 | 미래의창 |
| publish_date | 출판일 | 2025년 09월 24일 |
| price | 정가 | 18000 |
| isbn | ISBN | 9791193638859 |
| product_code | 교보문고 상품코드 | S000217467412 |
| rating | 평점 (10점 만점) | 9.7 |
| review_count | 리뷰 수 | 331 |
| description | 짧은 설명 | 2026 대한민국 소비트렌드 전망 |
| intro_text | 상세 소개글 | (텍스트) |
| keywords | 도서 키워드 | 트렌드, 소비... |
| image_url | 표지 이미지 URL | https://... |
| product_url | 상품 페이지 URL | https://... |

#### 리뷰 데이터 (`kyobo_reviews_*.csv`)
| 컬럼명 | 설명 | 예시 |
|--------|------|------|
| product_code | 교보문고 상품코드 | S000217467412 |
| review_content | 리뷰 내용 | 기사 챗gpt만 찿아봐도... |
| rating | 별점 (1~5) | 3 |
| emotion_keyword | 감정 키워드 | 쉬웠어요 |
| reviewer_id | 리뷰어 ID (마스킹) | eu**** |
| review_date | 리뷰 작성일 | 2025-11-01 |
| helpful_count | 도움됨 수 | 27 |
| comment_count | 댓글 수 | 2 |

---

### 카테고리 분류 (10개)
- 주식투자/트레이딩, 거시경제/금융정책, 부동산/자산투자
- 재테크/개인금융, 자기계발/리더십, 창업/스타트업/경영
- 마케팅/세일즈, 경제이론/학술, 부자마인드/성공철학, 산업/기업분석

## 분석 방법론

### 1. 바이럴 지수 (Viral Index)

단순 기사 건수 대신 **급등/급락 패턴**을 포착하는 복합 지표를 설계했습니다.

```
바이럴 지수 = MoM × 0.5 + MA편차 × 0.3 + Z-Score × 0.2
```

| 구성요소 | 가중치 | 설명 |
|---------|--------|------|
| MoM (전월 대비 증감률) | 50% | 단기 급등/급락 감지 |
| MA 편차 (3개월 이동평균 대비) | 30% | 중기 트렌드 이탈 감지 |
| Z-Score (표준화 점수) | 20% | 전체 기간 내 이상치 탐지 |

**왜 단순 기사 건수가 아닌가?**
- 기사 "개수"가 아니라 "급등/급락"이 도서 구매에 영향
- 매달 100건 나오는 주제 vs 50→200건으로 급등한 주제 → 후자가 더 큰 영향

### 2. 역순위 가중치 (Inverse Rank Weighting)

베스트셀러 순위를 **멱법칙(Power Law)**에 기반한 가중치로 변환했습니다.

```
가중치 = 1/rank
```

- 1위: 가중치 1.0 (최대)
- 10위: 가중치 0.1
- 20위: 가중치 0.05

**왜 역순위 가중치인가?**
- 베스트셀러 1위와 20위의 실제 판매량 차이는 단순 순위 차이(19)보다 훨씬 큼
- 1/rank는 이러한 멱법칙적 분포를 반영

### 3. Spearman 상관계수

**피어슨 대신 스피어만을 선택한 이유:**

| 기준 | Pearson | Spearman |
|------|---------|----------|
| 데이터 분포 가정 | 정규분포 필요 | 분포 무관 |
| 이상치 민감도 | 매우 민감 | 강건함 |
| 소표본 적합성 | 제한적 | 적합 |
| 측정 대상 | 선형 관계 | 단조 관계 |

본 분석의 경우:
- 표본 크기가 작음 (n=12개월)
- 뉴스/베스트셀러 데이터 모두 정규분포가 아님
- 특정 월에 이상치 존재 가능

### 4. 시차(Lag) 분석

뉴스 트렌드가 베스트셀러에 반영되기까지의 시간 지연을 분석했습니다.

```
Lag 0: 뉴스 t월 → 베스트셀러 t월 (동시)
Lag 1: 뉴스 t월 → 베스트셀러 t+1월 (1개월 후)
Lag 2: 뉴스 t월 → 베스트셀러 t+2월 (2개월 후)
Lag 3: 뉴스 t월 → 베스트셀러 t+3월 (3개월 후)
```

## 주요 분석 결과

### 통계적으로 유의한 상관관계 (p < 0.05)

| 카테고리 | 최적 시차 | Spearman r | p-value | 해석 |
|---------|----------|------------|---------|------|
| **주식투자/트레이딩** | 1개월 | **+0.700** | 0.017 | 강한 양의 상관 |
| 거시경제/금융정책 | 0개월 | +0.580 | 0.048 | 중간 양의 상관 |
| 재테크/개인금융 | 0개월 | +0.580 | 0.048 | 중간 양의 상관 |

### 경계 수준 유의성 (0.05 < p < 0.10)

| 카테고리 | 최적 시차 | Spearman r | p-value |
|---------|----------|------------|---------|
| 경제이론/학술 | 1개월 | +0.528 | 0.095 |

### 핵심 인사이트

1. **주식투자/트레이딩**: 뉴스 바이럴 지수가 1개월 후 베스트셀러 순위와 강한 양의 상관관계 (r=0.700)
   - 주식 관련 뉴스가 급증하면 약 1개월 후 관련 도서 판매 증가

2. **거시경제/금융정책, 재테크/개인금융**: 뉴스와 동시에 베스트셀러 반응 (Lag 0)
   - 금리, 환율 등 거시경제 이슈는 즉각적인 도서 구매로 이어짐

3. **대부분의 카테고리**: 0~1개월 시차에서 가장 높은 상관관계
   - 도서 구매 의사결정이 비교적 빠르게 이루어짐

## 파일 구조

```
book-review-analysis/
├── analysis/
│   ├── correlation/                    # 상관분석 관련 데이터
│   │   ├── news_monthly_by_category.csv    # 월별 카테고리별 뉴스 집계
│   │   └── bestseller_monthly_by_category.csv
│   ├── viral_index_eda.ipynb           # EDA 시각화 노트북 (Plotly)
│   └── ...
├── new_analysis/
│   ├── categorize_news.py              # 뉴스 카테고리 분류 스크립트
│   ├── calculate_viral_index.py        # 바이럴 지수 계산
│   ├── calc_indicators.py              # MoM, MA편차, Z-Score 계산
│   ├── correlation_analysis_v3.ipynb   # 메인 상관분석 노트북
│   └── ...
├── bestseller-crawler/                 # 베스트셀러 크롤러
├── kyobo_review_crawler/              # 리뷰 크롤러
└── README.md
```

## 시각화 (EDA)

`analysis/viral_index_eda.ipynb`에서 다음 시각화를 제공합니다:

1. **카테고리별 월별 순위 변동 차트** - 순위 추이 비교
2. **바이럴 지수 히트맵** - 월별×카테고리 지수 분포
3. **순위 히트맵** - 월별 카테고리 순위 (1위=초록, 10위=빨강)
4. **Top 3 카테고리 트렌드 비교** - 주요 카테고리 시계열
5. **카테고리별 변동성 비교** - 표준편차 기반 변동성

## 한계점 및 향후 과제

### 한계점
- 12개월 데이터로 통계적 검정력 제한
- 키워드 기반 카테고리 분류의 정확도 한계
- 뉴스 외 다른 요인(광고, 저자 인지도 등) 미반영

### 향후 과제
- 2년 이상 데이터 확보 시 시계열 예측 모델(Prophet) 적용 가능
- 교차 상관 분석(CCF)으로 정밀한 시차 탐색
- 감성 분석 추가 (뉴스 긍/부정 vs 도서 판매)
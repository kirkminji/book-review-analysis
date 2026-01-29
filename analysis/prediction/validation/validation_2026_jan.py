#!/usr/bin/env python3
"""
2026년 1월 베스트셀러 예측 Validation
1. 새 뉴스 데이터 카테고라이징
2. 주차별 바이럴 인덱스 계산
3. 학습된 모델로 예측
"""

import pandas as pd
import numpy as np
from collections import Counter
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# === 1. 카테고리 키워드 (기존과 동일) ===
CATEGORY_KEYWORDS = {
    "주식투자/트레이딩": [
        "주식투자", "트레이더", "나스닥", "코스피", "종목", "매매", "손절매", "시가총액",
        "급락", "급등", "저점", "고점", "상승세", "외국인", "etf", "코스닥", "거래량",
        "시총", "변동성", "목표주가", "순매도", "레버리지", "서학개미", "매수세",
        "밸류에이션", "대형주", "공공기관", "구조조정", "상장지수펀드", "금융기관",
        "컨센서스", "사모펀드", "트레이딩", "개인 투자자", "초보 투자자", "투자자",
        "배당", "수익", "포트폴리오", "분산 투자", "장기 투자", "시세 차익", "밸류"
    ],
    "투자철학/대가": [
        "워런 버핏", "버핏", "가치 투자", "투자 철학", "서한", "주주", "명언",
        "필립 피셔", "피터 린치", "하워드 막스", "주주가치", "피셔", "린치",
        "하워드", "필립", "보통주", "투자 원칙", "통찰"
    ],
    "재테크/개인금융": [
        "재테크", "부자되는법", "종잣돈", "절세", "노후 준비", "연말 정산", "배당금",
        "현금 흐름", "원금", "계좌", "국민연금", "퇴직연금", "노후 자금", "자산",
        "절세 방법", "금융 지식", "재투자", "퇴직 연금", "배당 소득", "월배당"
    ],
    "거시경제/금융정책": [
        "금리", "인플레이션", "환율", "통화 정책", "기준 금리", "경기 순환",
        "디플레이션", "버블", "중앙은행", "한국은행", "연준", "한은", "기준금리",
        "유동성", "수출액", "달러", "gdp", "고환율", "유로", "거시 경제", "거시경제",
        "금리 인상", "경제 원리", "경제 개념", "글로벌 경제", "한국 경제", "실물 경제"
    ],
    "지정학/국제정세": [
        "트럼프", "우크라", "중국", "패권", "관세", "국제 질서", "지정학", "국가 전략",
        "자유 무역", "이스라엘", "공급망", "대만", "중동", "러시아", "국제 정세",
        "국제 정치", "도널드 트럼프", "국가 안보", "제2차 냉전", "양극"
    ],
    "부동산/실물자산": [
        "부동산 투자", "주택 가격", "집값", "건폐율", "용도지역", "금", "실물",
        "부동산", "재건축", "분양가", "금융당국", "금융사", "실거래", "보험금",
        "재개발", "주담대", "보증금", "금융권", "원자재", "보조금", "토지거래허가구역",
        "투자금", "갭투자", "지원금", "정비사업", "과징금", "금값", "원리금", "계약금",
        "임대료", "다주택자", "증거금", "임차인", "주택담보대출", "전셋값", "무주택자",
        "건물주", "월세"
    ],
    "기업경영/리더십": [
        "리더십", "경영자", "비즈니스 모델", "브랜드 전략", "경쟁력", "혁신 기업",
        "매출", "다각화", "영업이익", "브랜드", "임직원", "ceo", "매출액", "이사회",
        "순이익", "상장사", "경영진", "ipo", "경영", "상장", "지배구조", "최고경영자",
        "덕목", "대전환", "행동 방식", "실행력", "조직"
    ],
    "테크/스타트업": [
        "실리콘밸리", "스타트업", "AI", "프롬프트", "에이전트", "반도체", "휴머노이드",
        "오픈소스", "ai", "인공지능", "전기차", "클라우드", "빅테크", "데이터센터",
        "hbm", "자율주행", "로보틱스", "ces", "파운드리", "빅데이터", "오픈ai", "낸드",
        "중소벤처기업부", "드론", "실리콘", "밸리", "창업자", "오픈", "신경망", "컴퓨팅",
        "병렬", "반도체 산업", "엔비디아", "클로드", "트랜스포머", "모달", "커서"
    ],
    "경제이론/학술": [
        "거시 경제학", "미시 경제학", "케인스", "하이에크", "경쟁 시장", "외부 효과",
        "노벨 경제학", "행동경제학", "연구개발", "연구소", "실수요자", "수요예측",
        "효율성", "공급", "수요자", "경제학", "경제이론", "국부론", "생산요소시장"
    ],
    "금융시스템/위기": [
        "금융 위기", "금융 시스템", "화폐", "기축 통화", "부채", "가계 부채",
        "글로벌 금융 위기", "코인", "비트코인", "암호 화폐", "알트코인", "국제 금융 시장"
    ]
}

# 카테고리명 -> 피처명 매핑
CAT_TO_FEATURE = {
    "거시경제/금융정책": "macro_economy",
    "경제이론/학술": "econ_theory",
    "금융시스템/위기": "financial_crisis",
    "기업경영/리더십": "business",
    "부동산/실물자산": "real_estate",
    "재테크/개인금융": "personal_finance",
    "주식투자/트레이딩": "stock_trading",
    "지정학/국제정세": "geopolitics",
    "테크/스타트업": "tech_startup",
    "투자철학/대가": "invest_philosophy",
    "미분류": "unclassified"
}


def categorize_news(text):
    """텍스트를 카테고리로 분류"""
    if not isinstance(text, str):
        return "미분류"
    text_lower = text.lower()
    scores = Counter()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                scores[cat] += 1
    return scores.most_common(1)[0][0] if scores else "미분류"


def calculate_viral_index(weekly_counts):
    """바이럴 지수 계산"""
    prev = weekly_counts.shift(1)
    wow = ((weekly_counts - prev) / (prev + 1)) * 100
    ma4 = weekly_counts.rolling(4, min_periods=1).mean()
    ma_dev = ((weekly_counts - ma4) / (ma4 + 1)) * 100
    z = (weekly_counts - weekly_counts.mean()) / (weekly_counts.std() + 1e-9)

    viral = (wow.clip(upper=300).fillna(0) * 0.4 +
             ma_dev.clip(upper=300).fillna(0) * 0.4 +
             z.clip(-3, 3).fillna(0) * 10 * 0.2)
    return viral.rolling(2, min_periods=1).mean()


def main():
    print("="*60)
    print("2026년 1월 베스트셀러 예측 Validation")
    print("="*60)

    # === 1. 새 뉴스 로드 및 카테고라이징 ===
    print("\n[1] 뉴스 데이터 로드 및 카테고라이징...")
    news = pd.read_excel('/Users/minzzy/Desktop/statrack/book-review-analysis/raw_biz_news_data/NewsResult_20251231-20260127.xlsx')
    news['date'] = pd.to_datetime(news['일자'], format='%Y%m%d')
    news['text'] = news['제목'].fillna('') + ' ' + news['키워드'].fillna('')
    news['category'] = news['text'].apply(categorize_news)

    print(f"  기간: {news['date'].min().date()} ~ {news['date'].max().date()}")
    print(f"  기사 수: {len(news):,}개")
    print("\n  카테고리 분포:")
    for cat, cnt in news['category'].value_counts().items():
        print(f"    {cat}: {cnt}개 ({cnt/len(news)*100:.1f}%)")

    # === 2. 주차 정의 (2026년 1월) ===
    # 교보 베스트셀러 주차 기준 (화~월)
    weeks_2026_jan = [
        {'ymw': '2026011', 'start': '2025-12-31', 'end': '2026-01-06'},  # 1주차
        {'ymw': '2026012', 'start': '2026-01-07', 'end': '2026-01-13'},  # 2주차
        {'ymw': '2026013', 'start': '2026-01-14', 'end': '2026-01-20'},  # 3주차
        {'ymw': '2026014', 'start': '2026-01-21', 'end': '2026-01-27'},  # 4주차
    ]

    # === 3. 주차별 바이럴 인덱스 계산 ===
    print("\n[2] 주차별 바이럴 인덱스 계산...")

    # 기존 바이럴 데이터 로드 (연속성 위해)
    old_viral = pd.read_csv('/Users/minzzy/Desktop/statrack/book-review-analysis/analysis/viral_index/weekly_news_viral_index_revised.csv')
    old_counts = old_viral.pivot_table(index='ymw', columns='category', values='article_count', fill_value=0)

    all_cats = list(CATEGORY_KEYWORDS.keys()) + ['미분류']

    # 새 주차 기사 수 집계
    new_counts = []
    for w in weeks_2026_jan:
        start, end = pd.to_datetime(w['start']), pd.to_datetime(w['end'])
        week_news = news[(news['date'] >= start) & (news['date'] <= end)]
        counts = week_news['category'].value_counts().reindex(all_cats, fill_value=0)
        row = {'ymw': w['ymw']}
        row.update(counts.to_dict())
        new_counts.append(row)

    new_counts_df = pd.DataFrame(new_counts).set_index('ymw')

    # 기존 + 새 데이터 합쳐서 바이럴 계산
    combined = pd.concat([old_counts, new_counts_df])
    viral_combined = calculate_viral_index(combined)

    # 새 주차만 추출
    new_viral = viral_combined.loc[['2026011', '2026012', '2026013', '2026014']]

    print("\n  2026년 1월 주차별 바이럴 인덱스:")
    print(new_viral.round(2).to_string())

    # === 4. 기존 ML 데이터에서 최근 책 정보 가져오기 ===
    print("\n[3] 예측 대상 책 준비...")
    # Data is in the parent directory
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ml_data = pd.read_csv(os.path.join(BASE_DIR, '..', 'books_ml_dataset_v4.csv'))

    # 마지막 주차(2025124) 데이터에서 책 목록
    last_week = ml_data[ml_data['ymw'] == 2025124].copy()
    print(f"  2025년 12월 4주차 책: {len(last_week)}권")

    # === 5. 예측용 피처 생성 ===
    print("\n[4] 2026년 1월 1주차 예측...")

    # 카테고리 컬럼 (category_1 ~ category_10)
    cat_cols = [c for c in ml_data.columns if c.startswith('category_') and 'x_viral' not in c]

    # 선택된 8개 피처 (Feature Selection 결과)
    selected_features = ['category_4', 'category_10', 'prophet_forecast_stock_trading',
                         'category_3', 'kospi', 'category_9',
                         'prophet_forecast_financial_crisis', 'category_6']

    # 2026년 1월 1주차 바이럴 인덱스
    viral_2026011 = new_viral.loc['2026011']

    # 피처 매핑 (v4 데이터셋의 prophet_forecast는 category × 예측값)
    # 여기서는 바이럴 인덱스를 prophet 예측값 대신 사용
    cat_to_num = {
        'macro_economy': 1, 'econ_theory': 2, 'real_estate': 3, 'business': 4,
        'personal_finance': 5, 'invest_philosophy': 6, 'stock_trading': 7,
        'financial_crisis': 8, 'tech_startup': 9, 'geopolitics': 10
    }

    # === LightGBM 모델 학습 (전체 2025년 데이터) ===
    feature_cols = selected_features + ['y_lag1']

    # y_lag1 생성: 같은 책의 전주 판매점수
    df_lag = ml_data.copy()
    df_lag['y_lag1'] = df_lag.groupby('product_code')['y_sales_score'].shift(1)
    df_lag = df_lag.dropna(subset=['y_lag1'])

    X_train = df_lag[feature_cols]
    y_train = df_lag['y_sales_score']

    model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    from sklearn.metrics import r2_score
    train_r2 = r2_score(y_train, model.predict(X_train))
    print(f"  LightGBM Train R²: {train_r2:.4f}")

    # === 예측: 2025년 12월 4주차 → 2026년 1월 1주차 ===
    last_week['y_lag1'] = last_week['y_sales_score']
    X_pred = last_week[feature_cols]
    last_week['pred_score'] = np.maximum(model.predict(X_pred), 0)

    pred_df = last_week[['product_code', 'y_lag1', 'pred_score']].copy()
    pred_df = pred_df.sort_values('pred_score', ascending=False).reset_index(drop=True)
    pred_df['pred_rank'] = range(1, len(pred_df)+1)

    # === 6. 결과 출력 ===
    print("\n" + "="*60)
    print("2026년 1월 1주차 베스트셀러 예측 Top 20")
    print("="*60)
    print(f"{'순위':<6}{'책 코드':<15}{'전주 점수':<12}{'예측 점수':<12}")
    print("-"*60)

    for _, row in pred_df.head(20).iterrows():
        print(f"{row['pred_rank']:<6}{row['product_code']:<15}{row['y_lag1']:<12.2f}{row['pred_score']:<12.2f}")

    # 예측 결과 저장
    pred_df.to_csv('prediction_2026_jan_week1.csv', index=False)
    print(f"\n예측 결과 저장: prediction_2026_jan_week1.csv")

    # === 7. 바이럴 인덱스 인사이트 ===
    print("\n" + "="*60)
    print("2026년 1월 바이럴 인덱스 인사이트")
    print("="*60)

    # 가장 핫한 카테고리
    hot_cats = viral_2026011.sort_values(ascending=False).head(3)
    print("\n[가장 핫한 카테고리 Top 3]")
    for cat, val in hot_cats.items():
        print(f"  {cat}: {val:.2f}")

    # 기사 수 변화
    print("\n[2026년 1월 1주차 기사 수]")
    for cat, cnt in new_counts_df.loc['2026011'].sort_values(ascending=False).head(5).items():
        print(f"  {cat}: {int(cnt)}개")


if __name__ == "__main__":
    main()

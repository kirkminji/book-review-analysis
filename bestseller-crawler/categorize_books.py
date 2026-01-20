#!/usr/bin/env python3
"""
books 테이블의 keywords를 기반으로 카테고리 분류
연관성 순서대로 category_1, category_2, category_3에 저장
"""

import os
from dotenv import load_dotenv
from collections import Counter

load_dotenv()

from supabase import create_client
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# 카테고리 매핑 테이블
CATEGORY_KEYWORDS = {
    "주식투자/트레이딩": [
        "주식투자", "트레이더", "나스닥", "코스피", "종목", "매매", "손절매", "시가총액",
        "급락", "급등", "저점", "고점", "상승세", "외국인", "etf", "코스닥", "거래량",
        "시총", "변동성", "목표주가", "순매도", "레버리지", "서학개미", "매수세",
        "밸류에이션", "대형주", "공공기관", "구조조정", "상장지수펀드", "금융기관",
        "컨센서스", "사모펀드", "트레이딩", "개인 투자자", "초보 투자자", "투자자",
        "배당", "수익", "포트폴리오", "분산 투자", "장기 투자", "시세 차익", "밸류",
        "차트", "개미", "파동", "매수", "매도"
    ],
    "투자철학/대가": [
        "워런 버핏", "버핏", "가치 투자", "투자 철학", "서한", "주주", "명언",
        "필립 피셔", "피터 린치", "하워드 막스", "주주가치", "피셔", "린치",
        "하워드", "필립", "보통주", "투자 원칙", "통찰", "부의 공식", "마인드"
    ],
    "재테크/개인금융": [
        "재테크", "부자되는법", "종잣돈", "절세", "노후 준비", "연말 정산", "배당금",
        "현금 흐름", "원금", "계좌", "국민연금", "퇴직연금", "노후 자금", "자산",
        "절세 방법", "금융 지식", "재투자", "퇴직 연금", "배당 소득", "월배당",
        "돈 공부", "부자", "절약"
    ],
    "거시경제/금융정책": [
        "금리", "인플레이션", "환율", "통화 정책", "기준 금리", "경기 순환",
        "디플레이션", "버블", "중앙은행", "한국은행", "연준", "한은", "기준금리",
        "유동성", "수출액", "달러", "gdp", "고환율", "유로", "거시 경제", "거시경제",
        "금리 인상", "경제 원리", "경제 개념", "글로벌 경제", "한국 경제", "실물 경제",
        "트렌드", "전망", "소비", "미래", "시대예보", "인구"
    ],
    "지정학/국제정세": [
        "트럼프", "우크라", "중국", "패권", "관세", "국제 질서", "지정학", "국가 전략",
        "자유 무역", "이스라엘", "공급망", "대만", "중동", "러시아", "국제 정세",
        "국제 정치", "도널드 트럼프", "국가 안보", "제2차 냉전", "양극", "항로", "물류"
    ],
    "부동산/실물자산": [
        "부동산 투자", "주택 가격", "집값", "건폐율", "용도지역", "금", "실물",
        "부동산", "재건축", "분양가", "금융당국", "금융사", "실거래", "보험금",
        "재개발", "주담대", "보증금", "금융권", "원자재", "보조금", "토지거래허가구역",
        "투자금", "갭투자", "지원금", "정비사업", "과징금", "금값", "원리금", "계약금",
        "임대료", "다주택자", "증거금", "임차인", "주택담보대출", "전셋값", "무주택자",
        "건물주", "월세", "경매", "아파트"
    ],
    "기업경영/리더십": [
        "리더십", "경영자", "비즈니스 모델", "브랜드 전략", "경쟁력", "혁신 기업",
        "매출", "다각화", "영업이익", "브랜드", "임직원", "ceo", "매출액", "이사회",
        "순이익", "상장사", "경영진", "ipo", "경영", "상장", "지배구조", "최고경영자",
        "덕목", "대전환", "행동 방식", "실행력", "조직", "재벌", "회장", "기업가",
        "대행사", "강의"
    ],
    "테크/스타트업": [
        "실리콘밸리", "스타트업", "AI", "프롬프트", "에이전트", "반도체", "휴머노이드",
        "오픈소스", "ai", "인공지능", "전기차", "클라우드", "빅테크", "데이터센터",
        "hbm", "자율주행", "로보틱스", "ces", "파운드리", "빅데이터", "오픈ai", "낸드",
        "중소벤처기업부", "드론", "실리콘", "밸리", "창업자", "오픈", "신경망", "컴퓨팅",
        "병렬", "반도체 산업", "엔비디아", "클로드", "트랜스포머", "모달", "커서",
        "플랫폼", "창업"
    ],
    "경제이론/학술": [
        "거시 경제학", "미시 경제학", "케인스", "하이에크", "경쟁 시장", "외부 효과",
        "노벨 경제학", "행동경제학", "연구개발", "연구소", "실수요자", "수요예측",
        "효율성", "공급", "수요자", "경제학", "경제이론", "국부론", "생산요소시장",
        "자본주의"
    ],
    "금융시스템/위기": [
        "금융 위기", "금융 시스템", "화폐", "기축 통화", "부채", "가계 부채",
        "글로벌 금융 위기", "코인", "비트코인", "암호 화폐", "알트코인", "국제 금융 시장",
        "가상자산", "스테이블코인"
    ]
}

def categorize_book(keywords_str: str) -> list:
    """키워드 문자열을 분석하여 연관성 순으로 카테고리 반환 (최대 3개)"""
    if not keywords_str:
        return []

    # 키워드 분리 (쉼표 구분)
    keywords = [k.strip().lower() for k in keywords_str.split(',')]

    # 각 카테고리별 매칭 점수 계산
    category_scores = Counter()

    for keyword in keywords:
        for category, cat_keywords in CATEGORY_KEYWORDS.items():
            # 키워드가 카테고리 키워드에 포함되는지 확인 (부분 매칭)
            for cat_kw in cat_keywords:
                if keyword in cat_kw.lower() or cat_kw.lower() in keyword:
                    category_scores[category] += 1
                    break

    # 점수가 1 이상인 카테고리만 선택, 점수 순 정렬
    ranked = [cat for cat, score in category_scores.most_common() if score >= 1]

    # 최대 3개까지만 반환
    return ranked[:3]


def main():
    print("=" * 60)
    print("[도서 카테고리 분류]")
    print("=" * 60)

    # category_1이 없는 도서만 가져오기
    response = supabase.table('books').select('product_code, title, keywords').is_('category_1', 'null').execute()
    books = response.data

    if not books:
        print("분류할 도서가 없습니다.")
        return

    print(f"총 {len(books)}개 도서 분류 시작\n")

    categorized_count = 0
    no_category_count = 0

    for idx, book in enumerate(books, 1):
        product_code = book['product_code']
        title = book['title'] or ""
        keywords = book['keywords'] or ""

        # 카테고리 분류 (키워드 + 제목 활용)
        categories = categorize_book(keywords + "," + title if keywords else title)

        # 업데이트 데이터 준비
        update_data = {
            'category_1': categories[0] if len(categories) > 0 else None,
            'category_2': categories[1] if len(categories) > 1 else None,
            'category_3': categories[2] if len(categories) > 2 else None,
        }

        # Supabase 업데이트
        supabase.table('books').update(update_data).eq('product_code', product_code).execute()

        # 결과 출력
        cat_str = " > ".join(categories) if categories else "(없음)"
        print(f"[{idx}/{len(books)}] {title[:30]}...")
        print(f"    >> {cat_str}")

        if categories:
            categorized_count += 1
        else:
            no_category_count += 1

    print("\n" + "=" * 60)
    print(f"[완료] 카테고리 분류됨: {categorized_count}개")
    print(f"       카테고리 없음: {no_category_count}개")
    print("=" * 60)


if __name__ == "__main__":
    main()

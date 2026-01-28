from pytrends.request import TrendReq
from datetime import datetime
from calendar import monthrange
import re
from typing import List, Dict
import json
import sys
import io
import csv
import time
import random

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class EconomyTrendsAnalyzer:
    def __init__(self):
        """경제 트렌드 분석기 초기화"""
        self.pytrends = TrendReq(hl='ko', tz=540)  # 한국어, 한국 시간대

    def get_economy_trends_by_month(self, year: int, month: int, top_n: int = 30) -> List[Dict]:
        """
        특정 월의 경제 관련 Google Trends 키워드 수집

        Args:
            year: 연도
            month: 월
            top_n: 수집할 키워드 수

        Returns:
            경제 트렌드 키워드 리스트
        """
        start_date = f"{year}-{month:02d}-01"
        last_day = monthrange(year, month)[1]
        end_date = f"{year}-{month:02d}-{last_day}"

        print(f"🔍 {year}년 {month}월 경제 트렌드 수집 중... (기간: {start_date} ~ {end_date})")

        keywords = []

        try:
            timeframe = f'{start_date} {end_date}'

            # 경제/경영 관련 seed 키워드들
            seed_keywords = [
                # 거시경제
                '금리', '환율', '인플레이션', '물가', 'GDP', '경기',
                '실업률', '경제 성장', '경제 위기', '경제 정책',

                # 금융/투자
                '주식', '코스피', '코스닥', '나스닥', '다우', 'S&P',
                '비트코인', '이더리움', '가상화폐', '암호화폐',
                '펀드', '채권', '금', '달러', '엔화',

                # 부동산
                '부동산', '아파트', '전세', '월세', '집값',
                '청약', '분양', '재개발', '재건축',

                # 기업/산업
                '삼성', 'SK', '현대', 'LG', '네이버', '카카오',
                '테슬라', '애플', '엔비디아', '마이크로소프트',
                'AI', '반도체', '배터리', '전기차',

                # 재테크/소비
                '재테크', '저축', '예금', '적금', '연금', '보험',
                '신용카드', '대출', '이자', '세금', '소득세',
                '소비', '할인', '배송', '쇼핑', '이커머스'
            ]

            collected_keywords = {}

            # 일반적인 단어 필터
            generic_words = {
                '경제', '경영', '금융', '투자', '재테크', '뉴스',
                '정보', '분석', '전망', '예측', '시장', '증시'
            }

            # 각 seed 키워드로 관련 검색어 수집
            for idx, keyword in enumerate(seed_keywords[:20], 1):  # 20개 seed 사용
                try:
                    print(f"  {idx}/20: '{keyword}' 관련 검색어 수집 중...")

                    # 해당 키워드의 관련 검색어 가져오기
                    self.pytrends.build_payload(
                        [keyword],
                        cat=7,  # 비즈니스 카테고리
                        timeframe=timeframe,
                        geo='KR'
                    )

                    # 관련 검색어 (rising - 급상승 검색어 우선)
                    try:
                        related_queries = self.pytrends.related_queries()

                        # Rising queries (급상승 검색어)
                        if keyword in related_queries and related_queries[keyword]['rising'] is not None:
                            rising = related_queries[keyword]['rising']
                            for _, row in rising.iterrows():
                                query = row['query'].strip()
                                value = row['value'] if row['value'] != 'Breakout' else 10000

                                # 일반 카테고리 단어 제외
                                if query.lower() not in generic_words and len(query) > 1:
                                    if query not in collected_keywords:
                                        collected_keywords[query] = {
                                            'keyword': query,
                                            'count': 0,
                                            'total_engagement': 0,
                                            'viral_score': 0,
                                            'avg_engagement': 0,
                                        }
                                    collected_keywords[query]['count'] += 1
                                    collected_keywords[query]['total_engagement'] += value
                                    collected_keywords[query]['viral_score'] += value

                        # Top queries도 수집
                        if keyword in related_queries and related_queries[keyword]['top'] is not None:
                            top_related = related_queries[keyword]['top']
                            for _, row in top_related.iterrows():
                                query = row['query'].strip()
                                value = row['value']

                                if query.lower() not in generic_words and len(query) > 1:
                                    if query not in collected_keywords:
                                        collected_keywords[query] = {
                                            'keyword': query,
                                            'count': 0,
                                            'total_engagement': value,
                                            'viral_score': value,
                                            'avg_engagement': value,
                                        }
                                    else:
                                        collected_keywords[query]['total_engagement'] += value
                                        collected_keywords[query]['viral_score'] += value * 0.5
                                    collected_keywords[query]['count'] += 1
                    except Exception as e:
                        pass

                    # Rate limit 방지
                    time.sleep(random.uniform(1.5, 2.5))

                except Exception as e:
                    print(f"    ⚠️ '{keyword}' 처리 실패: {e}")
                    continue

            # 수집된 키워드 정리
            keywords = []
            for kw_data in collected_keywords.values():
                if kw_data['count'] > 0:
                    kw_data['avg_engagement'] = int(kw_data['total_engagement'] / kw_data['count'])
                keywords.append(kw_data)

            # Viral score 기준 정렬
            keywords.sort(key=lambda x: x['viral_score'], reverse=True)

            # 필터링
            filtered_keywords = []
            for kw in keywords:
                kw_lower = kw['keyword'].lower()
                if (kw_lower not in generic_words and
                    len(kw['keyword']) > 1 and
                    not kw['keyword'].isdigit()):
                    filtered_keywords.append(kw)

            print(f"✅ {len(filtered_keywords)}개 경제 트렌드 키워드 수집 완료")

            return filtered_keywords[:top_n]

        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return []

    def analyze_year_by_month(self, year: int = 2025, analyze_full_year: bool = False, top_n: int = 30):
        """연도별 월별 경제 트렌드 분석"""

        results = {}

        # 전체 연도 분석 또는 현재 월까지만 분석
        if analyze_full_year or datetime.now().year > year:
            current_month = 12
        else:
            current_month = datetime.now().month if datetime.now().year == year else 12

        for month in range(1, current_month + 1):
            print(f"\n{'='*50}")
            print(f"📅 {year}년 {month}월 경제 분석 시작")
            print(f"{'='*50}")

            # 트렌드 키워드 수집
            keywords = self.get_economy_trends_by_month(year, month, top_n=top_n)

            if keywords:
                results[f"{year}-{month:02d}"] = keywords

                # 결과 출력
                print(f"\n🏆 {year}년 {month}월 Top {min(len(keywords), top_n)} 경제 트렌드:")
                print("-" * 80)
                for i, kw in enumerate(keywords[:10], 1):
                    print(f"{i:2d}. {kw['keyword']:30s} | "
                          f"Viral Score: {kw['viral_score']:8.2f} | "
                          f"Engagement: {kw['total_engagement']:8,d}")
                if len(keywords) > 10:
                    print(f"... (나머지 {len(keywords) - 10}개 키워드는 파일에 저장됩니다)")
            else:
                print(f"⚠️ {year}년 {month}월: 경제 트렌드를 수집하지 못했습니다.")

            # 월별로 대기
            if month < current_month:
                print(f"\n⏳ 다음 월 수집을 위해 잠시 대기 중...")
                time.sleep(random.uniform(5, 10))

        return results

    def save_results(self, results: Dict, filename: str = "economy_trends_2025.json"):
        """결과 저장 (JSON)"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 결과가 {filename}에 저장되었습니다.")

    def save_results_to_csv(self, results: Dict, filename: str = "economy_trends_2025.csv"):
        """결과를 CSV 파일로 저장"""
        with open(filename, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)

            # 헤더 작성
            writer.writerow(['연월', '순위', '키워드', '출현횟수', '총_Engagement', '평균_Engagement', 'Viral_Score'])

            # 데이터 작성
            for month, keywords in results.items():
                for i, kw in enumerate(keywords, 1):
                    writer.writerow([
                        month,
                        i,
                        kw['keyword'],
                        kw.get('count', 1),
                        kw['total_engagement'],
                        kw['avg_engagement'],
                        round(kw['viral_score'], 2)
                    ])

        print(f"💾 CSV 결과가 {filename}에 저장되었습니다.")


# 실행
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 2025년 1월~12월 경제 분야 Google Trends 분석 시작")
    print("="*80)
    print("\n📊 Google Trends를 사용하여 한국의 월별 경제 트렌드 키워드를 수집합니다.")
    print("⏳ 월별로 약 3-4분 소요되며, 총 30-40분 정도 걸릴 수 있습니다.\n")

    # 분석기 초기화
    analyzer = EconomyTrendsAnalyzer()

    try:
        # 2025년 전체(1월~12월) 월별 경제 트렌드 분석
        results = analyzer.analyze_year_by_month(
            year=2025,
            analyze_full_year=True,  # 전체 연도 분석
            top_n=30                  # 월별 Top 30 키워드
        )

        # 결과가 있는 경우에만 저장
        if results:
            # 결과 저장 (JSON)
            analyzer.save_results(results)

            # 결과 저장 (CSV)
            analyzer.save_results_to_csv(results)

            # 전체 요약 출력
            print("\n" + "="*80)
            print("📊 2025년 경제 트렌드 전체 요약")
            print("="*80)

            for month, keywords in results.items():
                print(f"\n{month}:")
                top_5 = keywords[:5]
                for i, kw in enumerate(top_5, 1):
                    print(f"  {i}. {kw['keyword']} (Viral Score: {kw['viral_score']:.2f})")

            print(f"\n✅ 분석 완료! 총 {len(results)}개월 경제 데이터 수집")
        else:
            print("\n❌ 수집된 데이터가 없습니다.")
            print("네트워크 연결 또는 Google Trends API 상태를 확인하세요.")

    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()

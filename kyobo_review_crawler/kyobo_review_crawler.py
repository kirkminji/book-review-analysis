#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
교보문고 리뷰 크롤러 (Supabase 연동)
- books 테이블에서 도서 목록 가져와서 리뷰 수집
- reviews 테이블에 저장
"""

import requests
import pandas as pd
from typing import List, Dict, Optional
import time
import sys
import platform
import os
from dotenv import load_dotenv

# .env 파일 로드 (상위 디렉토리의 bestseller-crawler/.env 사용)
env_path = os.path.join(os.path.dirname(__file__), '..', 'bestseller-crawler', '.env')
load_dotenv(env_path)

# Windows 콘솔 인코딩 설정
if platform.system() == 'Windows':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass

# Supabase 설정
try:
    from supabase import create_client, Client
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    if SUPABASE_URL and SUPABASE_KEY:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        SUPABASE_ENABLED = True
        print("[Supabase] 연결 설정 완료")
    else:
        SUPABASE_ENABLED = False
        print("[Supabase] 환경변수 미설정 - CSV만 저장됩니다")
except ImportError:
    SUPABASE_ENABLED = False
    print("[Supabase] 라이브러리 미설치 - CSV만 저장됩니다")


def get_books_from_supabase() -> List[Dict]:
    """
    Supabase books 테이블에서 도서 목록 가져오기
    """
    if not SUPABASE_ENABLED:
        print("[오류] Supabase가 연결되지 않았습니다.")
        return []

    try:
        result = supabase.table('books').select('product_code, title').execute()
        books = result.data
        print(f"[Supabase] {len(books)}개 도서 목록 로드 완료")
        return books
    except Exception as e:
        print(f"[오류] 도서 목록 로드 실패: {e}")
        return []


def fetch_book_reviews(product_code: str, max_pages: int = 30) -> List[Dict]:
    """
    교보문고 API에서 책 리뷰 정보를 가져옵니다.

    Args:
        product_code: 상품 코드 (예: S000217467412)
        max_pages: 가져올 최대 페이지 수

    Returns:
        리뷰 정보 리스트
    """
    base_url = "https://product.kyobobook.co.kr/api/review/list"
    all_reviews = []

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'
    }

    for page in range(1, max_pages + 1):
        params = {
            'page': page,
            'pageLimit': 10,
            'reviewSort': '001',
            'revwPatrCode': '002',
            'saleCmdtids': product_code,
            'webToonYsno': 'N',
            'allYsno': 'N',
            'revwSummeryYn': 'Y',
            'saleCmdtid': product_code
        }

        try:
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()

            data = response.json()

            if 'data' in data and 'reviewList' in data['data']:
                reviews = data['data']['reviewList']

                if not reviews:
                    break

                # product_code 추가
                for review in reviews:
                    review['product_code'] = product_code

                all_reviews.extend(reviews)
            else:
                break

            time.sleep(0.3)  # API 요청 간격

        except requests.exceptions.RequestException as e:
            print(f"    >> 페이지 {page} 수집 중 오류: {e}")
            break

    return all_reviews


def parse_reviews(reviews: List[Dict]) -> List[Dict]:
    """
    리뷰 데이터를 DB 저장 형식으로 변환
    """
    parsed = []

    for review in reviews:
        row = {
            'product_code': review.get('product_code', ''),
            'review_content': review.get('revwCntt', '').strip(),
            'rating': review.get('revwRvgr', 0),
            'emotion_keyword': review.get('revwEmtnKywrName', ''),
            'reviewer_id': review.get('mmbrId', '익명'),
            'review_date': review.get('cretDttm', '')[:10] if review.get('cretDttm') else None,
            'helpful_count': review.get('reviewRecommendCount', 0),
            'comment_count': review.get('reviewCommentCount', 0)
        }
        parsed.append(row)

    return parsed


def save_reviews_to_supabase(reviews: List[Dict]) -> bool:
    """
    리뷰 데이터를 Supabase에 저장
    """
    if not SUPABASE_ENABLED or not reviews:
        return False

    try:
        # 배치로 저장 (한 번에 100개씩)
        batch_size = 100
        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i + batch_size]
            supabase.table('reviews').insert(batch).execute()

        return True
    except Exception as e:
        print(f"    >> Supabase 저장 오류: {e}")
        return False


def crawl_all_reviews(max_pages_per_book: int = 30):
    """
    books 테이블의 모든 도서에 대해 리뷰 수집
    """
    print("=" * 70)
    print("[교보문고 리뷰 크롤러 - Supabase 연동]")
    print("=" * 70)

    # 도서 목록 가져오기
    books = get_books_from_supabase()

    if not books:
        print("\n[오류] 수집할 도서가 없습니다.")
        print("먼저 베스트셀러 크롤러를 실행해서 books 테이블을 채워주세요.")
        return

    print(f"\n총 {len(books)}개 도서의 리뷰를 수집합니다.")
    print(f"페이지 제한: 도서당 최대 {max_pages_per_book}페이지 (약 {max_pages_per_book * 10}개 리뷰)")
    print()

    total_reviews = 0
    all_reviews_data = []

    for idx, book in enumerate(books, 1):
        product_code = book['product_code']
        title = book['title'][:30] if book.get('title') else product_code

        print(f"[{idx}/{len(books)}] {title}...")

        # 리뷰 수집
        reviews = fetch_book_reviews(product_code, max_pages=max_pages_per_book)

        if reviews:
            parsed = parse_reviews(reviews)

            # Supabase에 저장
            if save_reviews_to_supabase(parsed):
                print(f"    >> {len(reviews)}개 리뷰 저장 완료")
            else:
                print(f"    >> {len(reviews)}개 리뷰 수집 (DB 저장 실패)")

            all_reviews_data.extend(parsed)
            total_reviews += len(reviews)
        else:
            print(f"    >> 리뷰 없음")

        time.sleep(0.5)  # 도서 간 대기

    # CSV 백업 저장
    if all_reviews_data:
        df = pd.DataFrame(all_reviews_data)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"kyobo_reviews_all_{timestamp}.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n[CSV 백업] {filename}")

    print()
    print("=" * 70)
    print(f"[완료] 총 {total_reviews}개 리뷰 수집")
    print(f"       {len(books)}개 도서 처리")
    print("=" * 70)


def main():
    """
    메인 실행 함수
    """
    # 명령줄 인자로 페이지 수 지정 가능
    max_pages = 30
    if len(sys.argv) > 1:
        try:
            max_pages = int(sys.argv[1])
        except ValueError:
            print("경고: 잘못된 페이지 수 입력, 기본값 30 사용")

    crawl_all_reviews(max_pages_per_book=max_pages)


if __name__ == "__main__":
    main()

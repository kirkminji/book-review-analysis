#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
교보문고 주간 베스트셀러 크롤러
- 전체 카테고리 주간 베스트셀러 수집
- 각 책의 상세 정보, 소개글, 키워드 수집
- Playwright 기반 동적 페이지 크롤링
"""

import argparse
import asyncio
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Optional, List
from datetime import datetime
import re
import sys
import platform
import os
from dotenv import load_dotenv

load_dotenv()

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

# Windows 콘솔 인코딩 설정
if platform.system() == 'Windows':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass

try:
    from playwright.async_api import async_playwright, Page
except ImportError:
    print("Playwright가 설치되어 있지 않습니다.")
    print("설치 명령어: pip install playwright && playwright install chromium")
    sys.exit(1)


@dataclass
class BookInfo:
    """책 정보 데이터 클래스"""
    rank: int
    title: str
    author: str
    translator: Optional[str]
    publisher: str
    publish_date: str
    price: int
    isbn: str
    product_code: str
    rating: float
    review_count: int
    description: str
    intro_text: str
    keywords: List[str]
    image_url: str
    product_url: str
    bestseller_week: str  # 베스트셀러 주간 (예: 2025-01-13 ~ 2025-01-19)
    ymw: str = ""  # ymw 파라미터 값 (예: 2025012)


def generate_ymw_list(start_year: int, start_month: int, start_week: int,
                       end_year: int, end_month: int, end_week: int) -> List[str]:
    """
    ymw 파라미터 리스트 생성
    예: 2025년 1월 2째주 ~ 2025년 12월 4째주
    """
    ymw_list = []

    for year in range(start_year, end_year + 1):
        month_start = start_month if year == start_year else 1
        month_end = end_month if year == end_year else 12

        for month in range(month_start, month_end + 1):
            week_start = start_week if (year == start_year and month == start_month) else 1
            week_end = end_week if (year == end_year and month == end_month) else 5  # 최대 5주

            for week in range(week_start, week_end + 1):
                ymw = f"{year}{month:02d}{week}"
                ymw_list.append(ymw)

    return ymw_list


async def get_weekly_bestseller_list_by_ymw(page: Page, ymw: str, top_n: int = 20) -> List[dict]:
    """
    주간 베스트셀러 목록 페이지에서 책 정보 추출 (ymw 파라미터 사용)
    ymw: YYYYMMW 형식 (예: 2025012 = 2025년 1월 2째주)
    """
    url = f"https://store.kyobobook.co.kr/bestseller/total/weekly/economics?page=1&ymw={ymw}"

    print(f"\n[주간 베스트셀러 목록 수집]")
    print(f"URL: {url}")

    response = await page.goto(url, wait_until="domcontentloaded", timeout=60000)
    await asyncio.sleep(3)

    # 페이지 유효성 확인 (404 또는 빈 페이지 체크)
    if response and response.status == 404:
        print(f"  >> 페이지 없음 (404)")
        return []

    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    await asyncio.sleep(2)

    # 현재 주간 정보 추출
    current_week = ""
    try:
        week_span = await page.query_selector('div[class*="w-[200px]"][class*="cursor-pointer"] span')
        if week_span:
            current_week = (await week_span.inner_text()).strip()
        else:
            dropdown = await page.query_selector('div[class*="w-[200px]"][class*="cursor-pointer"]')
            if dropdown:
                full_text = (await dropdown.inner_text()).strip()
                current_week = full_text.split('\n')[0].strip()

        print(f"  >> 현재 선택된 주간: {current_week}")
    except Exception as e:
        print(f"  >> 선택 확인 오류: {e}")

    # 페이지에 데이터가 없는지 확인
    no_data = await page.query_selector('div.no_data, div.empty_data, div:has-text("데이터가 없습니다")')
    if no_data:
        print(f"  >> 해당 주간 데이터 없음")
        return []

    books = []
    seen_urls = set()

    # ol.grid 내의 li 요소들 선택
    items = await page.query_selector_all('ol.grid > li')

    for item in items:
        try:
            # 제목 링크 찾기
            title_link = await item.query_selector('div.ml-4 > a[href*="/detail/"]')
            if not title_link:
                continue

            url = await title_link.get_attribute('href') or ""
            title = (await title_link.inner_text()).strip()

            if not url or not title or url in seen_urls:
                continue

            seen_urls.add(url)
            rank = len(books) + 1

            if not url.startswith('http'):
                url = f"https://product.kyobobook.co.kr{url}"

            # 평점 추출
            rating = 0.0
            rating_elem = await item.query_selector('span.font-bold.text-black')
            if rating_elem:
                try:
                    rating_text = (await rating_elem.inner_text()).strip()
                    rating = float(rating_text)
                except:
                    pass

            # 리뷰수 추출 (예: "(607개의 리뷰)" -> 607)
            review_count = 0
            review_elem = await item.query_selector('span.font-normal.text-gray-700')
            if review_elem:
                try:
                    review_text = (await review_elem.inner_text()).strip()
                    # 숫자만 추출
                    review_text = re.sub(r'[^\d]', '', review_text)
                    if review_text:
                        review_count = int(review_text)
                except:
                    pass

            books.append({
                'rank': rank,
                'title': title,
                'product_url': url,
                'week': current_week,
                'ymw': ymw,
                'rating': rating,
                'review_count': review_count
            })
            print(f"  {rank}위: {title[:40]}... (평점: {rating}, 리뷰: {review_count})")

            if len(books) >= top_n:
                break
        except Exception as e:
            print(f"  >> 아이템 파싱 오류: {e}")
            continue

    print(f"  >> 총 {len(books)}개 책 목록 수집 완료")
    return books


async def get_book_detail(page: Page, product_url: str, rank: int, week_str: str) -> Optional[BookInfo]:
    """
    개별 책 상세 페이지에서 정보 추출
    """
    try:
        await page.goto(product_url, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(3)

        # 제목
        title_elem = await page.query_selector('span.prod_title')
        title = ""
        if title_elem:
            title = (await title_elem.inner_text()).strip()

        # 저자 및 번역자
        author = ""
        translator = None
        author_box = await page.query_selector('div.prod_author_box')
        if author_box:
            author_text = await author_box.inner_text()
            parts = author_text.split('·')
            if len(parts) >= 1:
                author = parts[0].split('저자')[0].strip()
            if len(parts) >= 2 and '번역' in parts[1]:
                translator = parts[1].split('번역')[0].strip()

        # 출판사 및 출판일
        publisher = ""
        publish_date = ""
        publish_info = await page.query_selector('div.prod_info_text.publish_date')
        if publish_info:
            pub_link = await publish_info.query_selector('a.btn_publish_link')
            if pub_link:
                publisher = (await pub_link.inner_text()).strip()

            pub_text = await publish_info.inner_text()
            date_match = re.search(r'\d{4}년 \d{1,2}월 \d{1,2}일', pub_text)
            if date_match:
                publish_date = date_match.group()

        # 가격
        price = 0
        price_elem = await page.query_selector('span.prod_price')
        if price_elem:
            price_text = (await price_elem.inner_text()).strip()
            price_text = price_text.replace(',', '').replace('원', '')
            try:
                price = int(price_text)
            except:
                pass

        # ISBN
        isbn = ""
        isbn_meta = await page.query_selector('meta[property="books:isbn"]')
        if isbn_meta:
            isbn = await isbn_meta.get_attribute('content') or ""

        # 상품코드
        product_code = ""
        code_match = re.search(r'(S\d+)', product_url)
        if code_match:
            product_code = code_match.group(1)

        # 평점
        rating = 0.0
        rating_elem = await page.query_selector('span.review_score')
        if rating_elem:
            try:
                rating = float((await rating_elem.inner_text()).strip())
            except:
                pass

        # 리뷰 수
        review_count = 0
        review_box = await page.query_selector('div.prod_review_box')
        if review_box:
            review_val = await review_box.query_selector('span.val')
            if review_val:
                try:
                    review_count = int((await review_val.inner_text()).strip())
                except:
                    pass

        # 짧은 설명
        description = ""
        desc_elem = await page.query_selector('span.prod_desc')
        if desc_elem:
            description = (await desc_elem.inner_text()).strip()

        # 상세 소개글
        intro_text = ""
        intro_sections = await page.query_selector_all('div.intro_bottom div.info_text')
        for section in intro_sections:
            text = (await section.inner_text()).strip()
            if text and len(text) > 50:
                intro_text += text + "\n\n"
        intro_text = intro_text.strip()

        book_intro = await page.query_selector('div.book_intro div.info_text')
        if book_intro:
            book_intro_text = (await book_intro.inner_text()).strip()
            if book_intro_text:
                intro_text = book_intro_text + "\n\n" + intro_text

        # 키워드 리스트
        keywords = []
        keyword_tabs = await page.query_selector_all('div.product_keyword_pick ul.tabs li.tab_item a span')
        for kw in keyword_tabs:
            kw_text = (await kw.inner_text()).strip()
            if kw_text and kw_text != '더보기':
                keywords.append(kw_text)

        # 이미지 URL
        image_url = ""
        image_meta = await page.query_selector('meta[property="og:image"]')
        if image_meta:
            image_url = await image_meta.get_attribute('content') or ""

        return BookInfo(
            rank=rank,
            title=title,
            author=author,
            translator=translator,
            publisher=publisher,
            publish_date=publish_date,
            price=price,
            isbn=isbn,
            product_code=product_code,
            rating=rating,
            review_count=review_count,
            description=description,
            intro_text=intro_text[:2000] if intro_text else "",
            keywords=keywords,
            image_url=image_url,
            product_url=product_url,
            bestseller_week=week_str
        )

    except Exception as e:
        print(f"    >> 상세 정보 수집 오류: {e}")
        return None


async def collect_weekly_rankings(page: Page, ymw_list: List[str], top_n: int = 20) -> List[dict]:
    """
    1단계: 모든 주간의 베스트셀러 목록 수집
    """
    print("\n" + "="*70)
    print("[1단계] 모든 주간 베스트셀러 목록 수집")
    print("="*70)
    print(f"수집 대상: {len(ymw_list)}개 주간")

    all_rankings: List[dict] = []
    valid_weeks = 0

    for idx, ymw in enumerate(ymw_list, 1):
        print(f"\n[{idx}/{len(ymw_list)}] ymw={ymw} 수집 중...")

        book_list = await get_weekly_bestseller_list_by_ymw(page, ymw, top_n=top_n)

        if not book_list:
            print(f"    >> 데이터 없음 (건너뜀)")
            continue

        week_str = book_list[0].get('week', '')

        # 유효하지 않은 주간 필터링
        if not week_str or not week_str.strip():
            print(f"    >> 유효하지 않은 주간 데이터 (건너뜀)")
            continue

        # ymw 년도와 week_str 년도 비교
        ymw_year = ymw[:4]
        if ymw_year not in week_str:
            print(f"    >> 년도 불일치: ymw={ymw_year}, week={week_str} (건너뜀)")
            continue

        valid_weeks += 1

        for book in book_list[:top_n]:
            code_match = re.search(r'(S\d+)', book['product_url'])
            product_code = code_match.group(1) if code_match else None

            all_rankings.append({
                'week': week_str,
                'ymw': ymw,
                'rank': book['rank'],
                'title': book['title'],
                'product_url': book['product_url'],
                'product_code': product_code,
                'rating': book.get('rating', 0),
                'review_count': book.get('review_count', 0)
            })

        print(f"    >> {week_str}: {len(book_list)}권")

    print(f"\n[1단계 완료] 유효 주간: {valid_weeks}개, 총 {len(all_rankings)}건의 순위 데이터 수집")
    return all_rankings


def save_rankings_to_db(all_rankings: List[dict]) -> List[str]:
    """
    2단계: DB 저장 및 신규 상품코드 반환
    """
    print("\n" + "="*70)
    print("[2단계] DB 저장 및 상세 정보 필요 상품 확인")
    print("="*70)

    if not SUPABASE_ENABLED:
        # Supabase 미사용 시 모든 고유 상품 수집
        unique_codes = set()
        for item in all_rankings:
            if item['product_code']:
                unique_codes.add(item['product_code'])
        print(f"Supabase 미사용 - 전체 {len(unique_codes)}개 상품 수집 필요")
        return list(unique_codes)

    # 고유 상품코드 추출 (set 사용으로 효율화)
    unique_codes = set(item['product_code'] for item in all_rankings if item['product_code'])
    print(f"고유 상품코드: {len(unique_codes)}개")

    # books 테이블에서 이미 있는 상품코드 확인
    existing = supabase.table('books').select('product_code').in_('product_code', list(unique_codes)).execute()
    existing_codes = set(row['product_code'] for row in existing.data)
    print(f"이미 DB에 있는 상품: {len(existing_codes)}개")

    # 새로운 상품코드만 추출
    new_codes = unique_codes - existing_codes
    print(f"새로 수집 필요한 상품: {len(new_codes)}개")

    # 1) books 테이블에 신규 상품코드 먼저 추가 (FK 제약조건 충족)
    if new_codes:
        new_books_data = []
        seen_codes = set()
        for item in all_rankings:
            code = item['product_code']
            if code in new_codes and code not in seen_codes:
                seen_codes.add(code)
                new_books_data.append({
                    'product_code': code,
                    'product_url': item['product_url'],
                    'title': item['title']
                })
        supabase.table('books').insert(new_books_data).execute()
        print(f"books 테이블: {len(new_books_data)}개 신규 상품코드 추가")

    # 2) weekly_bestsellers 테이블에 순위 데이터 저장
    ymw_set = set(item.get('ymw', '') for item in all_rankings if item['product_code'])
    existing_ymw = supabase.table('weekly_bestsellers').select('ymw').in_('ymw', list(ymw_set)).execute()
    existing_ymw_set = set(row['ymw'] for row in existing_ymw.data)

    weekly_data = []
    for item in all_rankings:
        if item['product_code'] and item.get('ymw', '') not in existing_ymw_set:
            weekly_data.append({
                'bestseller_week': item['week'],
                'ymw': item.get('ymw', ''),
                'rank': item['rank'],
                'product_code': item['product_code'],
                'rating': item.get('rating', 0),
                'review_count': item.get('review_count', 0)
            })

    if weekly_data:
        supabase.table('weekly_bestsellers').insert(weekly_data).execute()
        print(f"weekly_bestsellers 테이블: {len(weekly_data)}건 저장")
    else:
        print(f"weekly_bestsellers 테이블: 이미 존재하는 데이터 (스킵)")

    return list(new_codes)


async def fetch_single_detail(context, code: str, url: str, title: str) -> tuple:
    """
    단일 상품 상세 정보 수집 (배치 처리용)
    """
    page = await context.new_page()
    try:
        book_info = await get_book_detail(page, url, 0, "")
        return (code, book_info, title)
    except Exception as e:
        print(f"    >> {title[:30]}... 수집 오류: {e}")
        return (code, None, title)
    finally:
        await page.close()


async def fetch_details_batch(context, products_to_fetch: dict, batch_size: int = 3) -> dict:
    """
    3단계: 신규 상품 상세 정보 배치 수집 (asyncio.gather 사용)
    """
    print("\n" + "="*70)
    print("[3단계] 신규 상품 상세 정보 수집 (배치 처리)")
    print("="*70)

    if not products_to_fetch:
        print("수집할 신규 상품 없음!")
        return {}

    results = {}
    items = list(products_to_fetch.items())
    total = len(items)

    for i in range(0, total, batch_size):
        batch = items[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        print(f"\n[배치 {batch_num}/{total_batches}] {len(batch)}개 상품 병렬 수집 중...")

        tasks = [
            fetch_single_detail(context, code, info['product_url'], info['title'])
            for code, info in batch
        ]

        batch_results = await asyncio.gather(*tasks)

        for code, book_info, title in batch_results:
            results[code] = book_info
            if book_info:
                print(f"    >> {title[:40]}... 수집 완료")

                if SUPABASE_ENABLED:
                    # books 테이블 업데이트
                    update_data = {
                        'isbn': book_info.isbn,
                        'title': book_info.title,
                        'author': book_info.author,
                        'translator': book_info.translator,
                        'publisher': book_info.publisher,
                        'publish_date': book_info.publish_date,
                        'price': book_info.price,
                        'description': book_info.description,
                        'intro_text': book_info.intro_text[:2000] if book_info.intro_text else "",
                        'keywords': ', '.join(book_info.keywords),
                        'image_url': book_info.image_url,
                        'product_url': book_info.product_url
                    }
                    supabase.table('books').update(update_data).eq('product_code', code).execute()

                    # weekly_bestsellers의 rating, review_count 업데이트
                    supabase.table('weekly_bestsellers').update({
                        'rating': book_info.rating,
                        'review_count': book_info.review_count
                    }).eq('product_code', code).execute()
            else:
                print(f"    >> {title[:40]}... 수집 실패")

        # 배치 간 간격
        await asyncio.sleep(1)

    return results


def build_result_books(all_rankings: List[dict]) -> List[BookInfo]:
    """
    최종 결과 생성 (CSV 저장용)
    """
    print("\n" + "="*70)
    print("[완료] 결과 정리")
    print("="*70)

    all_books = []
    for item in all_rankings:
        if item['product_code']:
            book = BookInfo(
                rank=item['rank'],
                title=item['title'],
                author="",
                translator=None,
                publisher="",
                publish_date="",
                price=0,
                isbn="",
                product_code=item['product_code'],
                rating=item.get('rating', 0),
                review_count=item.get('review_count', 0),
                description="",
                intro_text="",
                keywords=[],
                image_url="",
                product_url=item['product_url'],
                bestseller_week=item['week'],
                ymw=item.get('ymw', '')
            )
            all_books.append(book)

    print(f"총 {len(all_books)}건 (상세 정보는 DB에서 조회)")
    return all_books


async def crawl_weekly_bestsellers_by_ymw(ymw_list: List[str], top_n: int = 20) -> List[BookInfo]:
    """
    주간 베스트셀러 크롤링 메인 함수 (ymw 파라미터 사용)
    1단계: 모든 주간의 목록 수집 → weekly_bestsellers 저장
    2단계: 고유 상품코드 books 테이블에 추가
    3단계: 신규 상품만 상세 정보 배치 수집
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        page = await context.new_page()

        # 1단계: 목록 수집
        all_rankings = await collect_weekly_rankings(page, ymw_list, top_n)

        # 2단계: DB 저장 및 신규 상품 확인
        new_codes = save_rankings_to_db(all_rankings)

        # 3단계: 신규 상품 상세 정보 배치 수집
        if new_codes:
            new_codes_set = set(new_codes)
            products_to_fetch = {}
            for item in all_rankings:
                code = item['product_code']
                if code in new_codes_set and code not in products_to_fetch:
                    products_to_fetch[code] = {
                        'product_url': item['product_url'],
                        'title': item['title']
                    }
            await fetch_details_batch(context, products_to_fetch, batch_size=3)

        await browser.close()

    # 최종 결과 생성
    return build_result_books(all_rankings)


def save_to_csv(books: List[BookInfo], filename: str):
    """
    BookInfo 리스트를 CSV 파일로 저장
    """
    data = []
    for book in books:
        row = asdict(book)
        row['keywords'] = ', '.join(book.keywords)
        data.append(row)

    df = pd.DataFrame(data)

    columns = [
        'ymw', 'bestseller_week', 'rank', 'title', 'author', 'translator',
        'publisher', 'publish_date', 'price', 'isbn', 'product_code',
        'rating', 'review_count', 'description', 'intro_text',
        'keywords', 'image_url', 'product_url'
    ]
    df = df[columns]

    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\n[CSV 저장 완료] {filename}")
    print(f"총 {len(df)}개 책 정보 저장")

    return df


async def main():
    """
    메인 실행 함수

    사용법:
      python weekly_bestseller_crawler.py                  # 2025년 전체 크롤링 (1월 2째주 ~ 12월 4째주)
      python weekly_bestseller_crawler.py --ymw 2025012    # 특정 주간만 수집
      python weekly_bestseller_crawler.py --start-month 3 --end-month 6  # 2025년 3월~6월만 수집
      python weekly_bestseller_crawler.py --top 50         # 상위 50위까지 수집
    """
    parser = argparse.ArgumentParser(description='교보문고 주간 베스트셀러 크롤러')
    parser.add_argument('--ymw', type=str, help='특정 주간만 수집 (예: 2025012)')
    parser.add_argument('--start-month', type=int, default=1, help='시작 월 (기본: 1)')
    parser.add_argument('--start-week', type=int, default=2, help='시작 주차 (기본: 2, 1월은 첫째주 없음)')
    parser.add_argument('--end-month', type=int, default=12, help='종료 월 (기본: 12)')
    parser.add_argument('--end-week', type=int, default=4, help='종료 주차 (기본: 4)')
    parser.add_argument('--year', type=int, default=2025, help='수집 연도 (기본: 2025)')
    parser.add_argument('--top', type=int, default=20, help='수집할 순위 (기본: 20)')
    parser.add_argument('--csv', action='store_true', help='CSV 파일로 저장')
    args = parser.parse_args()

    print("="*70)
    print("[교보문고 주간 베스트셀러 크롤러 - ymw 방식]")
    print("="*70)

    # ymw 리스트 생성
    if args.ymw:
        # 특정 주간만 수집
        ymw_list = [args.ymw]
        print(f"\n수집 대상: ymw={args.ymw}")
    else:
        # 범위로 수집
        ymw_list = generate_ymw_list(
            start_year=args.year,
            start_month=args.start_month,
            start_week=args.start_week,
            end_year=args.year,
            end_month=args.end_month,
            end_week=args.end_week
        )
        print(f"\n수집 대상: {args.year}년 {args.start_month}월 {args.start_week}째주 ~ {args.end_month}월 {args.end_week}째주")
        print(f"예상 주간 수: {len(ymw_list)}개 (일부는 존재하지 않을 수 있음)")

    print(f"순위: 상위 {args.top}위")

    # 크롤링 실행
    books = await crawl_weekly_bestsellers_by_ymw(ymw_list=ymw_list, top_n=args.top)

    if not books:
        print("\n[오류] 수집된 데이터가 없습니다.")
        return

    # CSV 저장 (옵션)
    if args.csv:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"kyobo_weekly_bestseller_{timestamp}.csv"
        save_to_csv(books, filename)

    print("\n" + "="*70)
    print(f"[완료] 총 {len(books)}건 수집")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())

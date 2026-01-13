#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
교보문고 베스트셀러 크롤러
- 경제경영 카테고리 월별 베스트셀러 20위 수집
- 각 책의 상세 정보, 소개글, 키워드 수집
- Playwright 기반 동적 페이지 크롤링
"""

import argparse
import asyncio
import calendar
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Optional, List
from datetime import datetime
import re
import sys
import platform
import os
from dotenv import load_dotenv

# .env 파일 로드
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
    rank: int  # 순위
    title: str
    author: str
    translator: Optional[str]
    publisher: str
    publish_date: str
    price: int
    isbn: str
    product_code: str  # 교보문고 상품코드 (S로 시작)
    rating: float
    review_count: int
    description: str  # 짧은 설명
    intro_text: str  # 상세 소개글
    keywords: List[str]  # 키워드 리스트
    image_url: str
    product_url: str
    bestseller_month: str  # 베스트셀러 월 (예: 2025-12)


async def select_dropdown_option(page: Page, dropdown_selector: str, option_text: str, label: str) -> bool:
    """
    드롭다운에서 특정 옵션 선택
    """
    try:
        # 드롭다운 클릭
        dropdown = await page.wait_for_selector(dropdown_selector, timeout=5000)
        if not dropdown:
            print(f"  >> {label} 드롭다운을 찾을 수 없음")
            return False

        await dropdown.click()
        await asyncio.sleep(0.5)

        # 옵션 목록에서 정확히 일치하는 항목 찾기
        options = await page.query_selector_all('ul[role="listbox"] li, div[role="listbox"] div[role="option"], ul.absolute li')
        for opt in options:
            text = (await opt.inner_text()).strip()
            if text == option_text:
                await opt.click()
                await asyncio.sleep(1)
                print(f"  >> {label}: {option_text} 선택 완료")
                return True

        # 대안: 텍스트로 직접 클릭 시도
        await page.click(f'text="{option_text}"')
        await asyncio.sleep(1)
        print(f"  >> {label}: {option_text} 선택 완료 (텍스트 방식)")
        return True

    except Exception as e:
        print(f"  >> {label} 선택 오류: {e}")
        return False


async def get_bestseller_list(page: Page, year: int, month: int) -> List[dict]:
    """
    베스트셀러 목록 페이지에서 20위까지 책 정보 추출
    """
    # 경제경영 카테고리 월별 베스트셀러 URL (기본값으로 접근)
    url = "https://store.kyobobook.co.kr/bestseller/online/monthly/domestic/13"

    print(f"\n[베스트셀러 목록 수집] {year}년 {month}월")
    print(f"URL: {url}")

    await page.goto(url, wait_until="domcontentloaded", timeout=60000)
    await asyncio.sleep(2)

    # 드롭다운 선택 - 기간 형식 (예: "2025.01.01 ~ 2025.01.31")
    last_day = calendar.monthrange(year, month)[1]
    target_period = f"{year}.{month:02d}.01 ~ {year}.{month:02d}.{last_day:02d}"
    print(f"  >> 목표 기간: {target_period}")

    try:
        # 드롭다운 버튼 찾기
        dropdown = await page.query_selector('div.w-\\[200px\\].cursor-pointer, div[class*="w-[200px]"][class*="cursor-pointer"]')
        if dropdown:
            await dropdown.click()
            await asyncio.sleep(1)

            # 옵션 목록에서 해당 기간 찾기
            options = await page.query_selector_all('ul li')
            found = False
            for opt in options:
                text = (await opt.inner_text()).strip()
                if target_period in text:
                    await opt.click()
                    await asyncio.sleep(2)
                    print(f"  >> 기간 선택 완료: {text}")
                    found = True
                    break

            if not found:
                print(f"  >> 목표 기간을 찾을 수 없음, 사용 가능한 옵션들:")
                for opt in options[:5]:
                    text = (await opt.inner_text()).strip()
                    print(f"     - {text}")
                # 드롭다운 닫기
                await page.keyboard.press("Escape")

    except Exception as e:
        print(f"  >> 드롭다운 선택 오류: {e}")

    await asyncio.sleep(2)  # 동적 콘텐츠 로딩 대기

    # 현재 페이지에서 선택된 기간 확인
    try:
        current_btn = await page.query_selector('div.w-\\[200px\\].cursor-pointer span, div[class*="w-[200px]"][class*="cursor-pointer"] span')
        if current_btn:
            text = (await current_btn.inner_text()).strip()
            print(f"  >> 현재 선택된 기간: {text}")
    except Exception as e:
        print(f"  >> 선택 확인 오류: {e}")

    books = []
    seen_urls = set()

    # prod_link 클래스를 가진 모든 상품 링크 찾기
    all_links = await page.query_selector_all('a.prod_link[href*="/detail/"]')

    for link in all_links:
        try:
            url = await link.get_attribute('href') or ""
            if not url or url in seen_urls:
                continue

            # 제목 텍스트 추출
            title = (await link.inner_text()).strip()

            # 유효한 제목 링크만 수집 (새창보기 등 제외)
            if title and len(title) > 2 and "새창보기" not in title:
                seen_urls.add(url)
                rank = len(books) + 1

                if not url.startswith('http'):
                    url = f"https://product.kyobobook.co.kr{url}"

                books.append({
                    'rank': rank,
                    'title': title,
                    'product_url': url
                })
                print(f"  {rank}위: {title[:40]}...")

                if len(books) >= 20:
                    break

        except Exception as e:
            print(f"  >> 링크 파싱 오류: {e}")
            continue

    print(f"  >> 총 {len(books)}개 책 목록 수집 완료")
    return books


async def get_book_detail(page: Page, product_url: str, rank: int, month_str: str) -> Optional[BookInfo]:
    """
    개별 책 상세 페이지에서 정보 추출
    """
    try:
        await page.goto(product_url, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(3)  # 동적 콘텐츠 로딩 대기

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

        # 상품코드 (S로 시작하는 코드) - URL에서 추출
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

        # 상세 소개글 (intro_bottom의 info_text)
        intro_text = ""
        intro_sections = await page.query_selector_all('div.intro_bottom div.info_text')
        for section in intro_sections:
            text = (await section.inner_text()).strip()
            if text and len(text) > 50:  # 짧은 텍스트 제외
                intro_text += text + "\n\n"
        intro_text = intro_text.strip()

        # 책 소개 본문도 수집
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
            intro_text=intro_text[:2000] if intro_text else "",  # 너무 길면 자르기
            keywords=keywords,
            image_url=image_url,
            product_url=product_url,
            bestseller_month=month_str
        )

    except Exception as e:
        print(f"    >> 상세 정보 수집 오류: {e}")
        return None


async def crawl_bestsellers(year: int, months: List[int], top_n: int = 20) -> List[BookInfo]:
    """
    지정된 년도와 월들의 베스트셀러 크롤링
    """
    all_books = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        page = await context.new_page()

        for month in months:
            month_str = f"{year}-{month:02d}"
            print(f"\n{'='*70}")
            print(f"[{month_str}] 베스트셀러 크롤링 시작")
            print('='*70)

            # 베스트셀러 목록 수집
            book_list = await get_bestseller_list(page, year, month)

            # 각 책의 상세 정보 수집
            for idx, book in enumerate(book_list[:top_n], 1):
                print(f"\n[{idx}/{min(len(book_list), top_n)}] {book['title'][:40]}...")

                book_info = await get_book_detail(
                    page,
                    book['product_url'],
                    book['rank'],
                    month_str
                )

                if book_info:
                    all_books.append(book_info)
                    print(f"    >> 수집 완료: {book_info.title[:30]}...")
                else:
                    print(f"    >> 수집 실패")

                await asyncio.sleep(1)  # 서버 부하 방지

        await browser.close()

    return all_books


def save_to_csv(books: List[BookInfo], filename: str):
    """
    BookInfo 리스트를 CSV 파일로 저장
    """
    data = []
    for book in books:
        row = asdict(book)
        # 키워드 리스트를 문자열로 변환
        row['keywords'] = ', '.join(book.keywords)
        data.append(row)

    df = pd.DataFrame(data)

    # 컬럼 순서 정리
    columns = [
        'bestseller_month', 'rank', 'title', 'author', 'translator',
        'publisher', 'publish_date', 'price', 'isbn', 'product_code',
        'rating', 'review_count', 'description', 'intro_text',
        'keywords', 'image_url', 'product_url'
    ]
    df = df[columns]

    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\n[CSV 저장 완료] {filename}")
    print(f"총 {len(df)}개 책 정보 저장")

    return df


def clear_supabase_data():
    """
    Supabase의 기존 데이터 삭제 (--clear 옵션 사용 시에만 호출)
    삭제 순서: reviews -> bestsellers -> books (외래키 참조 순서)
    """
    if not SUPABASE_ENABLED:
        print("[Supabase] 비활성화 상태 - 삭제 건너뜀")
        return False

    try:
        # reviews 테이블 먼저 삭제 (books 참조)
        supabase.table('reviews').delete().neq('id', 0).execute()
        print("[Supabase] reviews 테이블 데이터 삭제 완료")

        # bestsellers 테이블 삭제 (books 참조)
        supabase.table('bestsellers').delete().neq('id', 0).execute()
        print("[Supabase] bestsellers 테이블 데이터 삭제 완료")

        # books 테이블 삭제
        supabase.table('books').delete().neq('product_code', '').execute()
        print("[Supabase] books 테이블 데이터 삭제 완료")

        return True
    except Exception as e:
        print(f"[Supabase 삭제 오류] {e}")
        return False


def save_to_supabase(books: List[BookInfo]):
    """
    BookInfo 리스트를 Supabase에 정규화하여 저장
    - books 테이블: 도서 마스터 (중복 제거)
    - bestsellers 테이블: 기간별 베스트셀러 순위 기록
    """
    if not SUPABASE_ENABLED:
        print("[Supabase] 비활성화 상태 - 저장 건너뜀")
        return False

    try:
        # 1. 도서 마스터 테이블 데이터 준비 (중복 제거)
        books_data = {}
        for book in books:
            if book.product_code and book.product_code not in books_data:
                books_data[book.product_code] = {
                    'product_code': book.product_code,
                    'isbn': book.isbn,
                    'title': book.title,
                    'author': book.author,
                    'translator': book.translator,
                    'publisher': book.publisher,
                    'publish_date': book.publish_date,
                    'price': book.price,
                    'description': book.description,
                    'intro_text': book.intro_text[:2000] if book.intro_text else "",
                    'keywords': ', '.join(book.keywords),
                    'image_url': book.image_url,
                    'product_url': book.product_url
                }

        # 2. 베스트셀러 테이블 데이터 준비 (기간별 기록, 중복 허용)
        bestsellers_data = []
        for book in books:
            if book.product_code:
                bestsellers_data.append({
                    'bestseller_month': book.bestseller_month,
                    'rank': book.rank,
                    'product_code': book.product_code,
                    'rating': book.rating,
                    'review_count': book.review_count
                })

        # 3. 도서 마스터 테이블에 upsert (중복 시 업데이트)
        if books_data:
            supabase.table('books').upsert(
                list(books_data.values()),
                on_conflict='product_code'
            ).execute()
            print(f"[Supabase] books 테이블: {len(books_data)}개 도서 정보 저장")

        # 4. 베스트셀러 테이블에 insert (기간별 기록 유지)
        if bestsellers_data:
            supabase.table('bestsellers').insert(bestsellers_data).execute()
            print(f"[Supabase] bestsellers 테이블: {len(bestsellers_data)}개 순위 기록 저장")

        return True

    except Exception as e:
        print(f"\n[Supabase 저장 오류] {e}")
        return False


async def main():
    """
    메인 실행 함수

    사용법:
      python kyobo_bestseller_crawler.py                    # 기본 (1~12월 수집)
      python kyobo_bestseller_crawler.py --clear            # DB 초기화 후 수집
      python kyobo_bestseller_crawler.py --months 1 2 3     # 특정 월만 수집
      python kyobo_bestseller_crawler.py --year 2024        # 특정 연도 수집
    """
    parser = argparse.ArgumentParser(description='교보문고 베스트셀러 크롤러')
    parser.add_argument('--clear', action='store_true', help='기존 DB 데이터 삭제 후 수집')
    parser.add_argument('--year', type=int, default=2025, help='수집 연도 (기본: 2025)')
    parser.add_argument('--months', type=int, nargs='+', default=list(range(1, 13)), help='수집할 월 (기본: 1~12)')
    args = parser.parse_args()

    print("="*70)
    print("[교보문고 경제경영 베스트셀러 크롤러]")
    print("="*70)

    year = args.year
    months = args.months

    print(f"\n수집 대상: {year}년 {months[0]}월 ~ {months[-1]}월")
    print(f"카테고리: 경제경영 (코드: 13)")
    print(f"순위: 상위 20위")

    # --clear 옵션이 있으면 기존 데이터 삭제
    if args.clear:
        print("\n[경고] 기존 데이터 삭제 옵션이 활성화되었습니다.")
        clear_supabase_data()

    # 크롤링 실행
    books = await crawl_bestsellers(year, months, top_n=20)

    if not books:
        print("\n[오류] 수집된 데이터가 없습니다.")
        return

    # CSV 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"kyobo_bestseller_{year}_all_{timestamp}.csv"
    df = save_to_csv(books, filename)

    # Supabase 저장
    save_to_supabase(books)

    # 미리보기
    print("\n" + "="*70)
    print("[데이터 미리보기]")
    print("-"*70)
    for idx, row in df.head(5).iterrows():
        print(f"\n{row['rank']}위: {row['title']}")
        print(f"   저자: {row['author']}")
        print(f"   출판사: {row['publisher']} | 가격: {row['price']:,}원")
        print(f"   ISBN: {row['isbn']} | 상품코드: {row['product_code']}")
        print(f"   평점: {row['rating']} | 리뷰: {row['review_count']}개")
        if row['keywords']:
            print(f"   키워드: {row['keywords'][:50]}...")

    print("\n" + "="*70)
    print(f"CSV 파일: {filename}")
    print(f"총 수집: {len(books)}개 (12개월 x 20권)")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())

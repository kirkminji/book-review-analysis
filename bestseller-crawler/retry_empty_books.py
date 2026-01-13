#!/usr/bin/env python3
"""
빈 데이터만 재크롤링하는 스크립트
대기 시간을 5초로 늘려서 실행
"""

import asyncio
import pandas as pd
import re
import os
from dotenv import load_dotenv

load_dotenv()

try:
    from supabase import create_client, Client
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    if SUPABASE_URL and SUPABASE_KEY:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        SUPABASE_ENABLED = True
    else:
        SUPABASE_ENABLED = False
except ImportError:
    SUPABASE_ENABLED = False

from playwright.async_api import async_playwright


async def get_book_detail(page, product_code: str) -> dict:
    """개별 책 상세 페이지에서 정보 추출 (대기 시간 5초)"""
    product_url = f"https://product.kyobobook.co.kr/detail/{product_code}"

    try:
        await page.goto(product_url, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(5)  # 대기 시간 5초로 증가

        # 제목
        title_elem = await page.query_selector('span.prod_title')
        title = (await title_elem.inner_text()).strip() if title_elem else ""

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

        # 키워드
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

        return {
            'product_code': product_code,
            'isbn': isbn,
            'title': title,
            'author': author,
            'translator': translator,
            'publisher': publisher,
            'publish_date': publish_date,
            'price': price,
            'description': description,
            'intro_text': intro_text[:2000] if intro_text else "",
            'keywords': ', '.join(keywords),
            'image_url': image_url,
            'product_url': product_url
        }

    except Exception as e:
        print(f"    >> 오류: {e}")
        return None


async def main():
    # 빈 데이터의 product_code 추출
    df = pd.read_csv('kyobo_bestseller_2025_all_20260111_212303.csv')
    empty_rows = df[df['title'].isna() | (df['title'] == '')]

    # 중복 제거
    empty_codes = empty_rows['product_code'].unique().tolist()
    print(f"재크롤링할 고유 상품코드: {len(empty_codes)}개")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        page = await context.new_page()

        success_count = 0
        failed_codes = []

        for idx, code in enumerate(empty_codes, 1):
            print(f"\n[{idx}/{len(empty_codes)}] {code}")

            book_data = await get_book_detail(page, code)

            if book_data and book_data['title']:
                print(f"    >> 성공: {book_data['title'][:30]}...")

                # Supabase 업데이트
                if SUPABASE_ENABLED:
                    try:
                        supabase.table('books').upsert(
                            book_data,
                            on_conflict='product_code'
                        ).execute()
                    except Exception as e:
                        print(f"    >> DB 저장 오류: {e}")

                success_count += 1
            else:
                print(f"    >> 실패")
                failed_codes.append(code)

            await asyncio.sleep(1)

        await browser.close()

    print(f"\n{'='*50}")
    print(f"완료: {success_count}/{len(empty_codes)} 성공")
    if failed_codes:
        print(f"실패한 코드: {failed_codes}")
    print('='*50)


if __name__ == "__main__":
    asyncio.run(main())

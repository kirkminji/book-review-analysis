#!/usr/bin/env python3
"""
제목 기반으로 카테고리 추론하는 스크립트
category_1이 NULL인 도서의 제목을 분석하여 카테고리 분류
"""

import os
from dotenv import load_dotenv
from collections import Counter

load_dotenv()

from supabase import create_client
from categorize_books import CATEGORY_KEYWORDS

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


def categorize_by_title(title: str) -> list:
    """제목을 분석하여 연관성 순으로 카테고리 반환"""
    if not title:
        return []

    title_lower = title.lower()
    category_scores = Counter()

    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in title_lower:
                # 키워드 길이에 비례한 가중치 (더 구체적인 키워드일수록 높은 점수)
                category_scores[category] += len(kw)

    # 점수가 있는 카테고리만 선택
    ranked = [cat for cat, score in category_scores.most_common() if score >= 1]
    return ranked[:3]


def main():
    # 카테고리가 없는 도서 가져오기
    response = supabase.table('books').select('product_code, title').is_('category_1', 'null').execute()
    books = response.data

    print(f"카테고리 없는 도서: {len(books)}개")
    print("=" * 60)

    updated = 0
    still_empty = 0

    for book in books:
        product_code = book['product_code']
        title = book['title'] or ""

        categories = categorize_by_title(title)

        if categories:
            update_data = {
                'category_1': categories[0] if len(categories) > 0 else None,
                'category_2': categories[1] if len(categories) > 1 else None,
                'category_3': categories[2] if len(categories) > 2 else None,
            }
            supabase.table('books').update(update_data).eq('product_code', product_code).execute()
            cat_str = " > ".join(categories)
            print(f"[O] {title[:40]}...")
            print(f"    >> {cat_str}")
            updated += 1
        else:
            print(f"[X] {title[:40]}... (분류 불가)")
            still_empty += 1

    print("=" * 60)
    print(f"[완료] 추가 분류: {updated}개")
    print(f"       여전히 없음: {still_empty}개")


if __name__ == "__main__":
    # 실행하려면 아래 주석 해제
    # main()
    pass

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
books_with_tags_variable_predictions.csv에 판매점수(sales_score) 추가
- DB의 weekly_bestsellers 테이블과 product_code, ymw로 조인
- 판매점수 = 21 - rank (베스트셀러 순위 기반)
"""

import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# Supabase 설정
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_weekly_bestsellers():
    """DB에서 weekly_bestsellers 데이터 로드"""
    print("📂 DB에서 weekly_bestsellers 데이터 로드 중...")

    all_data = []
    page_size = 1000
    offset = 0

    while True:
        res = supabase.table('weekly_bestsellers').select('ymw, rank, product_code').range(offset, offset + page_size - 1).execute()
        data = res.data
        if not data:
            break
        all_data.extend(data)
        offset += page_size
        if len(data) < page_size:
            break
        print(f"  >> 로딩 중... ({len(all_data):,}개)", end="\r")

    df = pd.DataFrame(all_data)
    print(f"\n  >> 로드 완료: {len(df):,}개 행")
    return df

def main():
    print("=" * 60)
    print("[판매점수 추가 스크립트]")
    print("=" * 60)

    # 1. 기존 CSV 로드
    input_path = '/Users/minzzy/Desktop/statrack/book-review-analysis/analysis/prediction/books_with_tags_variable_predictions.csv'
    print(f"\n📂 기존 CSV 로드 중: {input_path}")
    books_df = pd.read_csv(input_path)
    books_df['ymw'] = books_df['ymw'].astype(str)
    print(f"  >> 로드 완료: {len(books_df):,}개 행")

    # 2. DB에서 weekly_bestsellers 로드
    bs_df = fetch_weekly_bestsellers()
    bs_df['ymw'] = bs_df['ymw'].astype(str)

    # 3. 판매점수 계산 (21 - rank)
    bs_df['sales_score'] = 21 - bs_df['rank']

    # 4. 조인 (product_code, ymw 기준)
    print("\n🔗 데이터 조인 중...")
    merged_df = books_df.merge(
        bs_df[['product_code', 'ymw', 'rank', 'sales_score']],
        on=['product_code', 'ymw'],
        how='left'
    )

    # 5. 베스트셀러에 없는 도서는 sales_score = 0
    merged_df['rank'] = merged_df['rank'].fillna(0).astype(int)
    merged_df['sales_score'] = merged_df['sales_score'].fillna(0).astype(int)

    print(f"  >> 조인 완료: {len(merged_df):,}개 행")
    print(f"  >> 베스트셀러 진입 행: {(merged_df['sales_score'] > 0).sum():,}개")
    print(f"  >> 베스트셀러 미진입 행: {(merged_df['sales_score'] == 0).sum():,}개")

    # 6. 새 CSV 저장
    output_path = '/Users/minzzy/Desktop/statrack/book-review-analysis/analysis/prediction/books_with_sales_score.csv'
    merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 60)
    print(f"✅ 저장 완료: {output_path}")
    print(f"   - 총 행 수: {len(merged_df):,}개")
    print(f"   - 새로 추가된 컬럼: rank, sales_score")
    print("=" * 60)

    # 7. 결과 샘플 출력
    print("\n📊 결과 샘플 (sales_score > 0인 행):")
    sample = merged_df[merged_df['sales_score'] > 0][['product_code', 'title', 'ymw', 'rank', 'sales_score']].head(10)
    print(sample.to_string(index=False))

if __name__ == "__main__":
    main()

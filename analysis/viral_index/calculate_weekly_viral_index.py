#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
주간 뉴스 바이럴 지수 계산 스크립트 (DB 베스트셀러 주차 기준)
1. DB(Supabase)에서 모든 뉴스 데이터를 가져옴
2. DB의 'weekly_bestsellers' 테이블에서 주차 정의를 가져옴 (ymw, bestseller_week)
3. 각 베스트셀러 주차 구간에 맞춰 뉴스 기사 수를 집계
4. WoW, MA4 편차, Z-Score를 결합하여 바이럴 지수 및 Smoothing 지수 산출
5. 분석용 CSV 파일로 저장 (weekly_news_viral_index_revised.csv)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
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
        print("[오류] Supabase 환경변수가 설정되지 않았습니다.")
except ImportError:
    SUPABASE_ENABLED = False
    print("[오류] supabase 라이브러리가 설치되지 않았습니다.")

def fetch_news_data_from_db():
    if not SUPABASE_ENABLED:
        return None
    
    print("📂 DB에서 뉴스 데이터를 불러오는 중...")
    table_name = "news_2025_categorized"
    all_data = []
    page_size = 1000
    offset = 0
    
    try:
        while True:
            res = supabase.table(table_name).select('news_date, category').range(offset, offset + page_size - 1).execute()
            data = res.data
            if not data:
                break
            all_data.extend(data)
            offset += page_size
            if len(data) < page_size:
                break
            print(f"  >> 로딩 중... ({len(all_data):,}개)", end="\r")
        
        df = pd.DataFrame(all_data)
        if df.empty:
            print("\n  >> 불러온 데이터가 없습니다.")
            return None
        
        df = df.rename(columns={'news_date': 'date'})
        df['date'] = pd.to_datetime(df['date'])
        print(f"\n  >> 로드 완료: {len(df):,}개 기사")
        return df
    except Exception as e:
        print(f"\n  >> DB 로드 중 오류 발생: {e}")
        return None

def calculate_viral_indices(weekly_counts_df):
    print("\n📊 주간 바이럴 지수 및 Smoothing 지수 산출 중...")
    
    # 지수 구성 요소 계산 (안정성 강화)
    prev_counts = weekly_counts_df.shift(1)
    wow_growth = ((weekly_counts_df - prev_counts) / (prev_counts + 1)) * 100
    
    ma4 = weekly_counts_df.rolling(window=4, min_periods=1).mean()
    ma_deviation = ((weekly_counts_df - ma4) / (ma4 + 1)) * 100
    
    z_scores = (weekly_counts_df - weekly_counts_df.mean()) / (weekly_counts_df.std() + 1e-9)
    
    viral_index = (
        wow_growth.clip(upper=300).fillna(0) * 0.4 +
        ma_deviation.clip(upper=300).fillna(0) * 0.4 +
        z_scores.clip(lower=-3, upper=3).fillna(0) * 10 * 0.2
    )
    
    viral_index_smoothed = viral_index.rolling(window=2, min_periods=1).mean()
    
    print(f"  >> 계산 완료: {len(weekly_counts_df)}개 주차 x {len(weekly_counts_df.columns)}개 카테고리")
    return viral_index, viral_index_smoothed

def main():
    print("=" * 60)
    print("[주간 뉴스 바이럴 지수 산출 (DB 주차 기준)]")
    print("=" * 60)
    
    news_df = fetch_news_data_from_db()
    if news_df is None:
        return

    print("📂 DB에서 베스트셀러 주차 정보를 불러오는 중...")
    try:
        res_bs_weeks = supabase.table('weekly_bestsellers').select('ymw, bestseller_week').order('ymw').execute()
        df_bs_weeks_raw = pd.DataFrame(res_bs_weeks.data).drop_duplicates()
        
        def parse_week_string(week_str):
            parts = week_str.split(' ~ ')
            return pd.to_datetime(parts[0]), pd.to_datetime(parts[1])
        
        df_bs_weeks_raw[['start_date', 'end_date']] = df_bs_weeks_raw['bestseller_week'].apply(lambda x: pd.Series(parse_week_string(x)))
        df_bs_weeks = df_bs_weeks_raw.sort_values('start_date').reset_index(drop=True)

        if df_bs_weeks.empty:
            print("  >> 베스트셀러 주차 정보가 없습니다.")
            return
        
        print(f"  >> 베스트셀러 주차 정보 로드 완료: {len(df_bs_weeks)}개 주차")
    except Exception as e:
        print(f"  >> 베스트셀러 주차 정보 로드 중 오류 발생: {e}")
        return

    all_weekly_news_counts = []
    all_category_names = news_df['category'].unique()
    
    print("\n📊 각 DB 주차별 뉴스 기사 수 집계 중...")
    for idx, row in df_bs_weeks.iterrows():
        ymw_bs = row['ymw']
        bestseller_week_str = row['bestseller_week']
        start_date_bs = row['start_date']
        end_date_bs = row['end_date']
        
        # 해당 베스트셀러 주차 구간에 해당하는 뉴스만 필터링
        week_news_df = news_df[(news_df['date'] >= start_date_bs) & (news_df['date'] <= end_date_bs)]
        
        # 카테고리별 기사 수 집계 (모든 카테고리 포함)
        counts_series = week_news_df['category'].value_counts().reindex(all_category_names, fill_value=0)
        
        for category, count in counts_series.items():
            all_weekly_news_counts.append({
                'ymw': ymw_bs,
                'bestseller_week': bestseller_week_str,
                'category': category,
                'article_count': count,
                'start_date': start_date_bs,
                'end_date': end_date_bs
            })
    
    df_news_counts_long = pd.DataFrame(all_weekly_news_counts)
    
    if df_news_counts_long.empty:
        print("  >> 집계된 뉴스 데이터가 없습니다.")
        return

    # 바이럴 지수 계산을 위해 wide format으로 변환
    df_pivoted_counts = df_news_counts_long.pivot_table(index='ymw', columns='category', values='article_count', fill_value=0)
    df_pivoted_counts.index = df_pivoted_counts.index.astype(str) # ensure ymw is string

    # 바이럴 지수 계산 함수 호출
    viral_index_df, viral_index_smoothed_df = calculate_viral_indices(df_pivoted_counts)

    # 결과를 다시 long format으로 병합
    viral_long = viral_index_df.stack().reset_index()
    viral_long.columns = ['ymw', 'category', 'viral_index']
    
    viral_smoothed_long = viral_index_smoothed_df.stack().reset_index()
    viral_smoothed_long.columns = ['ymw', 'category', 'viral_index_smoothed']
    
    result_df = pd.merge(df_news_counts_long, viral_long, on=['ymw', 'category'], how='left')
    result_df = pd.merge(result_df, viral_smoothed_long, on=['ymw', 'category'], how='left')

    # 최종 컬럼 순서 조정
    result_df = result_df[['ymw', 'bestseller_week', 'category', 'viral_index', 'viral_index_smoothed', 'article_count', 'start_date', 'end_date']]
    
    output_dir = "/Users/minzzy/Desktop/statrack/book-review-analysis/analysis"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, "weekly_news_viral_index_revised.csv")
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 60)
    print(f"✅ 작업 완료: {output_path}")
    print(f"   - 데이터 기간: {result_df['start_date'].min().date()} ~ {result_df['end_date'].max().date()}")
    print(f"   - 총 데이터 행: {len(result_df)}개")
    print("=" * 60)

if __name__ == "__main__":
    main()

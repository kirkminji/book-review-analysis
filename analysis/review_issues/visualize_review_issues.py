import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from supabase import create_client
from dotenv import load_dotenv

# Setup
load_dotenv()
supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
plt.rcParams['font.family'] = 'AppleGothic'  # Mac environment
plt.rcParams['axes.unicode_minus'] = False
SAVE_PATH = 'analysis/review_issues'
os.makedirs(SAVE_PATH, exist_ok=True)

def fetch_all_reviews():
    print("Fetching reviews from database...")
    all_data = []
    start = 0
    page_size = 1000
    while True:
        res = supabase.table('reviews').select('product_code, review_content, review_date, rating').range(start, start + page_size - 1).execute()
        if not res.data: break
        all_data.extend(res.data)
        start += page_size
    return pd.DataFrame(all_data)

def visualize():
    df = fetch_all_reviews()
    if df.empty:
        print("No data found.")
        return

    # 1. Distribution of Reviews per Book (SKEWNESS)
    plt.figure(figsize=(10, 6))
    counts = df['product_code'].value_counts()
    sns.histplot(counts, bins=20, kde=True, color='teal')
    plt.title('도서별 리뷰 수 분포 (데이터 불균형)')
    plt.xlabel('리뷰 개수')
    plt.ylabel('도서 수')
    plt.axvline(counts.mean(), color='red', linestyle='--', label=f'평균: {counts.mean():.1f}')
    plt.legend()
    plt.savefig(f'{SAVE_PATH}/01_review_imbalance.png', dpi=150)
    print("Plot 1 saved.")

    # 2. Temporal Sparsity (Heatmap style)
    df['date'] = pd.to_datetime(df['review_date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['year_week'] = df['date'].dt.to_period('W').astype(str)
    
    # Top 20 books by review count for better visibility in heatmap
    top_books = counts.head(20).index
    pivot_df = df[df['product_code'].isin(top_books)].groupby(['product_code', 'year_week']).size().unstack(fill_value=0)
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_df, cmap='YlGnBu', cbar_kws={'label': '주간 리뷰 수'})
    plt.title('주요 도서별 주차별 리뷰 발생 현황 (시계열 희소성)')
    plt.xlabel('주차 (Year-Week)')
    plt.ylabel('도서 코드')
    plt.savefig(f'{SAVE_PATH}/02_temporal_sparsity.png', dpi=150)
    print("Plot 2 saved.")

    # 3. Content Quality (Review Length)
    df['text_len'] = df['review_content'].str.len().fillna(0)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['text_len'], bins=50, color='orange', kde=True)
    plt.xlim(0, 500) # Most are short
    plt.title('리뷰 텍스트 길이 분포 (정보 밀도 저하)')
    plt.xlabel('글자 수')
    plt.ylabel('리뷰 수')
    plt.axvline(20, color='red', linestyle='--', label='유의미성 임계치 (20자)')
    plt.legend()
    plt.savefig(f'{SAVE_PATH}/03_content_quality.png', dpi=150)
    print("Plot 3 saved.")

    print(f"\n모든 시각화 완료: {SAVE_PATH} 폴더 확인")

if __name__ == "__main__":
    visualize()

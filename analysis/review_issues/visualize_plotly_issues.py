import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client
from dotenv import load_dotenv

# Setup
load_dotenv()
supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
SAVE_PATH = 'analysis/review_issues'
os.makedirs(SAVE_PATH, exist_ok=True)

def fetch_all_reviews():
    print("Fetching reviews from database for Plotly visualization...")
    all_data = []
    start = 0
    page_size = 1000
    while True:
        res = supabase.table('reviews').select('product_code, review_content, review_date, rating').range(start, start + page_size - 1).execute()
        if not res.data: break
        all_data.extend(res.data)
        start += page_size
    return pd.DataFrame(all_data)

def visualize_plotly():
    df = fetch_all_reviews()
    if df.empty:
        print("No data found.")
        return

    # 1. Distribution of Reviews per Book (SKEWNESS)
    counts = df['product_code'].value_counts().reset_index()
    counts.columns = ['product_code', 'review_count']
    fig1 = px.bar(counts, x='product_code', y='review_count', 
                  title='도서별 리뷰 수 분포: 극심한 데이터 롱테일 현상',
                  labels={'product_code': '도서 코드', 'review_count': '리뷰 수'},
                  template='plotly_dark',
                  color='review_count',
                  color_continuous_scale='Viridis')
    fig1.add_hline(y=counts['review_count'].mean(), line_dash="dot", 
                    annotation_text=f"평균: {counts['review_count'].mean():.1f}", 
                    annotation_position="bottom right", line_color="orange")
    fig1.write_html(f'{SAVE_PATH}/01_review_imbalance.html')
    print("Plot 1 (HTML) saved.")

    # 2. Temporal Sparsity (Heatmap style)
    df['date'] = pd.to_datetime(df['review_date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['year_week'] = df['date'].dt.to_period('W').astype(str)
    
    # Top 30 books for visibility
    top_books = counts.head(30)['product_code']
    pivot_df = df[df['product_code'].isin(top_books)].groupby(['product_code', 'year_week']).size().unstack(fill_value=0)
    
    fig2 = px.imshow(pivot_df, 
                     title='시계열 데이터 희소성: 리뷰 발생의 불연속성',
                     labels=dict(x="주차 (Year-Week)", y="도서 코드", color="리뷰 수"),
                     aspect="auto",
                     template='plotly_dark',
                     color_continuous_scale='IceFire')
    fig2.write_html(f'{SAVE_PATH}/02_temporal_sparsity.html')
    print("Plot 2 (HTML) saved.")

    # 3. Content Quality (Review Length)
    df['text_len'] = df['review_content'].str.len().fillna(0)
    fig3 = px.histogram(df, x='text_len', 
                        title='리뷰 텍스트 길이 분포: 정보 밀도 부족 (45%가 20자 미만)',
                        labels={'text_len': '글자 수'},
                        template='plotly_dark',
                        nbins=100,
                        color_discrete_sequence=['#ff7f0e'])
    fig3.add_vline(x=20, line_dash="dash", line_color="red", 
                    annotation_text="유의미 임계치", annotation_position="top right")
    fig3.update_layout(xaxis_range=[0, 500])
    fig3.write_html(f'{SAVE_PATH}/03_content_quality.html')
    print("Plot 3 (HTML) saved.")

    print(f"\n✨ Plotly 인터랙티브 HTML 파일들이 {SAVE_PATH}에 저장되었습니다.")

if __name__ == "__main__":
    visualize_plotly()

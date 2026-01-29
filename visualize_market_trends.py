import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Paths
VIRAL_PATH = 'analysis/viral_index/weekly_news_viral_index_revised.csv'
SALES_PATH = 'analysis/viral_index/weekly_bestseller_scores_decay.csv'
SAVE_PATH = 'analysis/market_analytics'
os.makedirs(SAVE_PATH, exist_ok=True)

def create_market_trend_dashboard():
    print("Loading data for Market Trend visualization...")
    
    # Load Viral Data (using utf-8-sig for potential BOM)
    df_viral = pd.read_csv(VIRAL_PATH, encoding='utf-8-sig')
    df_sales = pd.read_csv(SALES_PATH, encoding='utf-8-sig')
    
    # Preprocessing
    df_viral['ymw'] = df_viral['ymw'].astype(str)
    df_sales['ymw'] = df_sales['ymw'].astype(str)
    
    # Filter for common categories and weeks
    categories = sorted(df_sales['category'].unique())
    
    for category in categories:
        print(f"Processing Category: {category}")
        
        # Filter metrics
        v_data = df_viral[df_viral['category'] == category].sort_values('ymw')
        s_data = df_sales[df_sales['category'] == category].sort_values('ymw')
        
        # Merge on ymw to align weeks
        merged = pd.merge(v_data[['ymw', 'viral_index', 'viral_index_smoothed']], 
                          s_data[['ymw', 'sales_score']], on='ymw', how='inner')
        
        if merged.empty: continue
        
        # Create Subplot (Dual Axis)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Viral Index (Background bars or light line)
        fig.add_trace(
            go.Bar(x=merged['ymw'], y=merged['viral_index'], 
                   name='뉴스 바이럴 지수', opacity=0.3, marker_color='gray'),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=merged['ymw'], y=merged['viral_index_smoothed'], 
                       name='바이럴 추세 (Smoothed)', line=dict(color='cyan', width=2)),
            secondary_y=False,
        )
        
        # Sales Score (Loud line)
        fig.add_trace(
            go.Scatter(x=merged['ymw'], y=merged['sales_score'], 
                       name='베스트셀러 판매지수', line=dict(color='orange', width=4)),
            secondary_y=True,
        )
        
        # Update layout
        clean_cat_name = category.replace('/', '_')
        fig.update_layout(
            title=f'<b>{category}</b> 카테고리 : 뉴스 vs 판매량 추이 분석',
            xaxis_title='주차 (ymw)',
            template='plotly_dark',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.update_yaxes(title_text="뉴스 바이럴 점수", secondary_y=False)
        fig.update_yaxes(title_text="판매량 합산 점수 (Decay)", secondary_y=True)
        
        # Save each category as HTML
        fig.write_html(f'{SAVE_PATH}/trend_{clean_cat_name}.html')
        
    print(f"\n✨ {len(categories)}개 카테고리의 추이 리포트가 {SAVE_PATH}에 생성되었습니다.")

if __name__ == "__main__":
    create_market_trend_dashboard()

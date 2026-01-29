import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 데이터 로드 및 전처리
df = pd.read_csv('books_ml_dataset_v4.csv')
df = df.dropna(subset=['y_sales_score']).reset_index(drop=True)
df['y_lag1'] = df.groupby('product_code')['y_sales_score'].shift(1)
df = df.dropna(subset=['y_lag1']).reset_index(drop=True)

# 분석에 사용된 Top 8 피처 (이전 단계 결과 기반)
selected_features = [
    'category_4', 'category_10', 'prophet_forecast_stock_trading', 'category_3',
    'kospi', 'category_9', 'prophet_forecast_financial_crisis', 'category_6'
]

# VIF 계산용 데이터프레임 (상수항 추가 없이 각 변수 간 독립성 확인)
X = df[selected_features + ['y_lag1']].copy()
# 결측치 제거 (이미 위에서 처리함)

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print("[VIF (다중공선성) 지수 확인]")
print("==============================")
print(vif_data.sort_values("VIF", ascending=False))
print("\n* 가이드라인: VIF < 10 이면 다중공선성 위험이 낮다고 판단함.")

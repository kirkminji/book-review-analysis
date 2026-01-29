import gradio as gr
import joblib
import pandas as pd
import numpy as np

import os

# Get path to current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models and metadata using absolute paths relative to this script
reg_model = joblib.load(os.path.join(BASE_DIR, 'best_reg_model.pkl'))
clf_model = joblib.load(os.path.join(BASE_DIR, 'best_clf_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'robust_scaler.pkl'))
feature_cols = joblib.load(os.path.join(BASE_DIR, 'feature_cols.pkl'))

CATEGORIES = [
    "1. 거시경제 (Macro Economy)",
    "2. 경제학이론 (Econ Theory)",
    "3. 금융위기 (Financial Crisis)",
    "4. 경영 (Business)",
    "5. 부동산 (Real Estate)",
    "6. 개인재무 (Personal Finance)",
    "7. 주식/투자 (Stock Trading)",
    "8. 지정학 (Geopolitics)",
    "9. 테크/스타트업 (Tech Startup)",
    "10. 투자철학 (Invest Philosophy)"
]

def predict(y_lag1, viral_index, category_name, kospi, usd_krw, brent_oil):
    # Prepare input dataframe with default zeros
    input_data = pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)
    
    # 1. Map Category
    cat_idx = int(category_name.split('.')[0])
    input_data[f'category_{cat_idx}'] = 1.0
    
    # 2. Map Viral Interaction
    input_data[f'category_{cat_idx}_x_viral_index'] = viral_index
    
    # 3. Map Basic Features
    input_data['y_lag1'] = y_lag1
    input_data['kospi'] = kospi
    input_data['usd_krw'] = usd_krw
    input_data['brent_oil'] = brent_oil
    
    # 4. Fill Prophet Forecasts (Using a default value of 1.0 if not provided, or can be improved)
    # For simplicity in this demo, we'll set the relevant category's forecast to a moderate value if user doesn't input it
    # Ideally, we should have a way to fetch current prophet forecasts
    # For now, let's just use 1.0 as a baseline multiplier
    forecast_cols = [c for c in feature_cols if c.startswith('prophet_forecast_')]
    for col in forecast_cols:
        input_data[col] = 1.0
        
    # Predict
    # Note: LightGBM reg/clf were trained on unscaled data in my script for simplicity, 
    # but v4 notebook uses RobustScaler for Linear/Ridge. 
    # If using LightGBM, scaling isn't strictly necessary but let's check notebook behavior.
    # Notebook lines 240-243 show LightGBM uses X_train (unscaled).
    
    score = reg_model.predict(input_data)[0]
    prob = clf_model.predict_proba(input_data)[0][1]
    
    res_label = "🔥 베스트셀러 진입 유력" if prob > 0.5 else "❄️ 진입 미달 예상"
    color = "green" if prob > 0.5 else "red"
    
    prob_text = f"진입 확률: {prob:.1%}"
    score_text = f"예측 판매점수: {max(0, score):.2f}점"
    
    return res_label, prob_text, score_text

# Define UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 도서 베스트셀러 예측 AI 라이브 (v4)")
    gr.Markdown("뉴스 바이럴 지수, 전주 판매 실적, 거시 지표를 활용하여 다음 주 성과를 예측합니다.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. 도서 및 트렌드 정보")
            y_lag1 = gr.Number(label="전주 판매 점수 (y_lag1)", value=10, info="20위=1점, 1위=20점 합산")
            viral_index = gr.Slider(0, 10, label="뉴스 바이럴 지수", value=2.5, step=0.1)
            category = gr.Dropdown(CATEGORIES, label="도서 카테고리", value=CATEGORIES[6])
            
            gr.Markdown("### 2. 거시 경제 지표")
            with gr.Row():
                kospi = gr.Number(label="KOSPI", value=2500)
                usd_krw = gr.Number(label="USD/KRW", value=1350)
                brent_oil = gr.Number(label="Brent Oil", value=80)
                
            btn = gr.Button("🚀 성과 예측하기", variant="primary")
            
        with gr.Column():
            gr.Markdown("### 3. 예측 결과")
            output_label = gr.Label(label="판정")
            output_prob = gr.Textbox(label="진입 확률")
            output_score = gr.Textbox(label="예상 판매 점수")
            
            gr.Markdown("---")
            gr.Markdown("**Tip**: 전주 점수가 높을수록(관성), 바이럴 지수가 높을수록 진입 확률이 상승합니다.")

    btn.click(
        predict, 
        inputs=[y_lag1, viral_index, category, kospi, usd_krw, brent_oil], 
        outputs=[output_label, output_prob, output_score]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=True)

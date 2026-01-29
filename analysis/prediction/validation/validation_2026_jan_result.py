#!/usr/bin/env python3
"""2026년 1월 Validation 결과 시각화"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
actual = pd.read_excel('kyobo_weekly_bestseller_2026011.xlsx')
pred = pd.read_csv('prediction_2026_jan_week1.csv')

actual_df = actual[['순위', '판매상품 ID', '상품명']].rename(
    columns={'순위': 'actual_rank', '판매상품 ID': 'product_code', '상품명': 'title'})
pred_df = pred[['product_code', 'pred_rank', 'y_lag1', 'pred_score']]
merged = pd.merge(actual_df, pred_df, on='product_code', how='outer')
merged['actual_rank'] = merged['actual_rank'].fillna(99).astype(int)
merged['pred_rank'] = merged['pred_rank'].fillna(99).astype(int)

# === Figure 1: 메인 결과 (PPT용) ===
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# 1. 정확도 파이차트
ax1 = axes[0]
matched = len(set(actual[actual['순위']<=20]['판매상품 ID']) & set(pred[pred['pred_rank']<=20]['product_code']))
ax1.pie([matched, 20-matched], labels=['예측 성공', '예측 실패'],
        colors=['#2ecc71', '#e74c3c'], autopct='%1.0f%%', startangle=90,
        textprops={'fontsize': 14, 'fontweight': 'bold'})
ax1.set_title('Top 20 예측 정확도', fontsize=16, fontweight='bold', pad=20)

# 2. 순위 비교 산점도
ax2 = axes[1]
both = merged[(merged['actual_rank']<=20) & (merged['pred_rank']<=20)]
new_entries = merged[(merged['actual_rank']<=20) & (merged['pred_rank']==99)]

ax2.scatter(both['pred_rank'], both['actual_rank'], s=150, c='#3498db',
            edgecolors='white', linewidth=2, label=f'기존 책 (n={len(both)})', zorder=3)
ax2.scatter([21]*len(new_entries), new_entries['actual_rank'], s=150, c='#e74c3c',
            marker='s', edgecolors='white', linewidth=2, label=f'신규 진입 (n={len(new_entries)})', zorder=3)
ax2.plot([0, 20], [0, 20], 'k--', alpha=0.3, label='완벽 예측선')
ax2.set_xlabel('예측 순위', fontsize=12)
ax2.set_ylabel('실제 순위', fontsize=12)
ax2.set_xlim(0, 23)
ax2.set_ylim(0, 21)
corr = both['actual_rank'].corr(both['pred_rank'])
ax2.set_title(f'순위 비교 (r = {corr:.3f})', fontsize=16, fontweight='bold', pad=20)
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.invert_yaxis()

# 3. 주요 지표
ax3 = axes[2]
ax3.axis('off')
metrics = [
    ('Top 20 일치율', f'{matched}/20 ({matched/20*100:.0f}%)', '#3498db'),
    ('순위 상관계수', f'{corr:.3f}', '#2ecc71'),
    ('신규 진입', f'{len(new_entries)}권', '#e74c3c'),
    ('기존 책 예측', f'{matched}권', '#9b59b6'),
]
for i, (name, val, color) in enumerate(metrics):
    y = 0.85 - i * 0.22
    ax3.text(0.5, y, val, fontsize=36, fontweight='bold', ha='center', va='center', color=color)
    ax3.text(0.5, y-0.08, name, fontsize=14, ha='center', va='center', color='gray')

ax3.set_title('Validation 지표', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('../ml_image_v4/validation_main_result.png', dpi=200, bbox_inches='tight', facecolor='white')
# ...
plt.savefig('../ml_image_v4/validation_detail_table.png', dpi=200, bbox_inches='tight', facecolor='white')
# ...
plt.savefig('../ml_image_v4/validation_insights.png', dpi=200, bbox_inches='tight', facecolor='white')

print("이미지 저장 완료:")
print("  - ../ml_image_v4/validation_main_result.png")
print("  - ../ml_image_v4/validation_detail_table.png")
print("  - ../ml_image_v4/validation_insights.png")

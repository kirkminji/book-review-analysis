#!/usr/bin/env python3
"""피처 진단: 다중공선성, 이분산성, 정규성, 상관관계 검증 및 시각화"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
import statsmodels.api as sm
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# === 데이터 로드 ===
df = pd.read_csv('books_ml_dataset_v4.csv')
df['y_lag1'] = df.groupby('product_code')['y_sales_score'].shift(1)
df = df.dropna(subset=['y_lag1'])

selected_features = ['category_4', 'category_10', 'prophet_forecast_stock_trading',
                     'category_3', 'kospi', 'category_9',
                     'prophet_forecast_financial_crisis', 'category_6']
feature_cols = selected_features + ['y_lag1']

X = df[feature_cols].copy()
y = df['y_sales_score'].copy()

# 피처 한글명
feature_labels = {
    'category_3': '부동산/실물자산',
    'category_4': '기업경영/리더십',
    'category_6': '투자철학/대가',
    'category_9': '테크/스타트업',
    'category_10': '지정학/국제정세',
    'prophet_forecast_stock_trading': 'Prophet(주식투자)',
    'prophet_forecast_financial_crisis': 'Prophet(금융위기)',
    'kospi': 'KOSPI',
    'y_lag1': '전주 판매점수'
}

print("=" * 60)
print("피처 진단 리포트 (Feature Selection 8개 + y_lag1)")
print("=" * 60)

# =============================================
# 1. 다중공선성 (VIF)
# =============================================
print("\n[1] 다중공선성 (VIF)")
X_const = sm.add_constant(X)
vif_data = []
for i, col in enumerate(X_const.columns):
    if col == 'const':
        continue
    vif = variance_inflation_factor(X_const.values, i)
    label = feature_labels.get(col, col)
    status = 'OK' if vif < 5 else ('주의' if vif < 10 else '심각')
    vif_data.append({'feature': col, 'label': label, 'VIF': round(vif, 2), 'status': status})
    print(f"  {label:20s} VIF={vif:.2f} [{status}]")

vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)

# =============================================
# 2. 피처 간 상관관계
# =============================================
print("\n[2] 피처 간 상관관계")
corr_matrix = X.corr()
high_corr_pairs = []
for i in range(len(feature_cols)):
    for j in range(i+1, len(feature_cols)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.3:
            high_corr_pairs.append((feature_cols[i], feature_cols[j], round(r, 3)))

if high_corr_pairs:
    print("  |r| > 0.3 피처 쌍:")
    for f1, f2, r in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
        print(f"    {feature_labels.get(f1, f1)} × {feature_labels.get(f2, f2)}: r={r}")
else:
    print("  |r| > 0.3인 피처 쌍 없음 → 다중공선성 우려 낮음")

# =============================================
# 3. 이분산성 검정
# =============================================
print("\n[3] 이분산성 검정")
model = sm.OLS(y, X_const).fit()
resid = model.resid
fitted = model.fittedvalues

# Breusch-Pagan
bp_stat, bp_pval, _, _ = het_breuschpagan(resid, X_const)
bp_result = '통과 (등분산)' if bp_pval > 0.05 else '이분산성 존재'
print(f"  Breusch-Pagan: LM={bp_stat:.2f}, p={bp_pval:.4f} → {bp_result}")

# White Test
try:
    white_stat, white_pval, _, _ = het_white(resid, X_const)
    white_result = '통과 (등분산)' if white_pval > 0.05 else '이분산성 존재'
    print(f"  White Test:    LM={white_stat:.2f}, p={white_pval:.4f} → {white_result}")
except:
    white_pval = None
    print("  White Test:    계산 불가 (피처 수 제한)")

# =============================================
# 4. 잔차 정규성 검정
# =============================================
print("\n[4] 잔차 정규성 검정")
# Shapiro-Wilk (n < 5000)
if len(resid) < 5000:
    sw_stat, sw_pval = stats.shapiro(resid)
    sw_result = '정규분포' if sw_pval > 0.05 else '비정규분포'
    print(f"  Shapiro-Wilk: W={sw_stat:.4f}, p={sw_pval:.4f} → {sw_result}")

# Jarque-Bera
jb_stat, jb_pval = stats.jarque_bera(resid)
jb_result = '정규분포' if jb_pval > 0.05 else '비정규분포'
print(f"  Jarque-Bera:  JB={jb_stat:.2f}, p={jb_pval:.4f} → {jb_result}")

skew = stats.skew(resid)
kurt = stats.kurtosis(resid)
print(f"  왜도(Skewness): {skew:.3f}  첨도(Kurtosis): {kurt:.3f}")

# =============================================
# 5. 피처 분포 통계
# =============================================
print("\n[5] 피처 분포 통계")
print(f"  {'피처':20s} {'평균':>8s} {'표준편차':>8s} {'왜도':>8s} {'영값비율':>8s}")
print("  " + "-" * 56)
for col in feature_cols:
    mean = X[col].mean()
    std = X[col].std()
    skewness = stats.skew(X[col])
    zero_pct = (X[col] == 0).sum() / len(X) * 100
    label = feature_labels.get(col, col)
    print(f"  {label:20s} {mean:8.2f} {std:8.2f} {skewness:8.2f} {zero_pct:7.1f}%")


# =============================================
# 시각화
# =============================================
print("\n시각화 생성 중...")

# === Figure 1: VIF + 상관행렬 ===
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# VIF 바 차트
ax1 = axes[0]
labels = [feature_labels.get(r['feature'], r['feature']) for _, r in vif_df.iterrows()]
colors = ['#e74c3c' if v >= 10 else '#f39c12' if v >= 5 else '#2ecc71' for v in vif_df['VIF']]
bars = ax1.barh(range(len(vif_df)), vif_df['VIF'], color=colors, edgecolor='white', height=0.6)
ax1.set_yticks(range(len(vif_df)))
ax1.set_yticklabels(labels, fontsize=11)
ax1.axvline(5, color='#f39c12', linestyle='--', alpha=0.7, label='주의 기준 (VIF=5)')
ax1.axvline(10, color='#e74c3c', linestyle='--', alpha=0.7, label='심각 기준 (VIF=10)')
ax1.set_xlabel('VIF', fontsize=12)
ax1.set_title('다중공선성 (VIF)', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=9)
for i, v in enumerate(vif_df['VIF']):
    ax1.text(v + 0.1, i, f'{v:.1f}', va='center', fontsize=10, fontweight='bold')
ax1.set_xlim(0, max(vif_df['VIF']) * 1.3)
ax1.invert_yaxis()

# 상관행렬 히트맵
ax2 = axes[1]
corr_labels = [feature_labels.get(c, c) for c in feature_cols]
im = ax2.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax2.set_xticks(range(len(feature_cols)))
ax2.set_yticks(range(len(feature_cols)))
ax2.set_xticklabels(corr_labels, rotation=45, ha='right', fontsize=9)
ax2.set_yticklabels(corr_labels, fontsize=9)
for i in range(len(feature_cols)):
    for j in range(len(feature_cols)):
        val = corr_matrix.values[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax2.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)
ax2.set_title('피처 간 상관행렬', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax2, shrink=0.8)

plt.tight_layout()
plt.savefig('ml_image_v4/diagnostic_vif_correlation.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  [1/3] diagnostic_vif_correlation.png")

# === Figure 2: 잔차 진단 4분할 ===
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Residuals vs Fitted
ax = axes[0, 0]
ax.scatter(fitted, resid, s=20, alpha=0.4, c='#3498db', edgecolors='none')
ax.axhline(0, color='red', linestyle='--', alpha=0.5)
lowess = sm.nonparametric.lowess(resid, fitted, frac=0.3)
ax.plot(lowess[:, 0], lowess[:, 1], 'r-', linewidth=2, label='LOWESS')
ax.set_xlabel('적합값 (Fitted Values)')
ax.set_ylabel('잔차 (Residuals)')
ax.set_title('잔차 vs 적합값 (이분산성 진단)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Q-Q Plot
ax = axes[0, 1]
(osm, osr), (slope, intercept, r) = stats.probplot(resid, dist="norm")
ax.scatter(osm, osr, s=20, alpha=0.4, c='#3498db', edgecolors='none')
ax.plot(osm, slope * np.array(osm) + intercept, 'r-', linewidth=2, label=f'이론선 (r={r:.3f})')
ax.set_xlabel('이론적 분위수')
ax.set_ylabel('잔차 분위수')
ax.set_title('Q-Q Plot (정규성 진단)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Residuals vs y_lag1
ax = axes[1, 0]
ax.scatter(X['y_lag1'], resid, s=20, alpha=0.4, c='#e67e22', edgecolors='none')
ax.axhline(0, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('y_lag1 (전주 판매점수)')
ax.set_ylabel('잔차 (Residuals)')
ax.set_title('잔차 vs y_lag1 (핵심 피처 진단)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Residual Distribution
ax = axes[1, 1]
ax.hist(resid, bins=50, color='#3498db', edgecolor='white', alpha=0.8, density=True, label='잔차 분포')
x_range = np.linspace(resid.min(), resid.max(), 100)
ax.plot(x_range, stats.norm.pdf(x_range, resid.mean(), resid.std()),
        'r-', linewidth=2, label=f'정규분포\n(skew={skew:.2f}, kurt={kurt:.2f})')
ax.set_xlabel('잔차')
ax.set_ylabel('밀도')
ax.set_title('잔차 분포 (정규성 진단)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ml_image_v4/diagnostic_residuals.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  [2/3] diagnostic_residuals.png")

# === Figure 3: 피처 분포 (스파스성 + 왜도) ===
fig, axes = plt.subplots(3, 3, figsize=(16, 14))
axes = axes.flatten()

for i, col in enumerate(feature_cols):
    ax = axes[i]
    data = X[col]
    label = feature_labels.get(col, col)
    zero_pct = (data == 0).sum() / len(data) * 100
    skewness = stats.skew(data)

    ax.hist(data, bins=40, color='#3498db', edgecolor='white', alpha=0.8)
    ax.set_title(f'{label}\n왜도={skewness:.1f} | 영값={zero_pct:.0f}%',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('빈도')
    ax.grid(True, alpha=0.3)

    # 왜도/영값 경고 표시
    if abs(skewness) > 2 or zero_pct > 50:
        ax.patch.set_facecolor('#fff3e0')
        ax.patch.set_alpha(0.3)

plt.suptitle('피처 분포 진단 (노란 배경 = 왜도>2 또는 영값>50%)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('ml_image_v4/diagnostic_feature_distributions.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  [3/3] diagnostic_feature_distributions.png")


# =============================================
# 진단 요약 출력
# =============================================
print("\n" + "=" * 60)
print("진단 요약")
print("=" * 60)

issues = []
# VIF 체크
high_vif = vif_df[vif_df['VIF'] >= 5]
if len(high_vif) > 0:
    for _, r in high_vif.iterrows():
        issues.append(f"다중공선성: {feature_labels.get(r['feature'], r['feature'])} VIF={r['VIF']}")

# 이분산성
if bp_pval <= 0.05:
    issues.append(f"이분산성: Breusch-Pagan p={bp_pval:.4f}")

# 정규성
if jb_pval <= 0.05:
    issues.append(f"잔차 비정규성: Jarque-Bera p={jb_pval:.4f} (skew={skew:.2f}, kurt={kurt:.2f})")

# 높은 상관
for f1, f2, r in high_corr_pairs:
    if abs(r) > 0.5:
        issues.append(f"높은 상관: {feature_labels.get(f1, f1)} × {feature_labels.get(f2, f2)} r={r}")

if issues:
    print("\n주의 항목:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("\n모든 진단 통과")

print("\n대응 방안:")
print("  - 트리 기반 모델(LightGBM/RF)은 다중공선성·비정규성에 강건")
print("  - RobustScaler로 이상치/왜도 영향 완화")
print("  - Feature Selection으로 43→8개 축소하여 공선성 이미 제거")

print("\n완료.")

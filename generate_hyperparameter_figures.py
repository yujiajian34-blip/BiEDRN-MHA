#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超参数调优图表生成脚本 (Figures 8-11)
根据EDRN-UD.csv数据生成参数搜索分析图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 配置参数
# ============================================================================
DPI = 300
FIGSIZE_SINGLE = (10, 6)
FIGSIZE_HEATMAP = (8, 6)
FIGSIZE_WIDE = (12, 5)
FIGSIZE_GRID = (12, 6)

OUTPUT_DIR = Path('code/超参数调优图表')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 颜色配置
COLORS_U = {
    32: '#2E86AB',    # 蓝色
    64: '#A23B72',    # 紫红色
    128: '#F18F01',   # 橙色
    256: '#C73E1D'    # 红褐色
}

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# ============================================================================
# 数据加载函数
# ============================================================================
def load_hyperparameter_data():
    """加载超参数调优数据"""
    df = pd.read_csv('code/U-D/EDRN-UD.csv')
    
    # 重命名列
    rename_map = {
        'ModelType': 'model',
        'U': 'U',
        'D': 'D',
        'epoch': 'epoch',
        'train_loss': 'train_loss',
        'val_loss': 'val_loss',
        'precision': 'precision',
        'recall': 'recall',
        'accuracy': 'accuracy',
        'f1': 'f1',
        'fpr': 'fpr',
        'fnr': 'fnr'
    }
    df = df.rename(columns=rename_map)
    
    print("=" * 70)
    print("加载超参数调优数据...")
    print("=" * 70)
    print(f"✓ 加载EDRN-UD数据: {len(df)} 行")
    print(f"✓ 参数U: {sorted(df['U'].unique())}")
    print(f"✓ 参数D: {sorted(df['D'].unique())}")
    print(f"✓ Epoch范围: {df['epoch'].min()}-{df['epoch'].max()}")
    print(f"✓ 指标列: {[col for col in df.columns if col not in ['model', 'U', 'D', 'epoch', 'train_loss', 'val_loss']]}")
    
    return df

# ============================================================================
# 图8: Grid Search F1 Score Heatmap
# ============================================================================
def figure_8_grid_search_heatmap():
    """
    图8: Hyperparameter Grid Search - F1 Score Heatmap
    Y轴: U值, X轴: D值
    """
    df = load_hyperparameter_data()
    
    # 对每个参数组合选择最大F1对应的数据
    df_best = df.loc[df.groupby(['U', 'D'])['f1'].idxmax()]
    df_final = df_best.copy()
    
    # 创建pivot table
    pivot_f1 = df_final.pivot_table(
        values='f1', 
        index='U', 
        columns='D', 
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP, dpi=DPI)
    
    # 绘制热力图
    sns.heatmap(pivot_f1, 
                annot=True,  # 显示数值
                fmt='.3f',
                cmap='RdYlGn',
                cbar_kws={'label': 'F1 Score'},
                linewidths=1,
                linecolor='white',
                ax=ax,
                vmin=0.90, vmax=0.98,
                cbar=True)
    
    ax.set_xlabel('D (Number of Substates d_max)', fontsize=13, fontweight='bold')
    ax.set_ylabel('U (Number of Hidden Units)', fontsize=13, fontweight='bold')
    ax.set_title('Figure 7: Hyperparameter Grid Search - F1 Score Heatmap', 
                fontsize=14, fontweight='bold', pad=15)
    
    # 美化边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(labelsize=11, width=1.5, length=5)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "08_hyperparameter_f1_heatmap.pdf"
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', format='pdf')
    print(f"✓ 生成图8: {output_path}")
    plt.savefig(output_path.with_suffix('.png'), dpi=DPI, bbox_inches='tight')
    plt.close()

# ============================================================================
# 图9: F1 Score Trend with D (across U values)
# ============================================================================
def figure_9_f1_trend():
    """
    图9: Hyperparameter Tuning - F1 Score Trend with D
    对每个U值，展示F1随D变化的趋势
    """
    df = load_hyperparameter_data()
    
    # 对每个参数组合选择最大F1对应的数据
    df_best = df.loc[df.groupby(['U', 'D'])['f1'].idxmax()]
    df_final = df_best.copy()
    
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, dpi=DPI)
    
    # 按U值分组绘制
    for u_val in sorted(df_final['U'].unique()):
        data_u = df_final[df_final['U'] == u_val].sort_values('D')
        
        ax.plot(data_u['D'], data_u['f1'],
               color=COLORS_U.get(u_val, '#1F77B4'),
               linewidth=2.5,
               marker='o',
               markersize=8,
               markeredgewidth=2,
               markeredgecolor='white',
               label=f'U={u_val}',
               alpha=0.85)
    
    ax.set_xlabel('D (Number of Substates d_max)', fontsize=13, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
    ax.set_title('Figure 8: Hyperparameter Tuning - F1 Score Trend with D', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_ylim([0.96, 0.976])
    
    # 图例
    ax.legend(loc='best', fontsize=11, frameon=True, framealpha=0.95, edgecolor='black')
    
    # 美化边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(labelsize=11, width=1.5, length=5)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "09_hyperparameter_f1_trend.pdf"
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', format='pdf')
    print(f"✓ 生成图9: {output_path}")
    plt.savefig(output_path.with_suffix('.png'), dpi=DPI, bbox_inches='tight')
    plt.close()

# ============================================================================
# 图10: Grid Search - Four Metrics Heatmaps
# ============================================================================
def figure_10_metrics_heatmaps():
    """
    图10: Hyperparameter Grid Search - Four Metrics Heatmaps
    2x2布局展示: F1 Score, Precision, Recall, Accuracy
    """
    df = load_hyperparameter_data()
    
    # 对每个参数组合选择最大F1对应的数据
    df_best = df.loc[df.groupby(['U', 'D'])['f1'].idxmax()]
    df_final = df_best.copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=DPI)
    fig.suptitle('Figure 9: Hyperparameter Grid Search - Four Metrics Heatmaps',
                fontsize=15, fontweight='bold', y=0.995)
    
    metrics = ['f1', 'precision', 'recall', 'accuracy']
    metric_names = ['F1 Score', 'Precision', 'Recall', 'Accuracy']
    cmaps = ['RdYlGn', 'Blues', 'Oranges', 'Greens']
    
    for (ax, metric, metric_name, cmap) in zip(axes.flat, metrics, metric_names, cmaps):
        # 创建pivot table
        pivot = df_final.pivot_table(
            values=metric, 
            index='U', 
            columns='D', 
            aggfunc='mean'
        )
        
        sns.heatmap(pivot,
                   annot=True,
                   fmt='.3f',
                   cmap=cmap,
                   cbar_kws={'label': metric_name},
                   linewidths=0.5,
                   linecolor='white',
                   ax=ax,
                   vmin=0.81, vmax=0.99)
        
        ax.set_xlabel('D', fontsize=11, fontweight='bold')
        ax.set_ylabel('U', fontsize=11, fontweight='bold')
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "10_hyperparameter_metrics_heatmaps.pdf"
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', format='pdf')
    print(f"✓ 生成图10: {output_path}")
    plt.savefig(output_path.with_suffix('.png'), dpi=DPI, bbox_inches='tight')
    plt.close()

# ============================================================================
# 图11: Performance Comparison - Bar Charts for Each U-D Combination
# ============================================================================
def figure_11_performance_comparison():
    """
    图11: Grid Search - Performance Comparison for Each U Value
    2x2子图，每个子图展示一个U值下D的影响
    """
    df = load_hyperparameter_data()
    
    # 对每个参数组合选择最大F1对应的数据
    df_best = df.loc[df.groupby(['U', 'D'])['f1'].idxmax()]
    df_final = df_best.copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=DPI)
    fig.suptitle('Figure 10: Grid Search - Performance Comparison for Each U Value',
                fontsize=15, fontweight='bold', y=0.995)
    
    u_values = sorted(df_final['U'].unique())
    d_values = sorted(df_final['D'].unique())
    
    # 颜色配置
    metric_colors = {
        'F1': '#2E86AB',
        'Precision': '#A23B72',
        'Recall': '#F18F01',
        'Accuracy': '#C73E1D'
    }
    
    for ax_idx, (ax, u_val) in enumerate(zip(axes.flat, u_values)):
        data_u = df_final[df_final['U'] == u_val].sort_values('D')
        
        x = np.arange(len(data_u))
        width = 0.2
        
        # 绘制多个指标的柱子
        ax.bar(x - 1.5*width, data_u['f1'], width, label='F1', color=metric_colors['F1'], alpha=0.85)
        ax.bar(x - 0.5*width, data_u['precision'], width, label='Precision', color=metric_colors['Precision'], alpha=0.85)
        ax.bar(x + 0.5*width, data_u['recall'], width, label='Recall', color=metric_colors['Recall'], alpha=0.85)
        ax.bar(x + 1.5*width, data_u['accuracy'], width, label='Accuracy', color=metric_colors['Accuracy'], alpha=0.85)
        
        ax.set_xlabel('D (Substates)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_title(f'U={u_val} - Metrics vs D', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{d}' for d in data_u['D'].values])
        ax.set_ylim([0.81, 1.00])
        ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
        ax.tick_params(labelsize=10)
        
        if ax_idx == 0:
            ax.legend(loc='lower right', fontsize=10, frameon=True, framealpha=0.95)
        
        # 美化边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "11_hyperparameter_performance_comparison.pdf"
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', format='pdf')
    print(f"✓ 生成图11: {output_path}")
    plt.savefig(output_path.with_suffix('.png'), dpi=DPI, bbox_inches='tight')
    plt.close()

# ============================================================================
# 主函数
# ============================================================================
def main():
    print("\n")
    print("=" * 70)
    print("根据真实数据生成超参数调优图表（序号8-11）")
    print("=" * 70)
    print()
    print("生成图表...")
    print("-" * 70)
    
    figure_8_grid_search_heatmap()
    figure_9_f1_trend()
    figure_10_metrics_heatmaps()
    figure_11_performance_comparison()
    
    print("-" * 70)
    print()
    print("✓ 所有超参数调优图表已生成！")
    print(f"✓ 输出目录: {OUTPUT_DIR}")
    print("✓ 分辨率: 300 DPI (出版级)")
    print()
    print("生成的图表:")
    print("  • 图8: Hyperparameter Grid Search - F1 Score Heatmap")
    print("  • 图9: Hyperparameter Tuning - F1 Score Trend with D")
    print("  • 图10: Hyperparameter Grid Search - Four Metrics Heatmaps")
    print("  • 图11: Grid Search - Performance Comparison for Each U Value")
    print()
    print("✓ 所有数据来自EDRN-UD.csv真实参数搜索数据")
    print("=" * 70)
    print()

if __name__ == '__main__':
    main()

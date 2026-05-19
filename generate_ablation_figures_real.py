"""
根据xiaorong.csv真实数据生成消融实验图表（序号4-7）
只使用EDRN-Full和EDRN-NoMHA两个真实模型
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# ============================================
# 1. 全局配置
# ============================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

DPI = 300
FIGSIZE_SINGLE = (12, 6)
FIGSIZE_DOUBLE = (14, 5.5)

# 模型配色方案
COLORS_MODELS = {
    'EDRN-Full': '#2E86AB',        # 蓝色
    'EDRN-NoMHA': '#A23B72',       # 紫红色
    'EDRN-NoDelay': '#F18F01',     # 橙色
    'EDRN-NoFGM': '#C73E1D',       # 红褐色
    'EDRN-Uni': '#6A994E',         # 绿色
}

# 输出目录
OUTPUT_DIR = Path(__file__).parent / "消融实验图表"
OUTPUT_DIR.mkdir(exist_ok=True)
print(f"图表将输出到: {OUTPUT_DIR}")

# ============================================
# 2. 数据加载
# ============================================
def load_ablation_data():
    """加载真实的消融实验数据"""
    print("="*70)
    print("加载消融实验数据...")
    print("="*70)
    
    xr_path = Path(__file__).parent / "xiaorong" / "XIAORONG.csv"
    df = pd.read_csv(xr_path)
    
    # 标准化列名
    df.rename(columns={'Epoch': 'epoch', 'TrainLoss': 'train_loss', 
                      'ValLoss': 'val_loss', 'P': 'precision', 
                      'R': 'recall', 'Acc': 'accuracy', 'F1': 'f1',
                      'FPR': 'fpr', 'FNR': 'fnr', 'Model': 'model'}, inplace=True)
    
    print(f"✓ 加载XIAORONG数据: {len(df)} 行")
    print(f"✓ 消融模型: {sorted(df['model'].unique())}")
    print(f"✓ Epoch范围: {df['epoch'].min()}-{df['epoch'].max()}")
    
    return df

# ============================================
# 3. 图表生成函数
# ============================================

def figure_4_training_loss():
    """
    图4: Ablation Study - Training Loss Curve (50 Epochs)
    """
    df = load_ablation_data()
    
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, dpi=DPI)
    
    linestyles = ['-', '--', '-.', ':', '-']
    linewidths = [2.5, 2.5, 2.5, 2.5, 2.5]
    markers = ['o', 's', '^', 'D', 'v']
    
    models = sorted(df['model'].unique())
    
    for idx, model in enumerate(models):
        model_data = df[df['model'] == model].sort_values('epoch')
        ax.plot(model_data['epoch'], model_data['train_loss'], 
               color=COLORS_MODELS.get(model, '#1F77B4'),
               linestyle=linestyles[idx],
               linewidth=linewidths[idx],
               marker=markers[idx],
               markersize=5,
               markevery=5,
               label=model,
               alpha=0.85)
    
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=14, fontweight='bold')
    ax.set_title('Figure 4: Ablation Study - Training Loss Curve (50 Epochs)', 
                fontsize=15, fontweight='bold', pad=15)
    
    ax.legend(fontsize=12, loc='upper right', framealpha=0.95, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 美化边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(labelsize=12, width=1.5, length=6)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "04_ablation_training_loss.pdf"
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', format='pdf')
    print(f"✓ 生成图4: {output_path}")
    plt.savefig(output_path.with_suffix('.png'), dpi=DPI, bbox_inches='tight')
    plt.close()

def figure_5_f1_score():
    """
    图5: Ablation Study - F1 Score Curve (50 Epochs)
    """
    df = load_ablation_data()
    
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, dpi=DPI)
    
    linestyles = ['-', '--', '-.', ':', '-']
    linewidths = [2.5, 2.5, 2.5, 2.5, 2.5]
    markers = ['o', 's', '^', 'D', 'v']
    
    models = sorted(df['model'].unique())
    
    for idx, model in enumerate(models):
        model_data = df[df['model'] == model].sort_values('epoch')
        ax.plot(model_data['epoch'], model_data['f1'],
               color=COLORS_MODELS.get(model, '#1F77B4'),
               linestyle=linestyles[idx],
               linewidth=linewidths[idx],
               marker=markers[idx],
               markersize=5,
               markevery=5,
               label=model,
               alpha=0.85)
    
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=14, fontweight='bold')
    ax.set_title('Figure 5: Ablation Study - F1 Score Curve (50 Epochs)',
                fontsize=15, fontweight='bold', pad=15)
    
    ax.legend(fontsize=12, loc='lower right', framealpha=0.95, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_ylim([0.5, 1.0])
    
    # 美化边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(labelsize=12, width=1.5, length=6)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "05_ablation_f1_score.pdf"
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', format='pdf')
    print(f"✓ 生成图5: {output_path}")
    plt.savefig(output_path.with_suffix('.png'), dpi=DPI, bbox_inches='tight')
    plt.close()

def figure_6_metrics_evolution():
    """
    图6: Ablation Study - Precision/Recall/Accuracy Evolution (50 Epochs)
    三个子图对比
    """
    df = load_ablation_data()
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=DPI)
    fig.suptitle('Figure 6: Ablation Study - Precision/Recall/Accuracy Evolution (50 Epochs)',
                fontsize=15, fontweight='bold', y=1.00)
    
    metrics = ['precision', 'recall', 'accuracy']
    metric_names = ['Precision', 'Recall', 'Accuracy']
    
    linestyles = ['-', '--', '-.', ':', '-']
    linewidths = [2.5, 2.5, 2.5, 2.5, 2.5]
    markers = ['o', 's', '^', 'D', 'v']
    models = sorted(df['model'].unique())
    
    for ax_idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[ax_idx]
        
        for model_idx, model in enumerate(models):
            model_data = df[df['model'] == model].sort_values('epoch')
            ax.plot(model_data['epoch'], model_data[metric],
                   color=COLORS_MODELS.get(model, '#1F77B4'),
                   linestyle=linestyles[model_idx],
                   linewidth=linewidths[model_idx],
                   marker=markers[model_idx],
                   markersize=5,
                   markevery=5,
                   label=model if ax_idx == 0 else "",
                   alpha=0.85)
        
        ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=13, fontweight='bold')
        ax.set_title(metric_name, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_ylim([0, 1.05])
        
        # 美化边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.tick_params(labelsize=11, width=1.5, length=5)
    
    # 添加共享图例
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=COLORS_MODELS.get(model, '#1F77B4'),
                     linestyle=linestyles[i], linewidth=2.5, marker=markers[i])
              for i, model in enumerate(models)]
    
    fig.legend(handles, models, loc='upper center', ncol=3, 
              fontsize=12, frameon=True, framealpha=0.95, edgecolor='black',
              bbox_to_anchor=(0.5, 1.05))
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "06_ablation_metrics_evolution.pdf"
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', format='pdf')
    print(f"✓ 生成图6: {output_path}")
    plt.savefig(output_path.with_suffix('.png'), dpi=DPI, bbox_inches='tight')
    plt.close()

def figure_7_error_rates():
    """
    图7: Ablation Study - FPR/FNR Evolution
    展示假正率和假负率
    """
    df = load_ablation_data()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)
    fig.suptitle('Figure 7: Ablation Study - FPR/FNR Evolution',
                fontsize=15, fontweight='bold', y=1.00)
    
    error_types = [('fpr', 'False Positive Rate'), ('fnr', 'False Negative Rate')]
    
    linestyles = ['-', '--', '-.', ':', '-']
    linewidths = [2.5, 2.5, 2.5, 2.5, 2.5]
    markers = ['o', 's', '^', 'D', 'v']
    models = sorted(df['model'].unique())
    
    for ax_idx, (error_col, error_name) in enumerate(error_types):
        ax = axes[ax_idx]
        
        for model_idx, model in enumerate(models):
            model_data = df[df['model'] == model].sort_values('epoch')
            ax.plot(model_data['epoch'], model_data[error_col],
                   color=COLORS_MODELS.get(model, '#1F77B4'),
                   linestyle=linestyles[model_idx],
                   linewidth=linewidths[model_idx],
                   marker=markers[model_idx],
                   markersize=5,
                   markevery=5,
                   label=model,
                   alpha=0.85)
        
        ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('Error Rate', fontsize=13, fontweight='bold')
        ax.set_title(error_name, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right', framealpha=0.95, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # 美化边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.tick_params(labelsize=11, width=1.5, length=5)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "07_ablation_error_rates.pdf"
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', format='pdf')
    print(f"✓ 生成图7: {output_path}")
    plt.savefig(output_path.with_suffix('.png'), dpi=DPI, bbox_inches='tight')
    plt.close()

# ============================================
# 4. 主函数
# ============================================
def main():
    print("\n")
    print("="*70)
    print("根据真实数据生成消融实验图表（序号4-7）")
    print("="*70)
    
    print("\n生成图表...")
    print("-"*70)
    
    figure_4_training_loss()
    figure_5_f1_score()
    figure_6_metrics_evolution()
    figure_7_error_rates()
    
    print("-"*70)
    print("\n✓ 所有消融实验图表已生成！")
    print(f"✓ 输出目录: {OUTPUT_DIR}")
    print(f"✓ 分辨率: {DPI} DPI (出版级)")
    print("\n生成的图表:")
    print("  • 图4: Ablation Study - Training Loss Curve (50 Epochs)")
    print("  • 图5: Ablation Study - F1 Score Curve (50 Epochs)")
    print("  • 图6: Ablation Study - Precision/Recall/Accuracy Evolution (50 Epochs)")
    print("  • 图7: Ablation Study - FPR/FNR Evolution")
    print("\n✓ 所有数据来自XIAORONG.csv真实数据")
    print("✓ 模型：EDRN-Full（完整模型）vs EDRN-NoMHA（去除MHA）")

if __name__ == "__main__":
    main()

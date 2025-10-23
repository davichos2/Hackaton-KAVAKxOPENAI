"""
Standalone plotting tool for optimization results
Run this after optimization completes to generate visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os


def plot_optimization_results():
    """Generate comprehensive plots from optimization results"""

    if not os.path.exists("optimization_progress.csv"):
        print("❌ Error: optimization_progress.csv not found!")
        print("   Run the evaluator first to generate results.")
        return

    # Load data
    df = pd.read_csv("optimization_progress.csv")

    if len(df) < 2:
        print("⚠️  Need at least 2 iterations to create meaningful plots")
        return

    print(f"📊 Creating plots from {len(df)} iterations...")

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Main title
    fig.suptitle('LLM Parameter Optimization Results',
                 fontsize=18, fontweight='bold', y=0.98)

    # Plot 1: Quality Score with confidence band (large, top left)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(df['iteration'], df['mean_quality'], 'b-o', linewidth=3,
             markersize=10, label='Mean Quality', zorder=3)
    ax1.fill_between(df['iteration'],
                     df['mean_quality'] - df['std_quality'],
                     df['mean_quality'] + df['std_quality'],
                     alpha=0.3, label='±1 Std Dev', zorder=2)
    ax1.fill_between(df['iteration'], df['min_quality'], df['max_quality'],
                     alpha=0.15, label='Min-Max Range', zorder=1)
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Quality Score', fontsize=12, fontweight='bold')
    ax1.set_title('Quality Score Progress', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 105])

    # Add trend line
    z = np.polyfit(df['iteration'], df['mean_quality'], 1)
    p = np.poly1d(z)
    ax1.plot(df['iteration'], p(df['iteration']), "r--",
             alpha=0.8, linewidth=2, label=f'Trend: {z[0]:.2f}x + {z[1]:.2f}')
    ax1.legend(loc='best', fontsize=10)

    # Plot 2: Best score so far (cumulative max)
    ax2 = fig.add_subplot(gs[0, 2])
    best_so_far = df['mean_quality'].cummax()
    ax2.plot(df['iteration'], best_so_far, 'g-o', linewidth=2, markersize=8)
    ax2.fill_between(df['iteration'], 0, best_so_far, alpha=0.3, color='green')
    ax2.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Best Quality', fontsize=11, fontweight='bold')
    ax2.set_title('Best Score So Far', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])

    # Plot 3: All metrics comparison
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(df['iteration'], df['mean_quality'], 'b-o',
             label='Quality', linewidth=2.5, markersize=8)
    ax3.plot(df['iteration'], df['mean_latency_score'], 'g-s',
             label='Latency', linewidth=2.5, markersize=8)
    ax3.plot(df['iteration'], df['mean_cost_score'], 'r-^',
             label='Cost', linewidth=2.5, markersize=8)
    ax3.plot(df['iteration'], df['mean_safety_score'], 'm-d',
             label='Safety', linewidth=2.5, markersize=8)
    ax3.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Score (0-100)', fontsize=12, fontweight='bold')
    ax3.set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=11, ncol=4)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_ylim([0, 105])

    # Plot 4: Variability (std dev)
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(df['iteration'], df['std_quality'], 'purple',
             linewidth=2.5, marker='o', markersize=8)
    ax4.fill_between(df['iteration'], 0, df['std_quality'],
                     alpha=0.3, color='purple')
    ax4.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Std Deviation', fontsize=11, fontweight='bold')
    ax4.set_title('Quality Variability', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Latency trend
    ax5 = fig.add_subplot(gs[2, 1])
    target_latency = 800  # Default target
    ax5.plot(df['iteration'], df['mean_latency_ms'], 'orange',
             linewidth=2.5, marker='o', markersize=8)
    ax5.axhline(y=target_latency, color='r', linestyle='--',
                linewidth=2, label=f'Target: {target_latency}ms')
    ax5.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Latency (ms)', fontsize=11, fontweight='bold')
    ax5.set_title('Response Latency', fontsize=12, fontweight='bold')
    ax5.legend(loc='best', fontsize=9)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Improvement rate
    ax6 = fig.add_subplot(gs[2, 2])
    if len(df) > 1:
        improvement = df['mean_quality'].diff()
        colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in improvement]
        ax6.bar(df['iteration'][1:], improvement[1:], color=colors[1:], alpha=0.7)
        ax6.axhline(y=0, color='black', linewidth=1)
        ax6.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Change in Quality', fontsize=11, fontweight='bold')
        ax6.set_title('Iteration-to-Iteration Change', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')

    # Save figure
    plt.savefig('optimization_progress.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: optimization_progress.png")

    # Also create a simple summary plot
    fig2, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Quality with all metrics as shaded regions
    ax_left.plot(df['iteration'], df['mean_quality'], 'b-o',
                 linewidth=3, markersize=10, label='Quality')
    ax_left.fill_between(df['iteration'], 0, df['mean_quality'],
                         alpha=0.2, color='blue')
    ax_left.set_xlabel('Iteration', fontsize=13, fontweight='bold')
    ax_left.set_ylabel('Mean Quality Score', fontsize=13, fontweight='bold')
    ax_left.set_title('Optimization Progress', fontsize=15, fontweight='bold')
    ax_left.grid(True, alpha=0.3)
    ax_left.set_ylim([0, 105])

    # Right: Parameter evolution heatmap
    try:
        params_list = [json.loads(p) for p in df['params_json']]
        param_names = list(params_list[0].keys())

        # Create matrix of parameter values (normalized)
        param_matrix = []
        for param_name in param_names:
            values = [p.get(param_name, 0) for p in params_list]
            # Normalize to 0-1 range
            if max(values) != min(values):
                values_norm = [(v - min(values)) / (max(values) - min(values))
                              for v in values]
            else:
                values_norm = [0.5] * len(values)
            param_matrix.append(values_norm)

        im = ax_right.imshow(param_matrix, aspect='auto', cmap='coolwarm',
                            interpolation='nearest')
        ax_right.set_yticks(range(len(param_names)))
        ax_right.set_yticklabels(param_names, fontsize=9)
        ax_right.set_xticks(range(len(df)))
        ax_right.set_xticklabels(df['iteration'], fontsize=9)
        ax_right.set_xlabel('Iteration', fontsize=13, fontweight='bold')
        ax_right.set_title('Parameter Evolution (Normalized)',
                          fontsize=15, fontweight='bold')
        plt.colorbar(im, ax=ax_right, label='Normalized Value')
    except:
        ax_right.text(0.5, 0.5, 'Parameter heatmap\nnot available',
                     ha='center', va='center', fontsize=12)
        ax_right.axis('off')

    plt.tight_layout()
    plt.savefig('optimization_summary.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: optimization_summary.png")

    # Print statistics
    print(f"\n{'='*60}")
    print("📈 OPTIMIZATION STATISTICS")
    print(f"{'='*60}")
    print(f"Total Iterations: {len(df)}")
    print(f"Starting Quality: {df['mean_quality'].iloc[0]:.2f}")
    print(f"Final Quality: {df['mean_quality'].iloc[-1]:.2f}")
    print(f"Best Quality: {df['mean_quality'].max():.2f} (Iteration {df['mean_quality'].idxmax() + 1})")
    print(f"Total Improvement: {df['mean_quality'].iloc[-1] - df['mean_quality'].iloc[0]:.2f}")
    print(f"Average Improvement per Iteration: {(df['mean_quality'].iloc[-1] - df['mean_quality'].iloc[0]) / len(df):.2f}")
    print(f"{'='*60}\n")

    # Show plots (optional - comment out if running on server)
    # plt.show()


if __name__ == "__main__":
    print("\n🎨 Optimization Results Plotter\n")
    plot_optimization_results()

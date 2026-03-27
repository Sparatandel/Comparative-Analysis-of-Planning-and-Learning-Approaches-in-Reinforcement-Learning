"""
Analysis & Visualization Module
Generates comparative plots and reports for RL algorithms
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import io


# Dark Knight Color Palette
DK_BG = '#0a0a0f'
DK_SURFACE = '#12121a'
DK_PANEL = '#1a1a28'
DK_ACCENT1 = '#c9a84c'   # Gold
DK_ACCENT2 = '#2a6dd9'   # Electric Blue
DK_ACCENT3 = '#d9322a'   # Crimson
DK_TEXT = '#e8e8f0'
DK_MUTED = '#6a6a8a'

ALGO_COLORS = {
    'Value Iteration': '#c9a84c',
    'Q-Learning': '#2a6dd9',
    'SARSA': '#d9322a'
}

def setup_dark_style():
    plt.rcParams.update({
        'figure.facecolor': DK_BG,
        'axes.facecolor': DK_SURFACE,
        'axes.edgecolor': '#2a2a3a',
        'axes.labelcolor': DK_TEXT,
        'text.color': DK_TEXT,
        'xtick.color': DK_MUTED,
        'ytick.color': DK_MUTED,
        'grid.color': '#1e1e2e',
        'grid.alpha': 0.7,
        'legend.facecolor': DK_PANEL,
        'legend.edgecolor': '#2a2a3a',
        'font.family': 'monospace',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def smooth(data, window=20):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_learning_curves(metrics_dict, title="Learning Curves"):
    setup_dark_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(DK_BG)
    fig.suptitle(title, color=DK_ACCENT1, fontsize=16, fontweight='bold', y=0.98)
    
    axes = axes.flatten()
    
    # 1. Reward per Episode
    ax = axes[0]
    ax.set_title('Cumulative Reward per Episode', color=DK_TEXT, fontsize=11, pad=10)
    for name, m in metrics_dict.items():
        if m['rewards_per_episode']:
            data = smooth(m['rewards_per_episode'], 30)
            ax.plot(data, color=ALGO_COLORS[name], label=name, linewidth=2.0, alpha=0.9)
    ax.set_xlabel('Episode', color=DK_MUTED)
    ax.set_ylabel('Reward', color=DK_MUTED)
    ax.legend(facecolor=DK_PANEL, edgecolor=DK_ACCENT1)
    ax.grid(True, alpha=0.3)
    
    # 2. Steps per Episode
    ax = axes[1]
    ax.set_title('Steps per Episode (lower = better)', color=DK_TEXT, fontsize=11, pad=10)
    for name, m in metrics_dict.items():
        if m['steps_per_episode']:
            data = smooth(m['steps_per_episode'], 30)
            ax.plot(data, color=ALGO_COLORS[name], label=name, linewidth=2.0, alpha=0.9)
    ax.set_xlabel('Episode', color=DK_MUTED)
    ax.set_ylabel('Steps', color=DK_MUTED)
    ax.legend(facecolor=DK_PANEL, edgecolor=DK_ACCENT1)
    ax.grid(True, alpha=0.3)
    
    # 3. Convergence Delta
    ax = axes[2]
    ax.set_title('Convergence Delta (Value Change)', color=DK_TEXT, fontsize=11, pad=10)
    for name, m in metrics_dict.items():
        if m.get('delta_history'):
            data = smooth(m['delta_history'], 20)
            ax.plot(data, color=ALGO_COLORS[name], label=name, linewidth=2.0, alpha=0.9)
    ax.set_xlabel('Iteration', color=DK_MUTED)
    ax.set_ylabel('Delta', color=DK_MUTED)
    ax.set_yscale('symlog', linthresh=0.01)
    ax.legend(facecolor=DK_PANEL, edgecolor=DK_ACCENT1)
    ax.grid(True, alpha=0.3)
    
    # 4. Time & Convergence Bar Chart
    ax = axes[3]
    ax.set_title('Training Time Comparison', color=DK_TEXT, fontsize=11, pad=10)
    names = list(metrics_dict.keys())
    times = [metrics_dict[n]['time_taken'] for n in names]
    colors = [ALGO_COLORS[n] for n in names]
    bars = ax.bar(names, times, color=colors, alpha=0.8, edgecolor=DK_ACCENT1, linewidth=0.5)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{t:.3f}s', ha='center', va='bottom', color=DK_TEXT, fontsize=9)
    ax.set_ylabel('Time (seconds)', color=DK_MUTED)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=DK_BG, edgecolor='none')
    plt.close()
    buf.seek(0)
    return buf


def plot_value_heatmaps(env, agents_dict):
    setup_dark_style()
    n = len(agents_dict)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor(DK_BG)
    fig.suptitle('Value Function Heatmaps', color=DK_ACCENT1, fontsize=15, fontweight='bold')
    
    batman_cmap = LinearSegmentedColormap.from_list('batman', 
        [DK_BG, '#1a1a3a', '#2a2a6a', '#1a4a8a', DK_ACCENT2, DK_ACCENT1])
    
    for ax, (name, agent) in zip(axes, agents_dict.items()):
        grid = agent.get_value_grid()
        
        # Mask obstacles
        mask = np.zeros((env.size, env.size))
        for (r, c) in env.obstacles:
            mask[r][c] = 1
        
        masked_grid = np.ma.masked_where(mask == 1, grid)
        
        im = ax.imshow(masked_grid, cmap=batman_cmap, aspect='equal', interpolation='bilinear')
        
        # Draw obstacles
        for (r, c) in env.obstacles:
            ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, 
                         color='#0a0a0f', zorder=2))
        
        # Mark start/goal
        sr, sc = env.start
        gr, gc = env.goal
        ax.plot(sc, sr, 's', color=DK_ACCENT2, markersize=14, zorder=3, label='Start')
        ax.plot(gc, gr, '*', color=DK_ACCENT1, markersize=16, zorder=3, label='Goal')
        
        ax.set_title(name, color=ALGO_COLORS[name], fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        for spine in ax.spines.values():
            spine.set_color(ALGO_COLORS[name])
            spine.set_linewidth(1.5)
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color=DK_MUTED)
        ax.legend(loc='upper left', fontsize=7, facecolor=DK_PANEL)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=DK_BG, edgecolor='none')
    plt.close()
    buf.seek(0)
    return buf


def plot_policy_grid(env, agent, title="Policy"):
    setup_dark_style()
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    fig.patch.set_facecolor(DK_BG)
    
    from algorithms import GridWorld
    
    size = env.size
    batman_cmap = LinearSegmentedColormap.from_list('batman2', 
        ['#0d0d1a', '#1a1a3a', '#252548'])
    
    # Background grid
    bg = np.zeros((size, size))
    for (r, c) in env.get_all_states():
        bg[r][c] = 1
    ax.imshow(bg, cmap=batman_cmap, aspect='equal', vmin=0, vmax=2)
    
    # Obstacles
    for (r, c) in env.obstacles:
        ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='#050508', zorder=2, linewidth=0))
        ax.add_patch(plt.Rectangle((c-0.45, r-0.45), 0.9, 0.9, color='#111120', zorder=2.5, linewidth=0))
    
    # Grid lines
    for i in range(size + 1):
        ax.axhline(i - 0.5, color='#1e1e2e', linewidth=0.5, zorder=1)
        ax.axvline(i - 0.5, color='#1e1e2e', linewidth=0.5, zorder=1)
    
    # Policy arrows
    arrow_map = {0: (0, -0.3), 1: (0, 0.3), 2: (-0.3, 0), 3: (0.3, 0)}
    
    for state in env.get_all_states():
        r, c = state
        if state == env.goal:
            continue
        action = agent.get_action(state)
        dx, dy = arrow_map[action]
        ax.annotate('', xy=(c + dx, r + dy), xytext=(c, r),
                   arrowprops=dict(arrowstyle='->', color=DK_ACCENT1, lw=1.5),
                   zorder=4)
    
    # Start / Goal markers
    sr, sc = env.start
    gr, gc = env.goal
    ax.add_patch(plt.Circle((sc, sr), 0.35, color=DK_ACCENT2, zorder=5, alpha=0.9))
    ax.text(sc, sr, 'S', ha='center', va='center', color='white', fontsize=10, 
            fontweight='bold', zorder=6)
    ax.add_patch(plt.Circle((gc, gr), 0.35, color=DK_ACCENT1, zorder=5, alpha=0.9))
    ax.text(gc, gr, 'G', ha='center', va='center', color='black', fontsize=10,
            fontweight='bold', zorder=6)
    
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_title(f'Optimal Policy — {title}', color=DK_ACCENT1, fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    
    for spine in ax.spines.values():
        spine.set_color(DK_ACCENT1)
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=DK_BG, edgecolor='none')
    plt.close()
    buf.seek(0)
    return buf


def plot_radar_chart(stats):
    setup_dark_style()
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(DK_BG)
    ax.set_facecolor(DK_SURFACE)
    
    categories = ['Avg Reward', 'Speed', 'Efficiency', 'Best Score', 'Convergence']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    # Normalize stats
    all_rewards = [stats[n]['avg_reward'] for n in stats]
    all_times = [stats[n]['time_taken'] for n in stats]
    all_steps = [stats[n]['avg_steps'] for n in stats]
    all_best = [stats[n]['best_reward'] for n in stats]
    all_conv = [stats[n]['convergence'] for n in stats]
    
    def normalize(vals, invert=False):
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return [0.5] * len(vals)
        norm = [(v - mn) / (mx - mn) for v in vals]
        return [1 - v for v in norm] if invert else norm
    
    norm_reward = normalize(all_rewards)
    norm_speed = normalize(all_times, invert=True)
    norm_eff = normalize(all_steps, invert=True)
    norm_best = normalize(all_best)
    norm_conv = normalize(all_conv, invert=True)
    
    for i, name in enumerate(stats):
        values = [norm_reward[i], norm_speed[i], norm_eff[i], norm_best[i], norm_conv[i]]
        values += values[:1]
        color = ALGO_COLORS[name]
        ax.plot(angles, values, color=color, linewidth=2.5, label=name)
        ax.fill(angles, values, color=color, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color=DK_TEXT, fontsize=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], color=DK_MUTED, fontsize=7)
    ax.grid(color='#1e1e2e', linewidth=0.8)
    ax.spines['polar'].set_color('#2a2a3a')
    
    ax.set_title('Algorithm Performance Radar', color=DK_ACCENT1, 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
              facecolor=DK_PANEL, edgecolor=DK_ACCENT1)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=DK_BG, edgecolor='none')
    plt.close()
    buf.seek(0)
    return buf


def generate_full_report(env, agents_dict, metrics_dict):
    """Generate a comprehensive multi-panel report"""
    from algorithms import get_comparison_stats
    stats = get_comparison_stats(metrics_dict)
    
    plots = {
        'learning_curves': plot_learning_curves(metrics_dict, "Comparative Learning Curves"),
        'value_heatmaps': plot_value_heatmaps(env, agents_dict),
        'radar': plot_radar_chart(stats),
    }
    
    policy_plots = {}
    for name, agent in agents_dict.items():
        policy_plots[name] = plot_policy_grid(env, agent, name)
    
    plots['policies'] = policy_plots
    
    return plots, stats

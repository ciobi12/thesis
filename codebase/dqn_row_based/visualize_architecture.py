"""
Visual diagram generator for Row-Based RL Architecture
Creates a comprehensive flow diagram showing the complete pipeline
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

def create_architecture_diagram():
    """Create a visual diagram of the row-based RL architecture."""
    
    fig = plt.figure(figsize=(16, 20))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 28)
    ax.axis('off')
    
    # Title
    ax.text(5, 27, 'Row-Based RL for Path Segmentation', 
            fontsize=20, weight='bold', ha='center')
    
    # ==================== 1. INPUT IMAGE ====================
    y_pos = 25.5
    input_box = FancyBboxPatch((2, y_pos), 6, 1.2, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='#2E86AB', facecolor='#A9D6E5', 
                               linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, y_pos + 0.6, 'Input Image (H × W)', 
            fontsize=12, weight='bold', ha='center', va='center')
    ax.text(5, y_pos + 0.2, 'Noisy CT-like / Vessel / Root scan', 
            fontsize=9, ha='center', va='center', style='italic')
    
    # Arrow down
    arrow1 = FancyArrowPatch((5, y_pos), (5, y_pos - 0.8),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    # ==================== 2. ROW-BY-ROW PROCESSING ====================
    y_pos = 23.5
    seq_box = FancyBboxPatch((1.5, y_pos), 7, 1, 
                             boxstyle="round,pad=0.1", 
                             edgecolor='#9B59B6', facecolor='#E8DAEF', 
                             linewidth=2)
    ax.add_patch(seq_box)
    ax.text(5, y_pos + 0.5, 'Sequential Row-by-Row Processing', 
            fontsize=11, weight='bold', ha='center', va='center')
    ax.text(5, y_pos + 0.1, '(Bottom → Top or Top → Bottom)', 
            fontsize=9, ha='center', va='center', style='italic')
    
    # Arrow down
    arrow2 = FancyArrowPatch((5, y_pos), (5, y_pos - 0.8),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    # ==================== 3. OBSERVATION CONSTRUCTION ====================
    y_pos = 21
    
    # Main observation box
    obs_main = FancyBboxPatch((1, y_pos - 0.3), 8, 2.2, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='#E67E22', facecolor='#FADBD8', 
                              linewidth=2, linestyle='--')
    ax.add_patch(obs_main)
    ax.text(5, y_pos + 1.7, 'Observation State Construction', 
            fontsize=11, weight='bold', ha='center')
    
    # Three components
    # Current row
    curr_box = FancyBboxPatch((1.5, y_pos + 0.5), 2, 0.9, 
                              boxstyle="round,pad=0.05", 
                              edgecolor='#16A085', facecolor='#A9DFBF', 
                              linewidth=1.5)
    ax.add_patch(curr_box)
    ax.text(2.5, y_pos + 1.1, 'Current Row', fontsize=9, weight='bold', ha='center')
    ax.text(2.5, y_pos + 0.85, 'Pixels (W × C)', fontsize=8, ha='center')
    
    # History
    hist_box = FancyBboxPatch((4, y_pos + 0.5), 2, 0.9, 
                              boxstyle="round,pad=0.05", 
                              edgecolor='#D35400', facecolor='#F8C471', 
                              linewidth=1.5)
    ax.add_patch(hist_box)
    ax.text(5, y_pos + 1.1, 'History', fontsize=9, weight='bold', ha='center')
    ax.text(5, y_pos + 0.85, 'K prev rows', fontsize=8, ha='center')
    ax.text(5, y_pos + 0.65, '(context)', fontsize=7, ha='center', style='italic')
    
    # Future
    fut_box = FancyBboxPatch((6.5, y_pos + 0.5), 2, 0.9, 
                             boxstyle="round,pad=0.05", 
                             edgecolor='#8E44AD', facecolor='#D7BDE2', 
                             linewidth=1.5)
    ax.add_patch(fut_box)
    ax.text(7.5, y_pos + 1.1, 'Future', fontsize=9, weight='bold', ha='center')
    ax.text(7.5, y_pos + 0.85, 'K next rows', fontsize=8, ha='center')
    ax.text(7.5, y_pos + 0.65, '(lookahead)', fontsize=7, ha='center', style='italic')
    
    # Row index indicator
    ax.text(1.8, y_pos + 0.1, 'Row Index: t/H', fontsize=8, ha='left', style='italic')
    
    # Arrow down from observation
    arrow3 = FancyArrowPatch((5, y_pos - 0.3), (5, y_pos - 1.2),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow3)
    
    # ==================== 4. NEURAL NETWORK ====================
    y_pos = 17.5
    
    # Main network box
    net_box = FancyBboxPatch((1.5, y_pos - 0.2), 7, 2.2, 
                             boxstyle="round,pad=0.1", 
                             edgecolor='#2874A6', facecolor='#D6EAF8', 
                             linewidth=2)
    ax.add_patch(net_box)
    ax.text(5, y_pos + 1.8, 'CNN Policy Network', 
            fontsize=11, weight='bold', ha='center')
    
    # Network layers
    layer_y = y_pos + 1.2
    layer_width = 1.6
    
    # Row encoder
    enc1 = Rectangle((1.8, layer_y - 0.3), layer_width, 0.5, 
                     edgecolor='#117864', facecolor='#A2D9CE', linewidth=1.5)
    ax.add_patch(enc1)
    ax.text(2.6, layer_y - 0.05, 'Row Enc', fontsize=8, ha='center', weight='bold')
    ax.text(2.6, layer_y - 0.25, '(64)', fontsize=7, ha='center')
    
    # History encoder
    enc2 = Rectangle((3.7, layer_y - 0.3), layer_width, 0.5, 
                     edgecolor='#9C640C', facecolor='#F9E79F', linewidth=1.5)
    ax.add_patch(enc2)
    ax.text(4.5, layer_y - 0.05, 'Hist Enc', fontsize=8, ha='center', weight='bold')
    ax.text(4.5, layer_y - 0.25, '(32)', fontsize=7, ha='center')
    
    # Future encoder
    enc3 = Rectangle((5.6, layer_y - 0.3), layer_width, 0.5, 
                     edgecolor='#6C3483', facecolor='#D7BDE2', linewidth=1.5)
    ax.add_patch(enc3)
    ax.text(6.4, layer_y - 0.05, 'Fut Enc', fontsize=8, ha='center', weight='bold')
    ax.text(6.4, layer_y - 0.25, '(32)', fontsize=7, ha='center')
    
    # Concatenation
    ax.text(5, layer_y - 0.65, '↓ Concatenate (128 channels)', 
            fontsize=8, ha='center', style='italic')
    
    # Attention
    att_box = Rectangle((3.5, layer_y - 1.2), 3, 0.4, 
                        edgecolor='#7D3C98', facecolor='#EBDEF0', linewidth=1.5)
    ax.add_patch(att_box)
    ax.text(5, layer_y - 1.0, 'Attention Layer', fontsize=8, ha='center', weight='bold')
    
    # Decision layers
    dec_box = Rectangle((3.2, layer_y - 1.75), 3.6, 0.4, 
                        edgecolor='#1A5490', facecolor='#AED6F1', linewidth=1.5)
    ax.add_patch(dec_box)
    ax.text(5, layer_y - 1.55, 'Decision Layers → Q-values', fontsize=8, ha='center', weight='bold')
    
    # Arrow down from network
    arrow4 = FancyArrowPatch((5, y_pos - 0.2), (5, y_pos - 1.1),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow4)
    
    # ==================== 5. Q-VALUES & ACTION ====================
    y_pos = 15.2
    
    # Q-values
    q_box = FancyBboxPatch((1.8, y_pos), 2.8, 0.8, 
                           boxstyle="round,pad=0.05", 
                           edgecolor='#0E6655', facecolor='#A9DFBF', 
                           linewidth=1.5)
    ax.add_patch(q_box)
    ax.text(3.2, y_pos + 0.55, 'Q-Values (W × 2)', fontsize=9, weight='bold', ha='center')
    ax.text(3.2, y_pos + 0.25, '[Background | Path]', fontsize=8, ha='center')
    
    # Epsilon-greedy
    ax.text(5.5, y_pos + 0.4, 'ε-greedy', fontsize=9, ha='center', 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Action
    action_box = FancyBboxPatch((5.8, y_pos), 2.8, 0.8, 
                                boxstyle="round,pad=0.05", 
                                edgecolor='#7D6608', facecolor='#F9E79F', 
                                linewidth=1.5)
    ax.add_patch(action_box)
    ax.text(7.2, y_pos + 0.55, 'Action (W,)', fontsize=9, weight='bold', ha='center')
    ax.text(7.2, y_pos + 0.25, 'Binary [0 or 1]', fontsize=8, ha='center')
    
    # Arrow between Q-values and action
    arrow5 = FancyArrowPatch((4.6, y_pos + 0.4), (5.8, y_pos + 0.4),
                            arrowstyle='->', mutation_scale=15, 
                            linewidth=1.5, color='black')
    ax.add_patch(arrow5)
    
    # Arrow down from action
    arrow6 = FancyArrowPatch((5, y_pos), (5, y_pos - 0.8),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow6)
    
    # ==================== 6. REWARD COMPONENTS ====================
    y_pos = 12.5
    
    # Main reward box
    rew_main = FancyBboxPatch((0.8, y_pos - 0.3), 8.4, 2.5, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='#C0392B', facecolor='#F5B7B1', 
                              linewidth=2, linestyle='--')
    ax.add_patch(rew_main)
    ax.text(5, y_pos + 2.0, 'Reward Calculation', 
            fontsize=11, weight='bold', ha='center')
    
    # Three reward components
    # Base reward
    base_rew = FancyBboxPatch((1.2, y_pos + 0.6), 2.2, 1.1, 
                              boxstyle="round,pad=0.05", 
                              edgecolor='#196F3D', facecolor='#ABEBC6', 
                              linewidth=1.5)
    ax.add_patch(base_rew)
    ax.text(2.3, y_pos + 1.35, 'Base Reward', fontsize=9, weight='bold', ha='center')
    ax.text(2.3, y_pos + 1.05, '(Accuracy)', fontsize=8, ha='center', style='italic')
    ax.text(2.3, y_pos + 0.8, '−|action − gt|', fontsize=8, ha='center', 
            family='monospace')
    
    # Continuity reward
    cont_rew = FancyBboxPatch((3.9, y_pos + 0.6), 2.2, 1.1, 
                              boxstyle="round,pad=0.05", 
                              edgecolor='#7D3C98', facecolor='#D7BDE2', 
                              linewidth=1.5)
    ax.add_patch(cont_rew)
    ax.text(5.0, y_pos + 1.35, 'Continuity', fontsize=9, weight='bold', ha='center')
    ax.text(5.0, y_pos + 1.05, '(Smoothness)', fontsize=8, ha='center', style='italic')
    ax.text(5.0, y_pos + 0.8, 'Vertical linking', fontsize=8, ha='center')
    
    # Gradient reward
    grad_rew = FancyBboxPatch((6.6, y_pos + 0.6), 2.2, 1.1, 
                              boxstyle="round,pad=0.05", 
                              edgecolor='#B9770E', facecolor='#F8C471', 
                              linewidth=1.5)
    ax.add_patch(grad_rew)
    ax.text(7.7, y_pos + 1.35, 'Gradient', fontsize=9, weight='bold', ha='center')
    ax.text(7.7, y_pos + 1.05, '(Intensity)', fontsize=8, ha='center', style='italic')
    ax.text(7.7, y_pos + 0.8, 'Smooth trans.', fontsize=8, ha='center')
    
    # Total reward
    total_rew = FancyBboxPatch((3, y_pos - 0.1), 4, 0.6, 
                               boxstyle="round,pad=0.05", 
                               edgecolor='#943126', facecolor='#F1948A', 
                               linewidth=2)
    ax.add_patch(total_rew)
    ax.text(5, y_pos + 0.2, 'Total Reward = Σ pixel_rewards', 
            fontsize=9, weight='bold', ha='center')
    
    # Arrow down from reward
    arrow7 = FancyArrowPatch((5, y_pos - 0.3), (5, y_pos - 1.0),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow7)
    
    # ==================== 7. BUFFER UPDATE & STORAGE ====================
    y_pos = 10.2
    
    # Update buffers
    buf_box = FancyBboxPatch((1.5, y_pos + 0.5), 3, 0.7, 
                             boxstyle="round,pad=0.05", 
                             edgecolor='#7D6608', facecolor='#F9E79F', 
                             linewidth=1.5)
    ax.add_patch(buf_box)
    ax.text(3, y_pos + 0.9, 'Update Buffers', fontsize=9, weight='bold', ha='center')
    ax.text(3, y_pos + 0.65, 'prev_preds, prev_rows', fontsize=7, ha='center')
    
    # Store transition
    store_box = FancyBboxPatch((5.5, y_pos + 0.5), 3, 0.7, 
                               boxstyle="round,pad=0.05", 
                               edgecolor='#512E5F', facecolor='#D7BDE2', 
                               linewidth=1.5)
    ax.add_patch(store_box)
    ax.text(7, y_pos + 0.9, 'Store Transition', fontsize=9, weight='bold', ha='center')
    ax.text(7, y_pos + 0.65, '(s, a, r, s\', done)', fontsize=7, ha='center')
    
    # Replay buffer
    replay_box = FancyBboxPatch((5.5, y_pos - 0.2), 3, 0.6, 
                                boxstyle="round,pad=0.05", 
                                edgecolor='#1B4F72', facecolor='#AED6F1', 
                                linewidth=1.5)
    ax.add_patch(replay_box)
    ax.text(7, y_pos + 0.15, '→ Replay Buffer', fontsize=8, weight='bold', ha='center')
    
    # Arrow down
    arrow8 = FancyArrowPatch((5, y_pos + 0.5), (5, y_pos - 0.7),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow8)
    
    # ==================== 8. DQN TRAINING LOOP ====================
    y_pos = 7.5
    
    dqn_box = FancyBboxPatch((1, y_pos - 0.2), 8, 2.2, 
                             boxstyle="round,pad=0.1", 
                             edgecolor='#154360', facecolor='#D4E6F1', 
                             linewidth=2)
    ax.add_patch(dqn_box)
    ax.text(5, y_pos + 1.8, 'DQN Training Loop', 
            fontsize=11, weight='bold', ha='center')
    
    # Training steps
    steps = [
        '1. Sample batch from replay buffer',
        '2. Compute Q(s,a) using policy network',
        '3. Compute target = r + γ·max Q(s\',a\') with target net',
        '4. Loss = MSE(Q(s,a), target)',
        '5. Backprop & update policy network',
        '6. Periodically: target_net ← policy_net'
    ]
    
    step_y = y_pos + 1.4
    for i, step in enumerate(steps):
        ax.text(1.3, step_y - i*0.25, step, fontsize=7.5, ha='left', va='center')
    
    # Arrow down
    arrow9 = FancyArrowPatch((5, y_pos - 0.2), (5, y_pos - 1.0),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow9)
    
    # ==================== 9. ITERATION ====================
    y_pos = 5.5
    
    # Next row box
    next_box = FancyBboxPatch((2.5, y_pos), 5, 0.9, 
                              boxstyle="round,pad=0.05", 
                              edgecolor='#7D3C98', facecolor='#E8DAEF', 
                              linewidth=2)
    ax.add_patch(next_box)
    ax.text(5, y_pos + 0.6, 'Move to Next Row (t + 1)', 
            fontsize=10, weight='bold', ha='center')
    ax.text(5, y_pos + 0.2, 'Repeat until all H rows processed', 
            fontsize=8, ha='center', style='italic')
    
    # Feedback arrow (going back up) - using path_effects for loop indication
    arrow_back = FancyArrowPatch((8.5, y_pos + 0.45), (8.5, 22),
                                arrowstyle='->', mutation_scale=20, 
                                linewidth=1.5, color='#7D3C98', 
                                linestyle='dashed',
                                connectionstyle="arc3,rad=.3")
    ax.add_patch(arrow_back)
    ax.text(9.2, 13, 'Loop', fontsize=9, ha='center', rotation=90, 
            color='#7D3C98', weight='bold')
    
    # Arrow down to final result
    arrow10 = FancyArrowPatch((5, y_pos), (5, y_pos - 1.0),
                             arrowstyle='->', mutation_scale=20, 
                             linewidth=2, color='black')
    ax.add_patch(arrow10)
    
    # ==================== 10. FINAL OUTPUT ====================
    y_pos = 3.2
    
    output_box = FancyBboxPatch((1.5, y_pos - 0.5), 7, 2.5, 
                                boxstyle="round,pad=0.1", 
                                edgecolor='#145A32', facecolor='#ABEBC6', 
                                linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, y_pos + 1.8, 'Final Reconstruction', 
            fontsize=11, weight='bold', ha='center')
    ax.text(5, y_pos + 1.45, '(H × W Binary Prediction)', 
            fontsize=9, ha='center', style='italic')
    
    # Simulated output visualization
    # Create a small example output image
    np.random.seed(42)
    example_img = np.zeros((20, 40))
    # Create some path-like structure
    for i in range(20):
        center = 20 + int(3 * np.sin(i * 0.3))
        example_img[i, max(0, center-2):min(40, center+3)] = 1
        # Add branching
        if i > 10:
            branch_center = 10 + int(2 * np.cos(i * 0.4))
            example_img[i, max(0, branch_center-1):min(40, branch_center+2)] = 1
    
    # Display the example
    ax_img = fig.add_axes([0.22, 0.04, 0.56, 0.08])
    ax_img.imshow(example_img, cmap='Greens', aspect='auto')
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title('Example: Reconstructed Path/Vessel', fontsize=8, pad=2)
    
    # Metrics box
    metrics_text = 'Metrics: IoU, F1, Accuracy, Coverage'
    ax.text(5, y_pos - 0.3, metrics_text, fontsize=8, ha='center', 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # ==================== SIDE NOTES ====================
    # Key parameters box
    param_box = FancyBboxPatch((0.1, 16), 1.2, 5, 
                               boxstyle="round,pad=0.05", 
                               edgecolor='gray', facecolor='lightyellow', 
                               linewidth=1, alpha=0.7)
    ax.add_patch(param_box)
    ax.text(0.7, 20.7, 'Key Params', fontsize=8, weight='bold', ha='center')
    
    params = [
        'history: 3-5',
        'future: 3',
        'cont_coef:',
        '  0.1-0.2',
        'ε: 1.0→0.01',
        'γ: 0.99',
        'batch: 32-64',
        'target_upd:',
        '  500 steps'
    ]
    param_y = 20.3
    for param in params:
        ax.text(0.7, param_y, param, fontsize=6.5, ha='center', va='top')
        param_y -= 0.4
    
    # Advantages box
    adv_box = FancyBboxPatch((8.7, 16), 1.2, 5, 
                             boxstyle="round,pad=0.05", 
                             edgecolor='gray', facecolor='lightgreen', 
                             linewidth=1, alpha=0.7)
    ax.add_patch(adv_box)
    ax.text(9.3, 20.7, 'Advantages', fontsize=8, weight='bold', ha='center')
    
    advantages = [
        '✓ Sequential',
        '  structure',
        '✓ Enforces',
        '  connectivity',
        '✓ Handles',
        '  occlusions',
        '✓ Fine-grained',
        '  detail',
        '✓ Efficient'
    ]
    adv_y = 20.3
    for adv in advantages:
        ax.text(9.3, adv_y, adv, fontsize=6.5, ha='center', va='top')
        adv_y -= 0.44
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    fig = create_architecture_diagram()
    
    # Save the figure
    output_path = '/home/razvan/DTU/thesis/codebase/dqn_row_based/architecture_diagram.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Diagram saved to: {output_path}")
    
    # Also save as PDF for high quality
    pdf_path = output_path.replace('.png', '.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"PDF version saved to: {pdf_path}")
    
    plt.show()

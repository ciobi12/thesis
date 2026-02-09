"""
Generate comparison dataset showing clear vs blurred versions with ground truth masks
for 10 different L-system axioms/patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from l_systems_2d.l_systems_2d import LSystem2DGenerator
from l_systems_2d.examples_2d import RootLSystemLibrary
import os

def generate_comparison_images(output_dir="l_systems_2d/examples/comparison_dataset"):
    """
    Generate 10 different root patterns, each with:
    - Clear version (no occlusion)
    - Extra blur version (with blur and occlusion)
    - Ground truth mask
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Select 10 diverse patterns from the library
    library = RootLSystemLibrary()
    all_systems = library.get_all_root_systems()
    
    # Handpick diverse patterns for better variety
    selected_patterns = [
        'fibrous_dense',
        'taproot_branched',
        'fractal_root',
        'bush_root',
        'pine_tree',
        'adventitious_spreading',
        'weed_root',
        'network_dense',
        'grass_roots',
        'tree_anchor'
    ]
    
    print(f"Generating comparison dataset for {len(selected_patterns)} patterns...")
    print("=" * 70)
    
    for idx, pattern_name in enumerate(selected_patterns, 1):
        config = all_systems[pattern_name]
        print(f"\n{idx}. Processing: {pattern_name}")
        print(f"   Description: {config['description']}")
        
        # Create L-system generator
        lsys = LSystem2DGenerator(
            axiom=config['axiom'],
            rules=config['rules']
        )
        
        # Build the L-system structure
        segments = lsys.build_l_sys(
            iterations=config['iterations'],
            step=config['step'],
            angle_deg=config['angle'],
            start_angle=-90  # Roots grow downward
        )
        
        print(f"   Generated {len(segments)} segments")
        
        # === GROUND TRUTH MASK (Full structure without gaps) ===
        _, ground_truth_mask = lsys.draw_lsystem_ct_style(
            canvas_size=(128, 256),
            margin=10,
            root_width=1,
            add_ct_noise=False,
            occlude_root=False,
            skip_segments=False,  # No gaps for ground truth
            ct_background_intensity=80,
            root_intensity_range=(10, 40),
            align_top=True
        )
        
        # Rebuild for fragmented blur version
        lsys.build_l_sys(
            iterations=config['iterations'],
            step=config['step'],
            angle_deg=config['angle'],
            start_angle=-90
        )
        
        # === EXTRA BLUR VERSION (With occlusion, noise, and gaps) ===
        img_blur, _ = lsys.draw_lsystem_ct_style(
            canvas_size=(128, 256),
            margin=10,
            root_width=1,
            add_ct_noise=True,  # Add noise
            occlude_root=True,  # Add occlusion
            occlusion_strength=0.4,
            skip_segments=True,  # Add gaps
            skip_probability=0.15,
            ct_background_intensity=80,
            root_intensity_range=(10, 40),
            align_top=True
        )
        
        # Save individual images
        blur_path = os.path.join(output_dir, f"{idx:02d}_{pattern_name}_blur.png")
        mask_path = os.path.join(output_dir, f"{idx:02d}_{pattern_name}_mask.png")
        
        from PIL import Image
        Image.fromarray(img_blur.astype(np.uint8)).save(blur_path)
        Image.fromarray((ground_truth_mask * 255).astype(np.uint8)).save(mask_path)
        
        # === CREATE COMPARISON FIGURE ===
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Blurred version
        axes[0].imshow(img_blur, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title('Extra blur', fontsize=14, weight='bold')
        axes[0].axis('off')
        
        # Ground truth mask
        axes[1].imshow(ground_truth_mask, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Ground Truth Mask', fontsize=14, weight='bold')
        axes[1].axis('off')
        
        plt.suptitle(config['description'], 
                     fontsize=12, y=0.98)
        plt.tight_layout()
        
        # Save comparison figure
        comparison_path = os.path.join(output_dir, f"{idx:02d}_{pattern_name}_comparison.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✓ Saved comparison to: {comparison_path}")
    
    print("\n" + "=" * 70)
    print(f"✓ Dataset generation complete!")
    print(f"✓ Location: {output_dir}/")
    print(f"✓ Total files: {len(selected_patterns) * 3} files")
    print(f"   - {len(selected_patterns)} comparison figures")
    print(f"   - {len(selected_patterns)} blur images")
    print(f"   - {len(selected_patterns)} masks")


def generate_combined_overview(output_dir="l_systems_2d/examples/comparison_dataset"):
    """
    Generate a single overview figure showing all 10 patterns in a grid.
    """
    library = RootLSystemLibrary()
    all_systems = library.get_all_root_systems()
    
    selected_patterns = [
        'fibrous_dense',
        'taproot_branched',
        'fractal_root',
        'bush_root',
        'pine_tree',
        'adventitious_spreading',
        'weed_root',
        'network_dense',
        'grass_roots',
        'tree_anchor'
    ]
    
    print("\nGenerating combined overview figure...")
    
    # Create a large grid: 10 rows (patterns) x 2 columns (blur, mask)
    fig, axes = plt.subplots(10, 2, figsize=(8, 30))
    
    for idx, pattern_name in enumerate(selected_patterns):
        config = all_systems[pattern_name]
        
        # Create L-system
        lsys = LSystem2DGenerator(
            axiom=config['axiom'],
            rules=config['rules']
        )
        
        # Generate ground truth mask (full structure)
        lsys.build_l_sys(
            iterations=config['iterations'],
            step=config['step'],
            angle_deg=config['angle'],
            start_angle=-90
        )
        
        _, mask = lsys.draw_lsystem_ct_style(
            canvas_size=(128, 256),
            margin=10,
            root_width=1,
            add_ct_noise=False,
            occlude_root=False,
            skip_segments=False,  # No gaps for ground truth
            ct_background_intensity=80,
            root_intensity_range=(10, 40),
            align_top=True
        )
        
        # Rebuild for blur version
        lsys.build_l_sys(
            iterations=config['iterations'],
            step=config['step'],
            angle_deg=config['angle'],
            start_angle=-90
        )
        
        # Generate blur version with gaps
        img_blur, _ = lsys.draw_lsystem_ct_style(
            canvas_size=(128, 256),
            margin=10,
            root_width=1,
            add_ct_noise=True,
            occlude_root=True,
            occlusion_strength=0.4,
            skip_segments=True,
            skip_probability=0.15,
            ct_background_intensity=80,
            root_intensity_range=(10, 40),
            align_top=True
        )
        
        # Plot in grid
        axes[idx, 0].imshow(img_blur, cmap='gray', vmin=0, vmax=255)
        axes[idx, 0].axis('off')
        if idx == 0:
            axes[idx, 0].set_title('Extra blur', fontsize=12, weight='bold')
        axes[idx, 0].set_ylabel(f"{idx+1}. {config['description']}", fontsize=9, rotation=0, 
                                 ha='right', va='center', labelpad=40)
        
        axes[idx, 1].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[idx, 1].axis('off')
        if idx == 0:
            axes[idx, 1].set_title('Ground Truth Mask', fontsize=12, weight='bold')
    
    plt.suptitle('L-System Root Patterns - Comparison Dataset', 
                 fontsize=16, weight='bold', y=0.995)
    plt.tight_layout()
    
    overview_path = os.path.join(output_dir, "00_overview_all_patterns.png")
    plt.savefig(overview_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved overview to: {overview_path}")


if __name__ == "__main__":
    output_dir = "l_systems_2d/examples/comparison_dataset"
    
    # Generate individual comparison images
    generate_comparison_images(output_dir)
    
    # Generate combined overview
    generate_combined_overview(output_dir)
    
    print("\n" + "=" * 70)
    print("✓ All visualizations complete!")
    print("=" * 70)

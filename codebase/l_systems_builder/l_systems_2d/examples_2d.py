"""
Collection of L-system rules for generating realistic plant root structures.
Each pattern mimics different types of root systems found in nature.
"""

import random
from typing import Dict, List, Tuple
from l_systems_2d.l_systems_2d import LSystem2DGenerator

import numpy as np
import os

class RootLSystemLibrary:
    """Library of L-system patterns for different root types"""
    
    @staticmethod
    def get_all_root_systems() -> Dict[str, Dict]:
        """
        Returns dictionary of all available root systems.
        Each entry contains: axiom, rules, iterations, angle, step, description
        """
        return {
            # === FIBROUS ROOT SYSTEMS ===
            "fibrous_dense": {
                "axiom": "F",
                "rules": {"F": "F[+F]F[-F]F"},
                "iterations": 4,
                "angle": 25.0,
                "step": 8,
                "description": "Dense fibrous roots with regular branching"
            },
            
            "fibrous_sparse": {
                "axiom": "F",
                "rules": {"F": "F[+F][-F]"},
                "iterations": 4,
                "angle": 30.0,
                "step": 10,
                "description": "Sparse fibrous roots with wider angles"
            },
            
            "fibrous_asymmetric": {
                "axiom": "F",
                "rules": {"F": "FF-[F+F+F]+[+F-F-F]"},
                "iterations": 3,
                "angle": 22.5,
                "step": 8,
                "description": "Asymmetric fibrous pattern (like grass roots)"
            },
            
            # === TAP ROOT SYSTEMS ===
            "taproot_simple": {
                "axiom": "F",
                "rules": {"F": "FF[+F][-F]"},
                "iterations": 4,
                "angle": 20.0,
                "step": 10,
                "description": "Simple tap root with lateral branches"
            },
            
            "taproot_deep": {
                "axiom": "F",
                "rules": {"F": "FFF[+F][-F]"},
                "iterations": 4,
                "angle": 15.0,
                "step": 12,
                "description": "Deep tap root with fewer laterals (like carrot)"
            },
            
            "taproot_branched": {
                "axiom": "X",
                "rules": {
                    "X": "F[+X]F[-X]+X",
                    "F": "FF"
                },
                "iterations": 4,
                "angle": 25.0,
                "step": 8,
                "description": "Highly branched tap root system"
            },
            
            # === ADVENTITIOUS ROOTS ===
            "adventitious_spreading": {
                "axiom": "F",
                "rules": {"F": "F[++F][--F]"},
                "iterations": 4,
                "angle": 35.0,
                "step": 8,
                "description": "Spreading adventitious roots"
            },
            
            "adventitious_prop": {
                "axiom": "F",
                "rules": {"F": "F[+F][F][-F]"},
                "iterations": 4,
                "angle": 28.0,
                "step": 10,
                "description": "Prop-like adventitious roots (like corn)"
            },
            
            # === COMPLEX BRANCHING ===
            "fractal_root": {
                "axiom": "X",
                "rules": {
                    "X": "F-[[X]+X]+F[+FX]-X",
                    "F": "FF"
                },
                "iterations": 3,
                "angle": 25.0,
                "step": 6,
                "description": "Fractal-like complex branching"
            },
            
            "bush_root": {
                "axiom": "F",
                "rules": {"F": "FF+[+F-F-F]-[-F+F+F]"},
                "iterations": 3,
                "angle": 22.5,
                "step": 8,
                "description": "Bush-like root system with multiple layers"
            },
            
            "weed_root": {
                "axiom": "F",
                "rules": {"F": "F[+F]F[-F][F]"},
                "iterations": 4,
                "angle": 27.0,
                "step": 7,
                "description": "Weed-type aggressive spreading roots"
            },
            
            # === SPECIALIZED ROOT SYSTEMS ===
            "lateral_dominant": {
                "axiom": "F",
                "rules": {"F": "F[++F][+F][F][-F][--F]"},
                "iterations": 3,
                "angle": 30.0,
                "step": 10,
                "description": "Lateral roots dominate over main root"
            },
            
            "fine_hair": {
                "axiom": "F",
                "rules": {
                    "F": "FF",
                    "X": "F[+X][-X]FX"
                },
                "iterations": 4,
                "angle": 20.0,
                "step": 5,
                "description": "Fine hair-like roots (root hairs)"
            },
            
            "network_dense": {
                "axiom": "F",
                "rules": {"F": "F[+F+F]F[-F-F]F"},
                "iterations": 3,
                "angle": 18.0,
                "step": 8,
                "description": "Dense network of interconnected roots"
            },
            
            "pine_tree": {
                "axiom": "F",
                "rules": {"F": "FF[++F][+F][-F][--F]"},
                "iterations": 4,
                "angle": 32.0,
                "step": 9,
                "description": "Pine tree-like spreading root system"
            },
            
            "shallow_spreading": {
                "axiom": "F",
                "rules": {"F": "F[+++F][++F][+F][-F][--F][---F]"},
                "iterations": 3,
                "angle": 25.0,
                "step": 10,
                "description": "Shallow spreading roots (like willow)"
            },
            
            # === STOCHASTIC VARIANTS ===
            "random_branching_1": {
                "axiom": "F",
                "rules": {"F": "F[+F][-F]F[+F]"},
                "iterations": 4,
                "angle": 23.0,
                "step": 8,
                "description": "Random-style branching pattern 1"
            },
            
            "random_branching_2": {
                "axiom": "F",
                "rules": {"F": "FF[++F][-F][+F]"},
                "iterations": 4,
                "angle": 26.0,
                "step": 9,
                "description": "Random-style branching pattern 2"
            },
            
            "random_branching_3": {
                "axiom": "F",
                "rules": {"F": "F[-F][+F][F]"},
                "iterations": 4,
                "angle": 28.0,
                "step": 7,
                "description": "Random-style branching pattern 3"
            },
            
            # === REALISTIC VARIATIONS ===
            "carrot_root": {
                "axiom": "F",
                "rules": {"F": "FFF[+F][-F]"},
                "iterations": 4,
                "angle": 18.0,
                "step": 11,
                "description": "Carrot-like tap root"
            },
            
            "grass_roots": {
                "axiom": "F",
                "rules": {"F": "F[++F][+F][-F][--F]"},
                "iterations": 4,
                "angle": 35.0,
                "step": 6,
                "description": "Grass fibrous root system"
            },
            
            "tree_anchor": {
                "axiom": "X",
                "rules": {
                    "X": "F[+X][-X]FX",
                    "F": "FF"
                },
                "iterations": 4,
                "angle": 20.0,
                "step": 10,
                "description": "Tree anchor roots (structural)"
            },
            
            "vine_roots": {
                "axiom": "F",
                "rules": {"F": "F[+++F][+F][-F][---F]"},
                "iterations": 3,
                "angle": 40.0,
                "step": 8,
                "description": "Vine-like spreading roots"
            },
            
            "succulent_roots": {
                "axiom": "F",
                "rules": {"F": "FF[+F][-F]"},
                "iterations": 3,
                "angle": 30.0,
                "step": 12,
                "description": "Succulent shallow root system"
            },
        }
    
    @staticmethod
    def get_random_root_system() -> Tuple[str, Dict]:
        """Get a random root system from the library"""
        systems = RootLSystemLibrary.get_all_root_systems()
        name = random.choice(list(systems.keys()))
        return name, systems[name]
    
    @staticmethod
    def get_by_category(category: str) -> Dict[str, Dict]:
        """
        Get root systems by category.
        Categories: 'fibrous', 'taproot', 'adventitious', 'complex', 'specialized'
        """
        all_systems = RootLSystemLibrary.get_all_root_systems()
        return {
            name: config for name, config in all_systems.items()
            if category.lower() in name.lower()
        }
    
    @staticmethod
    def create_variations(base_config: Dict, num_variations: int = 5) -> List[Dict]:
        """
        Create variations of a base configuration by randomly adjusting parameters.
        Useful for data augmentation.
        """
        variations = []
        for _ in range(num_variations):
            var = base_config.copy()
            
            # Randomly vary angle (±5 degrees)
            var['angle'] = base_config['angle'] + random.uniform(-5, 5)
            
            # Randomly vary step (±2 units)
            var['step'] = max(4, base_config['step'] + random.randint(-2, 2))
            
            # Randomly vary iterations (±1)
            var['iterations'] = max(2, min(4, base_config['iterations'] + random.randint(-1, 1)))
            
            variations.append(var)
        
        return variations


# Example usage and dataset generation
if __name__ == "__main__":
    import os
    
    # Create output directory
    output_dir = "l_systems_2d/dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    library = RootLSystemLibrary()
    all_systems = library.get_all_root_systems()
    
    print(f"=== Root L-System Library ===")
    print(f"Total root patterns: {len(all_systems)}\n")
    
    # Generate examples for each system
    for i, (name, config) in enumerate(all_systems.items(), 1):
        print(f"{i}. {name}: {config['description']}")
        
        # Create L-system generator
        lsys = LSystem2DGenerator(
            axiom=config['axiom'],
            rules=config['rules']
        )
        
        # Build the system
        segments = lsys.build_l_sys(
            iterations=config['iterations'],
            step=config['step'],
            angle_deg=config['angle'],
            start_angle=-90  # Roots grow downward
        )
        
        # Generate CT-style image
        img, mask = lsys.draw_lsystem_ct_style(
            canvas_size=(128, 256),
            margin=10,
            root_width=1,
            lsys_save_path=f"{output_dir}/{name}_ct.png",
            mask_save_path=f"{output_dir}/{name}_mask.png",
            add_ct_noise=True,
            occlude_root=True,
            occlusion_strength=0.2,
            skip_segments=False,
            skip_probability=0.5,
            ct_background_intensity=80,
            root_intensity_range=(10, 40),
            align_top=True
        )
        
        print(f"   Saved: {name}_ct.png ({len(segments)} segments)\n")
    
    print("\n=== Generating Variations for Data Augmentation ===")
    
    # Generate variations of the most interesting patterns
    interesting_patterns = ['fibrous_dense', 'taproot_branched', 'fractal_root', 'bush_root']
    
    for pattern_name in interesting_patterns:
        base_config = all_systems[pattern_name]
        variations = library.create_variations(base_config, num_variations=5)
        
        for j, var_config in enumerate(variations):
            lsys = LSystem2DGenerator(
                axiom=var_config['axiom'],
                rules=var_config['rules']
            )
            
            segments = lsys.build_l_sys(
                iterations=var_config['iterations'],
                step=var_config['step'],
                angle_deg=var_config['angle'],
                start_angle=-90
            )
            
            # Vary occlusion and skip parameters too
            img, mask = lsys.draw_lsystem_ct_style(
                canvas_size=(128, 256),
                margin=10,
                root_width=1,
                lsys_save_path=f"{output_dir}/{pattern_name}_var{j+1}_ct.png",
                mask_save_path=f"{output_dir}/{pattern_name}_var{j+1}_mask.png",
                add_ct_noise=True,
                occlude_root=random.choice([True, False]),
                occlusion_strength=random.uniform(0.2, 0.4),
                skip_segments=False,
                skip_probability=random.uniform(0.1, 0.25),
                ct_background_intensity=80,
                root_intensity_range=(10, 40),
                align_top=True
            )
            
            print(f"   {pattern_name}_var{j+1}: angle={var_config['angle']:.1f}, "
                  f"step={var_config['step']}, iter={var_config['iterations']}")
    
    print(f"\n=== Dataset Generation Complete ===")
    print(f"Total images: {len(all_systems)} base + {len(interesting_patterns) * 5} variations")
    print(f"Location: {output_dir}/")
    
    # Print category breakdown
    print("\n=== By Category ===")
    for category in ['fibrous', 'taproot', 'adventitious', 'complex']:
        cat_systems = library.get_by_category(category)
        print(f"{category.capitalize()}: {len(cat_systems)} patterns")

LSYSTEM_PRESETS = {
    # Original plant-like structure
    "plant_basic": {
        "axiom": "F",
        "rules": {"F": "F/[+F]F&[-F]F"},
        "angle": 25,
        "description": "Basic 3D plant with balanced branching"
    },
    
    # Taproot systems (main vertical root with laterals)
    "taproot_simple": {
        "axiom": "F",
        "rules": {"F": "FF[+F][-F]"},
        "angle": 30,
        "description": "Simple taproot with lateral branches"
    },
    "taproot_deep": {
        "axiom": "F",
        "rules": {"F": "FFF[+F][-F][&F]"},
        "angle": 25,
        "description": "Deep taproot with 3D branching"
    },
    "taproot_sparse": {
        "axiom": "F",
        "rules": {"F": "FFFF[+F][-F]"},
        "angle": 35,
        "description": "Very deep taproot with sparse laterals"
    },
    
    # Fibrous root systems (dense, spreading)
    "fibrous_dense": {
        "axiom": "F",
        "rules": {"F": "F[+F][&F][-F][^F]F"},
        "angle": 30,
        "description": "Dense fibrous root network"
    },
    "fibrous_shallow": {
        "axiom": "F",
        "rules": {"F": "F[+F][-F]F[&F][^F]"},
        "angle": 45,
        "description": "Shallow spreading fibrous roots"
    },
    
    # Adventitious roots (multiple starting points)
    "adventitious": {
        "axiom": "F[+F][-F]",
        "rules": {"F": "FF[+F][-F][/F]"},
        "angle": 30,
        "description": "Adventitious root system"
    },
    
    # Dichotomous branching (forking)
    "dichotomous": {
        "axiom": "F",
        "rules": {"F": "F[+F][-F]"},
        "angle": 30,
        "description": "Simple dichotomous forking"
    },
    "dichotomous_3d": {
        "axiom": "F",
        "rules": {"F": "F[+&F][-^F]"},
        "angle": 25,
        "description": "3D dichotomous branching"
    },
    
    # Herringbone pattern
    "herringbone": {
        "axiom": "F",
        "rules": {"F": "F[-F]FF[+F]F"},
        "angle": 35,
        "description": "Herringbone branching pattern"
    },
    "herringbone_3d": {
        "axiom": "F",
        "rules": {"F": "F[-F]F[&F]F[+F]F"},
        "angle": 30,
        "description": "3D herringbone pattern"
    },
    
    # Tree-like structures
    "tree_binary": {
        "axiom": "F",
        "rules": {"F": "FF[+F][−F]"},
        "angle": 25,
        "description": "Binary tree structure"
    },
    "tree_ternary": {
        "axiom": "F",
        "rules": {"F": "F[+F]F[-F]F"},
        "angle": 20,
        "description": "Ternary branching tree"
    },
    "tree_bushy": {
        "axiom": "F",
        "rules": {"F": "F[+F][&F]F[-F][^F]F"},
        "angle": 22,
        "description": "Bushy tree with many branches"
    },
    
    # Spiral patterns
    "spiral": {
        "axiom": "F",
        "rules": {"F": "F[+F]/F[-F]\\F"},
        "angle": 30,
        "description": "Spiral growth pattern"
    },
    "helix": {
        "axiom": "F",
        "rules": {"F": "F/[+F]\\[-F]F"},
        "angle": 25,
        "description": "Helical growth"
    },
    
    # Coral-like structures
    "coral_branching": {
        "axiom": "F",
        "rules": {"F": "FF[+F][+F][-F][-F]"},
        "angle": 25,
        "description": "Coral-like branching"
    },
    "coral_3d": {
        "axiom": "F",
        "rules": {"F": "F[+F][&F]F[-F][^F]"},
        "angle": 30,
        "description": "3D coral structure"
    },
    
    # Fern-like patterns
    "fern": {
        "axiom": "X",
        "rules": {"X": "F+[[X]-X]-F[-FX]+X", "F": "FF"},
        "angle": 25,
        "description": "Fern-like frond pattern"
    },
    
    # Sparse long roots
    "root_long_sparse": {
        "axiom": "F",
        "rules": {"F": "FFFFF[+F][-F]"},
        "angle": 30,
        "description": "Very long roots with sparse branching"
    },
    "root_asymmetric": {
        "axiom": "F",
        "rules": {"F": "FF[+F]FFF[-F]F"},
        "angle": 35,
        "description": "Asymmetric root branching"
    },
    
    # Complex 3D patterns
    "complex_3d_a": {
        "axiom": "F",
        "rules": {"F": "F[+F&F]F[-F^F]F"},
        "angle": 25,
        "description": "Complex 3D pattern A"
    },
    "complex_3d_b": {
        "axiom": "F",
        "rules": {"F": "F[/+F][\\-F]F[&F][^F]"},
        "angle": 28,
        "description": "Complex 3D pattern B"
    },
    
    # Vine-like structures
    "vine": {
        "axiom": "F",
        "rules": {"F": "F[+F]F[-F][F]"},
        "angle": 20,
        "description": "Vine-like growth"
    },
    "vine_3d": {
        "axiom": "F",
        "rules": {"F": "F/[+F]F\\[-F]F"},
        "angle": 22,
        "description": "3D vine structure"
    },
    
    # Lightning/crack patterns
    "lightning": {
        "axiom": "F",
        "rules": {"F": "F[++F][--F]F"},
        "angle": 15,
        "description": "Lightning bolt pattern"
    },
    
    # Alternating branching
    "alternating": {
        "axiom": "F",
        "rules": {"F": "F[+F]F", "F": "F[-F]F"},  # Note: will use last rule
        "angle": 30,
        "description": "Alternating branch sides"
    },
    "alternating_3d": {
        "axiom": "A",
        "rules": {"A": "F[+A]FB", "B": "F[-B]FA", "F": "FF"},
        "angle": 30,
        "description": "Alternating 3D branches"
    },
}
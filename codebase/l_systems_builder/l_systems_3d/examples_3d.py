import os
import math
from l_systems_3d.l_systems_3d import LSystem3DGenerator

EXAMPLES_DICT = {
    "bush_tree": dict(axiom="A",
                 rules={"A": "F[&+A][&-A][^+A][^-A]",
                        "F": "FF"}),

    "spiral": dict(axiom="X",
                   rules={"X": "F[/&X][//&X][///&X][////&X]",
                          "F": "FF"}),

    "complex": dict(axiom="F",
                    rules={"F": "F[&+F][&-F][^+F][^-F][/F][\\F]"}),

    "fractal": dict(axiom="X",
                    rules={"X": "F[//&+X][//^-X][&+X][^-X]",
                           "F": "FF"}),

    "simpodial": dict(axiom="A",
                      rules={"A": "F[&B]////[&B]////[&B]",
                             "B": "F[//&C]",
                             "C": "F[+F][-F]"}),

    "hilbert": dict(axiom="X",
                    rules={"X": "^<XF^<XFX-F^>>XFX&F+>>XFX-F>X->",
                           "F": "F"}),

    "bushy": dict(axiom="A",
                  rules={"A": "F[/&FA]F[//&FA]F[///&FA]F[////&FA]",
                         "F": "FF"}),

    "monopodial": dict(axiom="A",
                       rules={"A": "F[&+B]F[&-B]FA",
                              "B": "F[//+C][//-C]",
                              "C": "F"}),

    "seaweed": dict(axiom="F",
                    rules={"F": "FF[/&+F][//&-F][&+F]"}),

    "ternary": dict(axiom="F",
                    rules={"F": "F[&&&+F][&-F][^+F]"})
    
    }

def example_usage(iterations, struct_name: str, **kwargs):
    

    # Create an instance of the LSystem3DGenerator
    l_system = LSystem3DGenerator(**kwargs)

    # Build the L-system with specified iterations and parameters
    segments = l_system.build_l_sys(iterations, step = 5, angle_deg = 30)

    # Render the generated 3D L-system
    l_system.render(elev = 30, azim = 35, save_fig = True, filename = struct_name)
    l_system.to_voxel_grid(grid_size = (64,64, 64),
                           align_bottom=True,
                           save_voxel_grid = True,  
                           filepath = os.path.join("examples", "volumes", struct_name + '.npy'))

if __name__ == "__main__":
    for struct_name in EXAMPLES_DICT.keys():
       print(struct_name, '/n')
       if struct_name == 'complex':
              example_usage(2, struct_name = struct_name, **EXAMPLES_DICT[struct_name])
       else:
              example_usage(3, struct_name = struct_name, **EXAMPLES_DICT[struct_name])
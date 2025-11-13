from l_systems import LSystemGenerator

import numpy as np
import os

rules = {"plant": {"X": "F[-X][X]F[-X]+FX",
                    "F": "FF"},
         "bush": {"F": "FF-[-F+F+F]+[+F-F-F]"},
         "tree": {"X": "F[-X][X]F[-X]+FX",
                   "F": "FF"},
         "fern": {"X": "F+[[X]-X]-F[-FX]+X",
                  "F": "F[F]-F"},
         "fractal":{"X": "F[+X][-X]FX",
                    "F": "FF"},
         "palm":  {"X": "F[+X][-X]FX",
                   "F": "FF"}
         }

if __name__ == "__main__":
    for key in rules.keys():
        lsys_obj = LSystemGenerator(axiom = "X" if key != "bush" else "F", rules = rules[key])
        for _ in range(2):
            segments = lsys_obj.build_l_sys(iterations = np.random.randint(2, 4), 
                                            step = np.random.randint(5, 10),
                                            start_angle = 85 + 10 * np.random.rand(), 
                                            angle_deg = 20 + 10 * np.random.rand())
            img, mask = lsys_obj.draw_lsystem(canvas_size = (128, 256),
                                              lsys_save_path = os.path.join("data/noise_only", f"{key}_{_}.png"),
                                              mask_save_path = os.path.join("data/noise_only", f"{key}_{_}_mask.png"),
                                              add_artifacts=False)

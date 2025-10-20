from l_systems import LSystemGenerator

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


         
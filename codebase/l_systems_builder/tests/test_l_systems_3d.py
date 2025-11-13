import unittest
from l_systems_3d.l_systems_3d import LSystem3DGenerator

class TestLSystem3DGenerator(unittest.TestCase):

    def setUp(self):
        self.axiom = "F"
        self.rules = {"F": "F+F-F-F+F"}
        self.lsystem = LSystem3DGenerator(self.axiom, self.rules)

    def test_build_l_sys(self):
        segments = self.lsystem.build_l_sys(iterations=2)
        self.assertGreater(len(segments), 0, "The generated segments should not be empty.")

    def test_render(self):
        self.lsystem.build_l_sys(iterations=2)
        # Here we would normally check if the rendering function works, 
        # but since it shows a plot, we will just ensure it runs without error.
        try:
            self.lsystem.render()
        except Exception as e:
            self.fail(f"Rendering failed with exception: {e}")

    def test_stack_operations(self):
        self.lsystem.build_l_sys(iterations=1)
        # Check if the stack operations work correctly
        self.assertEqual(len(self.lsystem.segments), 4, "The number of segments should match the expected output.")

if __name__ == '__main__':
    unittest.main()
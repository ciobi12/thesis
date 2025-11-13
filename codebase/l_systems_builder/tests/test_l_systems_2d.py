import unittest
from l_systems_2d.l_systems_2d import LSystem2DGenerator

class TestLSystem2DGenerator(unittest.TestCase):

    def setUp(self):
        self.axiom = "F"
        self.rules = {"F": "F+F-F-F+F"}
        self.l_system = LSystem2DGenerator(self.axiom, self.rules)

    def test_initialization(self):
        self.assertEqual(self.l_system.axiom, self.axiom)
        self.assertEqual(self.l_system.rules, self.rules)

    def test_build_l_sys(self):
        iterations = 2
        expected_segments = [
            # Add expected segments based on the axiom and rules
        ]
        segments = self.l_system.build_l_sys(iterations)
        self.assertEqual(len(segments), len(expected_segments))
        # Further assertions can be added to check the actual segments

    def test_render(self):
        iterations = 2
        self.l_system.build_l_sys(iterations)
        # Here you would typically check if the rendering function runs without errors
        # Since rendering is visual, we may not have a direct assertion

if __name__ == "__main__":
    unittest.main()
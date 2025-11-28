# L-System Generator

This project implements Lindenmayer systems (L-systems) in both 2D and 3D. L-systems are a mathematical formalism used to model the growth processes of plants and other organisms. They are particularly useful for generating fractal-like structures and can be applied in computer graphics, procedural generation, and more.

## Project Structure

- **l_systems_2d/l_systems.py**: Contains the `LSystem2DGenerator` class for generating and rendering 2D L-systems.
  
- **l_systems_3d/l_systems_3d.py**: Contains the `LSystem3DGenerator` class for generating and rendering 3D L-systems.

- **l_systems_3d/examples_3d.py**: Provides example usages of the `LSystem3DGenerator` class, demonstrating how to create and visualize 3D L-systems.

- **tests/test_l_systems_2d.py**: Contains unit tests for the `LSystem2DGenerator` class.

- **tests/test_l_systems_3d.py**: Contains unit tests for the `LSystem3DGenerator` class.

- **requirements.txt**: Lists the dependencies required for the project.

## Usage

### 2D L-Systems

To generate a 2D L-system, instantiate the `LSystem2DGenerator` class with an axiom and a set of rules. Call the `build_l_sys` method to generate the L-system string, and use the `render` method to visualize it.

### 3D L-Systems

To generate a 3D L-system, instantiate the `LSystem3DGenerator` class similarly. Use the `build_l_sys` method to create the 3D structure and visualize it using the `render` method.

## Examples

Refer to the `examples_3d.py` file for examples of how to use the 3D L-system generator.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
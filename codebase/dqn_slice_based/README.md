# DQN Slice-Based Segmentation

This folder contains a slice-wise DQN approach for 3D volume segmentation.

## Overview

Unlike the row-based approach that processes 2D images row by row, this approach processes 3D volumes slice by slice:

- **Input**: 3D volumes of shape `(D, H, W)` where:
  - `D = 64` (number of slices/depth)
  - `H = 16` (height of each slice)
  - `W = 16` (width of each slice)

- **Processing**: The agent traverses the volume from bottom to top (or top to bottom), making decisions for each entire slice (16x16 pixels) at a time.

## Key Differences from Row-Based

1. **Architecture**: Uses Conv2d layers instead of Conv1d to process 2D slices
2. **Action Space**: Each action is a 2D binary mask (H×W) instead of 1D (W)
3. **History**: Maintains history of previous slices (not rows)
4. **Environment**: Processes volumes slice by slice

## Files

- `env.py`: Environment for slice-wise volume traversal
- `dqn.py`: Neural network architectures using Conv2d for slice processing
- `main.py`: Training loop and utilities for volume-based learning

## Usage

Expects `.npy` files containing 3D volumes in the data directory structure. The training automatically loads volumes and processes them slice by slice.

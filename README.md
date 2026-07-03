# Autonomous Planar Navigation and Obstacle Avoidance in MuJoCo

## Project Overview
This repository contains a complete autonomous planar navigation pipeline developed for a differential-drive robot operating within the MuJoCo physics engine simulator. The system transitions from a point-robot configuration space assumption during global planning to an expanded physical layout incorporating safety layers for realistic collision avoidance in cluttered environments.

The codebase features three unique path-planning frameworks evaluated across static and randomized obstacle maps, alongside a local LiDAR safety layer that dynamically scales and adjusts tracking commands.

---

## Architecture and Navigation Pipeline

### 1. Global Path Planners
* **Grid-Based A* Search:** Discretizes the workspace into an 8-connected occupancy grid at a 0.10 m resolution. It relies on an admissible Euclidean distance heuristic combined with shortcut smoothing and linear path densification to guarantee deterministic completeness and optimal trajectory configurations.
* **Rapidly-exploring Random Trees (RRT):** Conducts exploration in continuous space by incrementally growing a search tree from the start coordinates toward uniformly sampled configuration spaces, using a 10% target bias step to accelerate goal arrival.
* **Artificial Potential Fields (APF):** Models continuous motion using virtual forces where an attractive gradient field pulls the robot toward the target destination while rectangular and map-boundary obstacles generate repulsive vectors. An automatic fallback loop hands over tracking directly to the A* planner if the probe remains trapped in a local minimum saddle point for over 300 steps.

### 2. Control and Local Obstacle Avoidance
* **Waypoint Tracker:** Evaluates a lookahead distance vector 0.75 m along the calculated trajectory, adjusting forward speeds proportional to distance boundaries and turning rates relative to active heading errors.
* **Reactive LiDAR Layer:** Intercepts incoming twist commands to inspect three distinct laser-scan sectors (front, left, right). It dynamically decelerates the robot's linear progression when detecting front obstructions closer than 1.05 m and actively forces zero-forward rotation maneuvers away from hazards if the proximity boundary violates a critical 0.38 m threshold.

---

## File Structure
* `planner_for_stu.py`: Core ROS2 navigation node containing the geometry calculations, coordinate transforms, planning heuristics, and LiDAR filtering logic.
* `CS_475__Assignment_6.pdf`: Final comprehensive engineering report detailing experimental data parameters, algorithmic benchmarks, and failure cases under high-density layout constraints.

#!/usr/bin/env python3

# Iliana Platona csdp1436

import heapq
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray


Point = Tuple[float, float]
Cell = Tuple[int, int]


@dataclass(frozen=True)
class ObstacleBox:
    cx: float
    cy: float
    hx: float
    hy: float

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (self.cx - self.hx, self.cx + self.hx, self.cy - self.hy, self.cy + self.hy)


class AutonomousPlanner(Node):
    """
    Student implementation file for Assignment 6.

    The ROS2 wiring, simulator interface, replanning schedule, path publishing,
    waypoint tracking, and low-level path-following controller are already here.
    Students should complete the TODO methods marked below:

    - plan_astar()
    - plan_rrt()
    - plan_apf()
    - apply_lidar_avoidance()
    - the planning helper functions below those methods

    All coordinates in this file are in the odom/world frame unless a comment says
    otherwise. Obstacles arrive from /obstacle_boxes as rectangles described by
    center x/y and half-size x/y. Implement collision helpers so they account for
    robot radius and obstacle inflation.
    """

    def __init__(self) -> None:
        super().__init__("autonomous_planner")

        self.declare_parameter("planner_algorithm", "astar")
        self.declare_parameter("goal_x", 4.0)
        self.declare_parameter("goal_y", 3.6)
        self.declare_parameter("map_limit", 5.0)
        self.declare_parameter("map_resolution", 0.10)
        self.declare_parameter("robot_radius", 0.22)
        self.declare_parameter("obstacle_inflation", 0.12)
        self.declare_parameter("replan_period_s", 3.0)
        self.declare_parameter("random_seed", 7)

        self.declare_parameter("max_linear_speed", 1.00)
        self.declare_parameter("max_angular_speed", 2.30)
        self.declare_parameter("linear_gain", 1.65)
        self.declare_parameter("angular_gain", 2.45)
        self.declare_parameter("waypoint_tolerance", 0.20)
        self.declare_parameter("goal_tolerance", 0.25)
        self.declare_parameter("lookahead_distance", 0.75)
        self.declare_parameter("heading_slow_angle", 1.00)

        self.declare_parameter("front_sector_deg", 32.0)
        self.declare_parameter("side_sector_min_deg", 30.0)
        self.declare_parameter("side_sector_max_deg", 95.0)
        self.declare_parameter("front_slow_distance", 1.05)
        self.declare_parameter("front_stop_distance", 0.38)
        self.declare_parameter("avoidance_gain", 0.55)

        self.declare_parameter("rrt_step_size", 0.28)
        self.declare_parameter("rrt_goal_sample_rate", 0.10)
        self.declare_parameter("rrt_max_iterations", 4500)
        self.declare_parameter("rrt_replan_min_improvement_m", 0.20)

        self.declare_parameter("apf_step_size", 0.08)
        self.declare_parameter("apf_max_steps", 1400)
        self.declare_parameter("apf_attractive_gain", 1.0)
        self.declare_parameter("apf_repulsive_gain", 0.30)
        self.declare_parameter("apf_influence_distance", 0.85)

        self.planner_algorithm = (
            str(self.get_parameter("planner_algorithm").value).strip().lower()
        )
        if self.planner_algorithm not in {"astar", "rrt", "apf"}:
            self.get_logger().warning(
                f"Unknown planner_algorithm '{self.planner_algorithm}', using astar."
            )
            self.planner_algorithm = "astar"

        self.goal = (
            float(self.get_parameter("goal_x").value),
            float(self.get_parameter("goal_y").value),
        )
        self.map_limit = float(self.get_parameter("map_limit").value)
        self.map_resolution = float(self.get_parameter("map_resolution").value)
        self.robot_radius = float(self.get_parameter("robot_radius").value)
        self.obstacle_inflation = float(self.get_parameter("obstacle_inflation").value)
        self.replan_period_s = float(self.get_parameter("replan_period_s").value)

        self.max_linear_speed = float(self.get_parameter("max_linear_speed").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.linear_gain = float(self.get_parameter("linear_gain").value)
        self.angular_gain = float(self.get_parameter("angular_gain").value)
        self.waypoint_tolerance = float(self.get_parameter("waypoint_tolerance").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.lookahead_distance = float(self.get_parameter("lookahead_distance").value)
        self.heading_slow_angle = float(self.get_parameter("heading_slow_angle").value)

        self.front_sector = math.radians(float(self.get_parameter("front_sector_deg").value))
        self.side_sector_min = math.radians(
            float(self.get_parameter("side_sector_min_deg").value)
        )
        self.side_sector_max = math.radians(
            float(self.get_parameter("side_sector_max_deg").value)
        )
        self.front_slow_distance = float(self.get_parameter("front_slow_distance").value)
        self.front_stop_distance = float(self.get_parameter("front_stop_distance").value)
        self.avoidance_gain = float(self.get_parameter("avoidance_gain").value)

        self.rrt_step_size = float(self.get_parameter("rrt_step_size").value)
        self.rrt_goal_sample_rate = float(self.get_parameter("rrt_goal_sample_rate").value)
        self.rrt_max_iterations = int(self.get_parameter("rrt_max_iterations").value)
        self.rrt_replan_min_improvement_m = max(
            0.0, float(self.get_parameter("rrt_replan_min_improvement_m").value)
        )

        self.apf_step_size = float(self.get_parameter("apf_step_size").value)
        self.apf_max_steps = int(self.get_parameter("apf_max_steps").value)
        self.apf_attractive_gain = float(self.get_parameter("apf_attractive_gain").value)
        self.apf_repulsive_gain = float(self.get_parameter("apf_repulsive_gain").value)
        self.apf_influence_distance = float(
            self.get_parameter("apf_influence_distance").value
        )

        seed = int(self.get_parameter("random_seed").value)
        self.rng = np.random.default_rng(seed)
        self.warned_todos = set()

        self.latest_pose: Optional[Tuple[float, float, float]] = None
        self.obstacles: List[ObstacleBox] = []
        self.scan_ranges: Optional[np.ndarray] = None
        self.scan_angles: Optional[np.ndarray] = None
        self.scan_range_max = 8.0

        self.path: List[Point] = []
        self.waypoint_index = 0
        self.last_plan_time = -1.0e9
        self.plan_requested = True

        self.create_subscription(Odometry, "/odom", self.odom_callback, 20)
        self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.create_subscription(Float32MultiArray, "/obstacle_boxes", self.box_callback, 10)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.path_pub = self.create_publisher(Path, "/path", 10)
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)

        self.control_timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info(
            "Student autonomous planner started. "
            f"algorithm={self.planner_algorithm}, goal=({self.goal[0]:.2f}, {self.goal[1]:.2f})"
        )
        self.get_logger().info(
            "Complete the TODO methods in autonomous_planner.py before expecting motion."
        )

    @staticmethod
    def wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def distance(a: Point, b: Point) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def warn_unimplemented(self, method_name: str) -> None:
        """Log each TODO warning once so the terminal stays readable."""
        if method_name in self.warned_todos:
            return
        self.warned_todos.add(method_name)
        self.get_logger().warning(
            f"{method_name} is not implemented yet; returning no planner action."
        )

    @staticmethod
    def yaw_from_quaternion(q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg: Odometry) -> None:
        pose = msg.pose.pose
        self.latest_pose = (
            float(pose.position.x),
            float(pose.position.y),
            self.yaw_from_quaternion(pose.orientation),
        )

    def scan_callback(self, msg: LaserScan) -> None:
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        if ranges.size == 0:
            return
        ranges = np.nan_to_num(
            ranges,
            nan=float(msg.range_max),
            posinf=float(msg.range_max),
            neginf=float(msg.range_max),
        )
        ranges = np.clip(ranges, float(msg.range_min), float(msg.range_max))
        angles = float(msg.angle_min) + np.arange(ranges.size, dtype=np.float32) * float(
            msg.angle_increment
        )
        self.scan_ranges = ranges
        self.scan_angles = angles
        self.scan_range_max = float(msg.range_max)

    def box_callback(self, msg: Float32MultiArray) -> None:
        data = list(msg.data)
        if len(data) % 4 != 0:
            self.get_logger().warning("/obstacle_boxes length is not a multiple of 4.")
            return
        new_obstacles = [
            ObstacleBox(float(data[i]), float(data[i + 1]), float(data[i + 2]), float(data[i + 3]))
            for i in range(0, len(data), 4)
        ]
        if self.obstacles_changed(new_obstacles):
            self.plan_requested = True
        self.obstacles = new_obstacles

    def obstacles_changed(self, new_obstacles: Sequence[ObstacleBox]) -> bool:
        if len(new_obstacles) != len(self.obstacles):
            return True
        for old, new in zip(self.obstacles, new_obstacles):
            if (
                abs(old.cx - new.cx) > 1e-6
                or abs(old.cy - new.cy) > 1e-6
                or abs(old.hx - new.hx) > 1e-6
                or abs(old.hy - new.hy) > 1e-6
            ):
                return True
        return False

    def control_loop(self) -> None:
        self.publish_goal()
        if self.latest_pose is None:
            self.publish_stop()
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        if self.should_replan(now):
            self.plan_from_current_pose(now)

        if not self.path:
            self.publish_stop()
            return

        x, y, yaw = self.latest_pose
        robot = (x, y)
        if self.distance(robot, self.goal) <= self.goal_tolerance:
            self.path = []
            self.publish_path([])
            self.publish_stop()
            self.get_logger().info("Goal reached; stopping autonomous controller.")
            return

        self.advance_waypoint(robot)
        target = self.lookahead_target(robot)
        cmd = self.compute_tracking_command(robot, yaw, target)
        self.apply_lidar_avoidance(cmd)
        self.cmd_pub.publish(cmd)

    def should_replan(self, now: float) -> bool:
        if not self.obstacles or self.latest_pose is None:
            return False
        if self.plan_requested:
            return True
        if not self.path:
            return True
        return now - self.last_plan_time >= self.replan_period_s

    def plan_from_current_pose(self, now: float) -> None:
        if self.latest_pose is None:
            return
        start = (self.latest_pose[0], self.latest_pose[1])
        if self.distance(start, self.goal) <= self.goal_tolerance:
            return

        if self.planner_algorithm == "astar":
            planned = self.plan_astar(start, self.goal)
        elif self.planner_algorithm == "rrt":
            planned = self.plan_rrt(start, self.goal)
        else:
            planned = self.plan_apf(start, self.goal)

        if not planned and self.planner_algorithm != "astar":
            self.get_logger().warning(
                f"{self.planner_algorithm} failed; falling back to A* for this replan."
            )
            planned = self.plan_astar(start, self.goal)

        if planned:
            accept_plan, decision_msg = self.should_accept_planned_path(start, planned)
            if not accept_plan:
                self.last_plan_time = now
                self.plan_requested = False
                self.publish_path(self.path)
                self.get_logger().info(decision_msg)
                return

            self.path = planned
            self.waypoint_index = 0
            self.advance_waypoint(start)
            self.publish_path(planned)
            self.last_plan_time = now
            self.plan_requested = False
            if decision_msg:
                self.get_logger().info(decision_msg)
            self.get_logger().info(
                f"Planned {len(planned)} waypoints with {self.planner_algorithm}."
            )
        else:
            self.publish_stop()
            self.get_logger().error("No collision-free path found.")
            self.last_plan_time = now
            self.plan_requested = False

    def should_accept_planned_path(self, start: Point, planned: Sequence[Point]) -> Tuple[bool, str]:
        if self.planner_algorithm != "rrt" or not self.path:
            return True, ""

        current_remaining = self.remaining_current_path(start)
        if len(current_remaining) < 2:
            return True, ""

        current_length = self.path_length(current_remaining)
        new_length = self.path_length(planned)
        if current_length <= self.goal_tolerance:
            return True, ""

        if not self.path_is_free(current_remaining):
            return (
                True,
                "Current RRT path is no longer collision-free; accepting the new plan.",
            )

        improvement = current_length - new_length
        if improvement >= self.rrt_replan_min_improvement_m:
            return (
                True,
                f"Accepting new RRT path: {new_length:.2f} m is {improvement:.2f} m shorter "
                f"than the remaining current path ({current_length:.2f} m).",
            )

        return (
            False,
            f"Keeping current RRT path: new path is {new_length:.2f} m, remaining current "
            f"path is {current_length:.2f} m, improvement {improvement:.2f} m is below "
            f"{self.rrt_replan_min_improvement_m:.2f} m.",
        )

    def remaining_current_path(self, robot: Point) -> List[Point]:
        idx = self.waypoint_index
        while idx < len(self.path) - 1:
            if self.distance(robot, self.path[idx]) > self.waypoint_tolerance:
                break
            idx += 1
        return [robot] + list(self.path[idx:])

    @staticmethod
    def path_length(path: Sequence[Point]) -> float:
        if len(path) < 2:
            return 0.0
        return float(
            sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path[:-1], path[1:]))
        )

    def path_is_free(self, path: Sequence[Point]) -> bool:
        """
        TODO: return True only when every waypoint and every connecting segment
        in path is collision-free.

        This helper is useful for RRT replanning decisions and for testing any
        path produced by A*, RRT, or APF.
        """
        if len(path) < 2:
            return True

        for point in path:
            if not self.point_is_free(point):
                return False

        # checks every connecting segment between consecutive waypoints
        for a, b in zip(path[:-1], path[1:]):
            if not self.segment_is_free(a, b):
                return False

        return True
        # raise NotImplementedError("TODO: implement path_is_free()")

    def plan_astar(self, start: Point, goal: Point) -> List[Point]:
        """
        TODO: implement grid-based A*.

        Expected behavior:
        1. Convert the continuous world square into an occupancy grid. A good
           default is to use self.map_resolution for cell size and the square
           [-self.map_limit, self.map_limit] in both x and y.
        2. Inflate every obstacle by self.robot_radius + self.obstacle_inflation
           before marking cells occupied. This keeps the vehicle body away from
           walls and boxes.
        3. Convert start and goal from world coordinates to grid cells. Return []
           if either one is outside the grid or occupied.
        4. Search with 8-connected moves. Cardinal moves cost resolution;
           diagonal moves cost sqrt(2) * resolution.
        5. Use self.heuristic(cell, goal_cell, resolution) for the A* priority.
        6. Store parents in a dictionary so self.reconstruct_cells() can recover
           the cell path when the goal is reached.
        7. Convert cells back to world coordinates, force the first waypoint to
           exactly equal start and the last waypoint to exactly equal goal, then
           return self.smooth_path(path).

        Possible helper functions to implement below:
        - self.cell_in_grid(): decide whether a row/column index is valid.
        - self.heuristic(): estimate remaining cost from one cell to another.
        - self.reconstruct_cells(): walk backward through came_from.
        - self.inflated_bounds(): expand obstacle rectangles before marking them.
        - self.smooth_path(): optionally remove unnecessary intermediate points.
        """
        res = self.map_resolution
        limit = self.map_limit
        n = int(2 * limit / res)
        inflation = self.robot_radius + self.obstacle_inflation
        inflated = [self.inflated_bounds(obs, inflation) for obs in self.obstacles]

        def to_cell(x, y):
            return (int((y + limit) / res), int((x + limit) / res))

        def to_world(r, c):
            return (-limit + c * res + res * 0.5, -limit + r * res + res * 0.5)

        def occupied(r, c):
            x, y = to_world(r, c)
            if abs(x) > limit - self.robot_radius or abs(y) > limit - self.robot_radius:
                return True
            return any(xmin <= x <= xmax and ymin <= y <= ymax for xmin, xmax, ymin, ymax in inflated)

        start_c, goal_c = to_cell(*start), to_cell(*goal)
        if occupied(*goal_c):
            return []

        g_cost = {start_c: 0.0}
        came_from: Dict[Cell, Cell] = {}
        heap = [(self.heuristic(start_c, goal_c, res), 0.0, start_c)]
        sqrt2 = math.sqrt(2)
        moves = [(-1,0,res),(1,0,res),(0,-1,res),(0,1,res),
                 (-1,-1,sqrt2*res),(-1,1,sqrt2*res),(1,-1,sqrt2*res),(1,1,sqrt2*res)]

        while heap:
            f, g, cur = heapq.heappop(heap)
            if g > g_cost.get(cur, float("inf")):
                continue
            if cur == goal_c:
                raw = [to_world(r, c) for r, c in self.reconstruct_cells(came_from, cur)]
                raw[0], raw[-1] = start, goal
                return self.smooth_path(raw)
            for dr, dc, cost in moves:
                nb = (cur[0] + dr, cur[1] + dc)
                if not self.cell_in_grid(nb, n, n) or occupied(*nb):
                    continue
                new_g = g + cost
                if new_g < g_cost.get(nb, float("inf")):
                    g_cost[nb] = new_g
                    came_from[nb] = cur
                    heapq.heappush(heap, (new_g + self.heuristic(nb, goal_c, res), new_g, nb))

        # self.warn_unimplemented("plan_astar")
        return []

    @staticmethod
    def cell_in_grid(cell: Cell, width: int, height: int) -> bool:
        """
        TODO: return True when cell=(row, col) lies inside the grid.

        Students may need this helper for A* neighbor validation.
        """
        r, c = cell
        return 0 <= r < height and 0 <= c < width   # must be within the grid dimensions for a cell to be valid
        # raise NotImplementedError("TODO: implement cell_in_grid()")

    @staticmethod
    def heuristic(a: Cell, b: Cell, resolution: float) -> float:
        """
        TODO: compute an admissible A* heuristic between two cells.

        A common choice for an 8-connected grid is Euclidean distance multiplied
        by the map resolution.
        """
        return math.hypot(a[0] - b[0], a[1] - b[1]) * resolution
        # raise NotImplementedError("TODO: implement heuristic()")

    @staticmethod
    def reconstruct_cells(came_from: Dict[Cell, Cell], current: Cell) -> List[Cell]:
        """
        TODO: reconstruct a cell path from the A* parent dictionary.

        Start at current, repeatedly look up its parent in came_from, then reverse
        the resulting list so it runs from start to goal.
        """
        path = []
        while current in came_from: # walks backwards from the goal through the parent dictionary
            path.append(current)
            current = came_from[current]
        path.append(current)    # to collect reverse also
        path.reverse()
        return path
        # raise NotImplementedError("TODO: implement reconstruct_cells()")

    def plan_rrt(self, start: Point, goal: Point) -> List[Point]:
        """
        TODO: implement Rapidly-exploring Random Tree planning.

        Expected behavior:
        1. Return [] immediately if start or goal is not collision-free. Use
           self.point_is_free().
        2. Store tree nodes as tuples (x, y, parent_index). The first node should
           be (start_x, start_y, -1).
        3. Repeat up to self.rrt_max_iterations:
           - With probability self.rrt_goal_sample_rate, choose the goal as the
             sample. Otherwise sample x/y uniformly inside the map limits.
           - Find the nearest existing tree node. You may want to implement
             self.nearest_node() for this.
           - Extend from that node toward the sample by self.rrt_step_size. You
             may want to implement self.steer() for this.
           - Reject the extension unless self.segment_is_free(near, new_point)
             is true.
           - Add the new node with its parent index.
           - If the new node can connect to the goal, append the goal node and
             reconstruct the path with your self.reconstruct_rrt() helper.
        4. Return self.smooth_path(path) when successful, or [] if the iteration
           budget is exhausted.

        The launch parameter rrt_replan_min_improvement_m is handled elsewhere:
        this method only needs to generate a valid candidate path.
        """
        if not self.point_is_free(goal):
            return []

        # start node has no parent (-1)
        nodes = [(start[0], start[1], -1)]

        for _ in range(self.rrt_max_iterations):
            if self.rng.random() < self.rrt_goal_sample_rate:
                sample = goal
            else:
                sample = (
                    self.rng.uniform(-self.map_limit, self.map_limit),
                    self.rng.uniform(-self.map_limit, self.map_limit),
                )

            # finds closest existing node and extend toward the sample
            near_idx = self.nearest_node(nodes, sample)
            near = (nodes[near_idx][0], nodes[near_idx][1])
            new_pt = self.steer(near, sample, self.rrt_step_size)

            # only adds the new point if the path to it is collision-free
            if not self.segment_is_free(near, new_pt):
                continue

            nodes.append((new_pt[0], new_pt[1], near_idx))
            new_idx = len(nodes) - 1

            # checks if we can connect directly to the goal from here
            if (self.distance(new_pt, goal) <= self.rrt_step_size
                    and self.segment_is_free(new_pt, goal)):
                nodes.append((goal[0], goal[1], new_idx))
                path = self.reconstruct_rrt(nodes, len(nodes) - 1)
                return self.smooth_path(path)

        self.get_logger().error("RRT: max iterations reached, no path found.")
        # self.warn_unimplemented("plan_rrt")
        return []

    @staticmethod
    def nearest_node(nodes: Sequence[Tuple[float, float, int]], sample: Point) -> int:
        """
        TODO: return the index of the RRT node closest to sample.

        Each node is stored as (x, y, parent_index). Squared Euclidean distance
        is enough for comparison; the square root is not necessary.
        """
        best, best_idx = float("inf"), 0
        sx, sy = sample
        for i, (nx, ny, _) in enumerate(nodes):
            d = (nx - sx) ** 2 + (ny - sy) ** 2
            if d < best:
                best, best_idx = d, i
        return best_idx
        # raise NotImplementedError("TODO: implement nearest_node()")

    @staticmethod
    def steer(start: Point, target: Point, step_size: float) -> Point:
        """
        TODO: move from start toward target by at most step_size meters.

        Return target directly when it is closer than step_size. Otherwise return
        the point on the line segment start->target that is exactly step_size
        from start.
        """
        dx = target[0] - start[0]
        dy = target[1] - start[1]
        dist = math.hypot(dx, dy)

        # if already within one step, just return the target directly
        if dist <= step_size:
            return target

        # otherwise move step_size meters along the start->target direction
        ratio = step_size / dist
        return (start[0] + dx * ratio, start[1] + dy * ratio)
        # raise NotImplementedError("TODO: implement steer()")

    @staticmethod
    def reconstruct_rrt(nodes: Sequence[Tuple[float, float, int]], index: int) -> List[Point]:
        """
        TODO: reconstruct an RRT path from a goal node index.

        Follow parent indices until -1, collect (x, y) points, and reverse the
        list so it runs from start to goal.
        """
        path = []
        current = index

        while current != -1: # backwards to find parent
            nx, ny, parent = nodes[current]
            path.append((nx, ny))
            current = parent
        path.reverse()
        return path
        # raise NotImplementedError("TODO: implement reconstruct_rrt()")

    def plan_apf(self, start: Point, goal: Point) -> List[Point]:
        """
        TODO: implement Artificial Potential Field planning.

        Expected behavior:
        1. Treat the current point q and the goal as numpy arrays.
        2. Begin the returned path with start.
        3. Repeat up to self.apf_max_steps:
           - If q is within self.goal_tolerance of the goal, append the exact
             goal point and return self.smooth_path(path).
           - Compute an attractive force toward the goal. A standard first try is
             self.apf_attractive_gain * (goal - q).
           - Add a repulsive force from every known obstacle. You may want to
             implement self.repulsive_force(q, obstacle) for the rectangle-based
             repulsive term.
           - Add self.boundary_repulsive_force(q) so the vehicle does not drift
             into the world boundary.
           - Normalize the total force and move by self.apf_step_size.
           - If the candidate point is occupied, try another nearby direction.
             You may want to implement self.find_free_apf_step(q, force) for
             this recovery behavior.
           - Append waypoints only when the robot has moved enough to avoid
             publishing hundreds of nearly identical poses.
        4. Return [] if the method gets stuck or reaches the iteration limit.

        APF can get trapped in local minima. It is acceptable to add a small
        randomized nudge when the force norm is almost zero, as long as the
        resulting point is still collision-free.
        """
        q = np.array(start, dtype=float)
        goal_arr = np.array(goal, dtype=float)
        path = [start]
        min_dist_to_goal = float("inf")
        stuck_count = 0

        for _ in range(self.apf_max_steps):
            if np.linalg.norm(q - goal_arr) <= self.goal_tolerance:
                path.append(goal)
                return self.smooth_path(path)

            attractive = self.apf_attractive_gain * (goal_arr - q)  # to pull the robot towards the wall

            repulsive = sum(    # repulsive to pull the robot away from every obs/boundary
                (self.repulsive_force(q, obs) for obs in self.obstacles),
                np.zeros(2)
            )
            repulsive += self.boundary_repulsive_force(q)

            force = attractive + repulsive
            norm = np.linalg.norm(force)

            # if force ~0 => a local minimum
            if norm < 1e-3:
                force = self.rng.uniform(-1, 1, size=2)
                norm = np.linalg.norm(force)

            force /= norm
            candidate = q + force * self.apf_step_size

            if self.point_is_free((candidate[0], candidate[1])):
                q = candidate
            else:
                # if direct step blocked => try rotated direction
                recovered = self.find_free_apf_step(q, force)
                if recovered is None:
                    break
                q = recovered

            if len(path) == 0 or self.distance(path[-1], (q[0], q[1])) > self.map_resolution:
                path.append((float(q[0]), float(q[1])))

            # detects if stuck => no meaningful progress toward goal
            dist = np.linalg.norm(q - goal_arr)
            if dist < min_dist_to_goal - 0.01:
                min_dist_to_goal = dist
                stuck_count = 0
            else:
                stuck_count += 1
                if stuck_count > 300:
                    self.get_logger().warning("APF: stuck in local minimum.")
                    break

        # self.warn_unimplemented("plan_apf")
        return []

    def repulsive_force(self, q: np.ndarray, obstacle: ObstacleBox) -> np.ndarray:
        """
        TODO: compute a repulsive APF force from one rectangular obstacle.

        A useful approach:
        - Inflate the obstacle by robot radius plus obstacle inflation.
        - Find the closest point on that inflated rectangle to q.
        - Push away from that closest point only when the distance is within
          self.apf_influence_distance.
        - Return a numpy vector with shape (2,).
        """
        inflation = self.robot_radius + self.obstacle_inflation
        xmin, xmax, ymin, ymax = self.inflated_bounds(obstacle, inflation)

        # finds the closest point on the inflated rectangle to q
        closest = np.array([
            np.clip(q[0], xmin, xmax),
            np.clip(q[1], ymin, ymax),
        ])

        diff = q - closest
        dist = np.linalg.norm(diff)

        if dist >= self.apf_influence_distance or dist < 1e-6:
            return np.zeros(2)

        # magnitude grows as robot gets closer
        magnitude = self.apf_repulsive_gain * (1.0/dist - 1.0/self.apf_influence_distance) / (dist**2)
        return magnitude * diff / dist
        # raise NotImplementedError("TODO: implement repulsive_force()")

    def boundary_repulsive_force(self, q: np.ndarray) -> np.ndarray:
        """
        TODO: compute an APF repulsive force from the world boundary.

        The map is a square centered at the origin. Push q away from an edge when
        it gets close to the usable limit.
        """
        force = np.zeros(2)
        limit = self.map_limit

        # if too close to the walls => push away
        for axis in range(2):
            dist_min = q[axis] - (-limit)  # distance to lower wall
            dist_max = limit - q[axis]     # distance to upper wall
            if dist_min < self.apf_influence_distance and dist_min > 1e-6:
                force[axis] += self.apf_repulsive_gain * (1.0/dist_min - 1.0/self.apf_influence_distance) / (dist_min**2)
            if dist_max < self.apf_influence_distance and dist_max > 1e-6:
                force[axis] -= self.apf_repulsive_gain * (1.0/dist_max - 1.0/self.apf_influence_distance) / (dist_max**2)

        return force
        # raise NotImplementedError("TODO: implement boundary_repulsive_force()")

    def find_free_apf_step(self, q: np.ndarray, force: np.ndarray) -> Optional[np.ndarray]:
        """
        TODO: find a nearby collision-free APF step when the direct step fails.

        One strategy is to rotate the force direction through several positive
        and negative angle offsets, test each candidate with point_is_free(), and
        return the first valid candidate. Return None if no valid step is found.
        """
        for angle in [0.3, -0.3, 0.6, -0.6, 0.9, -0.9, 1.2, -1.2]:
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            rotated = np.array([
                cos_a * force[0] - sin_a * force[1],
                sin_a * force[0] + cos_a * force[1],
            ])
            rotated /= np.linalg.norm(rotated)
            candidate = q + rotated * self.apf_step_size
            if self.point_is_free((candidate[0], candidate[1])):
                return candidate
        return None
        # raise NotImplementedError("TODO: implement find_free_apf_step()")

    def inflated_bounds(
        self, obstacle: ObstacleBox, inflation: float
    ) -> Tuple[float, float, float, float]:
        """
        TODO: return obstacle bounds expanded by inflation on every side.

        ObstacleBox.bounds returns (xmin, xmax, ymin, ymax). This helper should
        subtract inflation from the lower bounds and add it to the upper bounds.
        """
        xmin, xmax, ymin, ymax = obstacle.bounds
        return (xmin - inflation, xmax + inflation, ymin - inflation, ymax + inflation)
        # raise NotImplementedError("TODO: implement inflated_bounds()")

    def point_is_free(self, point: Point) -> bool:
        """
        TODO: return True when a world-frame point is safe for the robot center.

        Check both:
        - the map boundary, using self.map_limit and self.robot_radius;
        - every obstacle rectangle, inflated by robot radius plus
          self.obstacle_inflation.
        """
        x, y = point
        inflation = self.robot_radius + self.obstacle_inflation # total inflation

        limit = self.map_limit - self.robot_radius  # to reject points too close to the world boundaries
        if not (-limit <= x <= limit and -limit <= y <= limit):
            return False

        for obs in self.obstacles:
            xmin, xmax, ymin, ymax = self.inflated_bounds(obs, inflation)   # to reject points that fall inside any inflated obstacle area
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return False

        return True
        # raise NotImplementedError("TODO: implement point_is_free()")

    def segment_is_free(self, a: Point, b: Point) -> bool:
        """
        TODO: return True when the straight segment from a to b is collision-free.

        A practical implementation samples points along the segment at a spacing
        related to self.map_resolution and checks each sample with point_is_free().
        """
        step = self.map_resolution * 0.5    # half-cell spacing for safety
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        length = math.hypot(dx, dy)

        if length < 1e-9:
            return self.point_is_free(a)

        n_steps = int(length / step)     # walks from a toward b in fixed steps and checks the samples
        for i in range(n_steps + 1):
            t = min(i * step / length, 1.0)
            sample = (a[0] + t * dx, a[1] + t * dy)
            if not self.point_is_free(sample):
                return False

        return self.point_is_free(b)    # checks last one
        # raise NotImplementedError("TODO: implement segment_is_free()")

    def smooth_path(self, path: List[Point]) -> List[Point]:
        """
        TODO: optionally remove unnecessary waypoints from a raw path.

        One approach is shortcut smoothing: from the current anchor waypoint, try
        to connect directly to the farthest later waypoint with segment_is_free().
        After shortcutting, you may densify the path so the controller receives
        reasonably spaced targets.
        """
        if len(path) < 3:
            return path

        # shortcut smoothing
        smoothed = [path[0]]
        anchor = 0

        while anchor < len(path) - 1:
            for far in range(len(path) - 1, anchor, -1):
                if self.segment_is_free(path[anchor], path[far]):
                    smoothed.append(path[far])
                    anchor = far
                    break
            # no shortcut found
            else:
                anchor += 1
                smoothed.append(path[anchor])

        return self.densify_path(smoothed, self.map_resolution * 2.0)
        # raise NotImplementedError("TODO: implement smooth_path()")

    @staticmethod
    def densify_path(path: List[Point], spacing: float) -> List[Point]:
        """
        TODO: insert intermediate points so consecutive waypoints are not too far
        apart for the tracker.

        This is optional but useful after smoothing. Preserve the first and last
        points exactly.
        """
        if len(path) < 2:
            return path

        dense = [path[0]]

        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            seg_len = math.hypot(dx, dy)

            n = int(seg_len / spacing)  # to see how many intermediate points fit to the segment

            # evenly spaced points between a and b
            for k in range(1, n + 1):
                t = k * spacing / seg_len
                dense.append((a[0] + t * dx, a[1] + t * dy))

            dense.append(b)

        return dense
        # raise NotImplementedError("TODO: implement densify_path()")

    def advance_waypoint(self, robot: Point) -> None:
        while self.waypoint_index < len(self.path) - 1:
            if self.distance(robot, self.path[self.waypoint_index]) > self.waypoint_tolerance:
                break
            self.waypoint_index += 1

    def lookahead_target(self, robot: Point) -> Point:
        idx = self.waypoint_index
        while idx < len(self.path) - 1:
            if self.distance(robot, self.path[idx]) >= self.lookahead_distance:
                break
            idx += 1
        return self.path[idx]

    def compute_tracking_command(self, robot: Point, yaw: float, target: Point) -> Twist:
        dx = target[0] - robot[0]
        dy = target[1] - robot[1]
        target_dist = math.hypot(dx, dy)
        target_heading = math.atan2(dy, dx)
        heading_error = self.wrap_angle(target_heading - yaw)

        speed = min(self.max_linear_speed, self.linear_gain * target_dist)
        if abs(heading_error) > self.heading_slow_angle:
            speed *= max(0.0, 1.0 - abs(heading_error) / math.pi)
        if abs(heading_error) > 1.35:
            speed = 0.0

        cmd = Twist()
        cmd.linear.x = float(np.clip(speed, 0.0, self.max_linear_speed))
        cmd.angular.z = float(
            np.clip(self.angular_gain * heading_error, -self.max_angular_speed, self.max_angular_speed)
        )
        return cmd

    def apply_lidar_avoidance(self, cmd: Twist) -> None:
        """
        TODO: modify cmd in-place using the latest LaserScan.

        This function is called after the waypoint tracker has produced a nominal
        velocity command. Your job is to make a local safety adjustment without
        replacing the global planner.

        Expected behavior:
        1. If self.scan_ranges or self.scan_angles is None, return without
           changing cmd.
        2. Build boolean masks for three sectors in the robot/base_scan frame:
           - front: abs(angle) <= self.front_sector
           - left: self.side_sector_min <= angle <= self.side_sector_max
           - right: -self.side_sector_max <= angle <= -self.side_sector_min
        3. Use self.masked_min_range() to compute the closest obstacle in each
           sector.
        4. If the front distance is below self.front_slow_distance, scale down
           cmd.linear.x smoothly.
        5. If the front distance is below self.front_stop_distance, set
           cmd.linear.x to zero and turn toward the side with more free space.
        6. If an obstacle is closer on one side than the other, add a small
           angular bias away from the closer side.
        7. Clip the final command to [0, self.max_linear_speed] for linear.x and
           [-self.max_angular_speed, self.max_angular_speed] for angular.z.

        Remember: mutate the incoming cmd object. Do not publish from here; the
        caller publishes after this method returns.
        """
        if self.scan_ranges is None or self.scan_angles is None:
            return

        angles = self.scan_angles
        ranges = self.scan_ranges

        # builds masks for the 3 sectors we care about
        front_mask = np.abs(angles) <= self.front_sector
        left_mask  = (angles >= self.side_sector_min) & (angles <= self.side_sector_max)
        right_mask = (angles >= -self.side_sector_max) & (angles <= -self.side_sector_min)

        # gets the closest obstacle distance in each sector
        front_dist = self.masked_min_range(ranges, front_mask)
        left_dist  = self.masked_min_range(ranges, left_mask)
        right_dist = self.masked_min_range(ranges, right_mask)

        if front_dist < self.front_slow_distance:
            scale = (front_dist - self.front_stop_distance) / (self.front_slow_distance - self.front_stop_distance)
            cmd.linear.x *= float(np.clip(scale, 0.0, 1.0))

        # too close => stop completely and turn toward the side with more free space
        if front_dist < self.front_stop_distance:
            cmd.linear.x = 0.0
            if left_dist >= right_dist:
                cmd.angular.z =  self.max_angular_speed
            else:
                cmd.angular.z = -self.max_angular_speed

        if left_dist < right_dist:
            cmd.angular.z -= self.avoidance_gain
        elif right_dist < left_dist:
            cmd.angular.z += self.avoidance_gain

        # clips final command to safe limits
        cmd.linear.x  = float(np.clip(cmd.linear.x,  0.0, self.max_linear_speed))
        cmd.angular.z = float(np.clip(cmd.angular.z, -self.max_angular_speed, self.max_angular_speed))
        # self.warn_unimplemented("apply_lidar_avoidance")
        return

    @staticmethod
    def masked_min_range(ranges: np.ndarray, mask: np.ndarray) -> float:
        values = ranges[mask]
        if values.size == 0:
            return float("inf")
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return float("inf")
        return float(np.min(finite))

    def publish_path(self, path: Sequence[Point]) -> None:
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        for x, y in path:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.path_pub.publish(msg)

    def publish_goal(self) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.pose.position.x = self.goal[0]
        msg.pose.position.y = self.goal[1]
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)

    def publish_stop(self) -> None:
        self.cmd_pub.publish(Twist())


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AutonomousPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
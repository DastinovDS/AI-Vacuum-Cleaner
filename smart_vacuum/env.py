import math
import random

import numpy as np


class SmartVacuumEnv:
    """
    A grid-world simulation for a Reinforcement Learning vacuum agent.

    The environment simulates a robot with limited battery life navigating a
    grid to clean randomly distributed dirt. It uses complex reward shaping
    to encourage cleaning, discourage hitting walls, and ensure the agent
    returns to its base station (0,0) before its battery expires.

    :ivar grid_size: Dimensions of the square grid.
    :ivar dirt_density: Probability of a tile containing dirt (0.0 to 1.0).
    :ivar move_cost: Energy cost of moving.
    :ivar clean_cost: Energy cost of cleaning.
    :ivar max_battery: Starting energy capacity/ maximal battery capacity.
    :ivar base_pos: Coordinates of the charging station (0, 0).
    """
    def __init__(
        self,
        grid_size=10,
        dirt_density=0.25,
        move_cost=2.0,
        clean_cost=0.5,
        seed=None,
    ):
        self.grid_size = grid_size
        self.dirt_density = dirt_density
        self.move_cost = move_cost
        self.clean_cost = clean_cost

        self.base_pos = (0, 0)
        self.rng = random.Random(seed)

        self.max_battery = 100.0

        # Reward parameters
        self.reward_death = -1000.0
        self.reward_clean_success = +100.0
        self.reward_clean_empty = -10.0
        # Idle movement penalty should be a small nudge, not overwhelming
        self.reward_idle_move = -0.2
        self.reward_hit_wall = -50.0
        # Docking should mainly be about survival, not farming reward
        self.reward_dock_base = 0.0
        self.reward_dock_scale = 0.0  # scaling factor for energy gained
        self.reward_stay_full_on_base = -10.0

        # Extra bonus for cleaning tiles far from the base
        self.far_clean_distance_threshold = 6  # Manhattan distance from base
        self.reward_clean_far_bonus = +25.0  # extra bonus for far tiles

        # State-related thresholds
        self.near_distance_threshold = 4  # Manhattan distance <= 4 => "near"

        # Internal state
        self.agent_pos = None
        self.battery = None
        self.dirt_grid = None
        self.total_dirt_initial = None

        # For visualization
        self.step_count = 0

        # For docking logic
        self.steps_since_last_dock = 0
        self.cleaned_since_last_dock = 0

        # Anti-loop / anti-stalling shaping (prevents ABAB ping-pong and
        # "hang around forever")
        self.recent_positions = []
        self.recent_positions_maxlen = 4  # detect ABAB loops
        # Small penalty just to break ties; too large will dominate learning
        self.loop_abab_penalty = -25.0
        self.steps_since_last_successful_clean = 0
        self.no_clean_penalty_start = 40  # steps without successful cleaning
        self.no_clean_step_penalty = -2.0  # applied per-step after
        # start while dirt remains
        # Penalty for walking away from a dirty tile without cleaning it
        self.leave_dirty_penalty = 10.0

    def reset(self):
        """
        Resets the environment to an initial state for a new episode.

        This involves:
        1. Refilling the battery to maximum capacity.
        2. Regenerating the dirt distribution based on dirt_density.
        3. Placing the agent at a random location on the grid.
        4. Resetting all performance and anti-loop counters.

        :return: int - The initial encoded discrete state [0..255].
        """
        self.battery = self.max_battery
        self.step_count = 0
        self.steps_since_last_dock = 0
        self.cleaned_since_last_dock = 0
        self.recent_positions = []
        self.steps_since_last_successful_clean = 0

        # Create dirt grid
        self.dirt_grid = np.zeros((self.grid_size, self.grid_size),
                                  dtype=np.int8)
        num_tiles = self.grid_size * self.grid_size
        num_dirt = int(num_tiles * self.dirt_density)

        # Choose random dirt positions (can include base; that's fine)
        all_positions = [(x, y) for x in range(self.grid_size)
                         for y in range(self.grid_size)]
        dirt_positions = self.rng.sample(all_positions, num_dirt)
        for (x, y) in dirt_positions:
            self.dirt_grid[y, x] = 1
        self.total_dirt_initial = int(np.sum(self.dirt_grid))

        # Random start position (anywhere)
        self.agent_pos = self.rng.choice(all_positions)
        self.recent_positions.append(self.agent_pos)

        return self._get_state()

    def step(self, action):
        """
        Processes the agent's action and updates the environment's state.

        This is the core logic engine. It calculates movement,
        wall collisions,
        cleaning success, and battery consumption. It also computes a
        multi-faceted reward including:
        - Survival bonuses for docking with low battery.
        - Penalties for 'ABAB' loops and hitting walls.
        - Progress rewards for reducing the total dirt count.
        - Efficiency penalties (cost of movement/cleaning).

        :param action: The action index (0:N, 1:S, 2:W, 3:E, 4:Clean).

        :return: tuple - (next_state: int, reward: float,
        done: bool, info: dict)
        """
        self.step_count += 1
        self.steps_since_last_dock += 1
        self.steps_since_last_successful_clean += 1

        x, y = self.agent_pos
        reward = 0.0
        done = False

        # Track global dirt count and whether we
        # are currently on dirt before the step
        prev_remaining = int(np.sum(self.dirt_grid))
        was_dirt_here = self.dirt_grid[y, x] == 1

        # Apply action
        if action in (0, 1, 2, 3):  # movement
            dx, dy = 0, 0
            if action == 0:  # North
                dy = -1
            elif action == 1:  # South
                dy = +1
            elif action == 2:  # West
                dx = -1
            elif action == 3:  # East
                dx = +1

            new_x = x + dx
            new_y = y + dy

            # Movement energy cost always applied
            self._apply_energy_cost(self.move_cost)

            # Wall check
            if not (0 <= new_x < self.grid_size and
                    0 <= new_y < self.grid_size):
                # Hit wall -> stay in place + penalty
                reward += self.reward_hit_wall
                # Also treat as idle movement penalty
                reward += self.reward_idle_move
            else:
                # Valid move
                self.agent_pos = (new_x, new_y)
                # Idle movement penalty
                reward += self.reward_idle_move

                # If we left a dirty tile without cleaning it,
                # discourage that behavior
                if was_dirt_here:
                    reward -= self.leave_dirty_penalty

        elif action == 4:  # Clean
            # Clean cost
            self._apply_energy_cost(self.clean_cost)
            if self.dirt_grid[y, x] == 1:
                # Successful cleaning
                self.dirt_grid[y, x] = 0
                reward += self.reward_clean_success
                self.cleaned_since_last_dock += 1
                self.steps_since_last_successful_clean = 0

                # Extra bonus if this tile is far from the
                # base (encourage exploring)
                bx, by = self.base_pos
                dist_from_base = abs(x - bx) + abs(y - by)
                if dist_from_base >= self.far_clean_distance_threshold:
                    reward += self.reward_clean_far_bonus
            else:
                # Cleaning empty tile
                reward += self.reward_clean_empty
        else:
            raise ValueError(f"Invalid action: {action}")

        # Track recent positions and penalize ABAB ping-pong loops
        # (including base in/out)
        self.recent_positions.append(self.agent_pos)
        if len(self.recent_positions) > self.recent_positions_maxlen:
            self.recent_positions.pop(0)
        if len(self.recent_positions) == 4:
            a, b, c, d = self.recent_positions
            if a == c and b == d and a != b:
                reward += self.loop_abab_penalty

        # Handle docking/base rewards + recharge
        prev_battery = self.battery
        if self.agent_pos == self.base_pos:
            if self.battery < self.max_battery:
                battery_pct_before = (prev_battery / self.max_battery) * 100.0

                # Recharge
                self.battery = self.max_battery

                # Give a strong "mission accomplished" / survival bonus
                if (battery_pct_before < 30.0 and
                        self.cleaned_since_last_dock > 0):
                    survival_bonus = (400.0 +
                                      (self.cleaned_since_last_dock * 20.0))
                    reward += survival_bonus
                else:
                    # Discourage abusive docking loops without doing work
                    reward += -2.0

                # Reset docking-related counters
                self.steps_since_last_dock = 0
                self.cleaned_since_last_dock = 0
            else:
                # fully charged and staying on base
                reward += self.reward_stay_full_on_base

        # Check if battery dead
        if self.battery <= 0.0:
            # Clamp
            self.battery = 0.0
            reward += self.reward_death
            done = True

        # Global progress-based reward and episode termination
        new_remaining = int(np.sum(self.dirt_grid))
        cleaned_this_step = prev_remaining - new_remaining
        if cleaned_this_step > 0:
            # Extra reward for reducing total dirt in the environment
            reward += 50.0 * cleaned_this_step

        if new_remaining > 0:
            # Small time pressure while dirt remains
            # (kept small to avoid swamping learning)
            reward -= 0.15
            # If we've been wandering too long without cleaning,
            # apply stronger pressure
            if (self.steps_since_last_successful_clean >=
                    self.no_clean_penalty_start):
                reward += self.no_clean_step_penalty

        # Check if all dirt cleaned
        if new_remaining == 0:
            done = True

        # Optionally limit episode length to avoid infinite wandering
        if self.step_count >= 1000:
            done = True

        # Additional terminal shaping
        if done:
            if new_remaining == 0:
                reward += 500.0
            elif self.battery == 0.0:
                reward -= 200.0

        next_state = self._get_state()
        info = {
            "position": self.agent_pos,
            "battery": self.battery,
            "remaining_dirt": int(np.sum(self.dirt_grid)),
        }
        return next_state, reward, done, info

    def _get_state(self):
        """
        Compresses the high-dimensional grid state into a
        single integer [0..255].

        The encoding uses a hierarchical bit-packing
        (or index-offset) approach
        to ensure the Q-agent can generalize its knowledge.
        The components are:
        1. Direction Sector (8): Compass direction to base (if low battery)
           or nearest dirt (if high battery).
        2. Distance to Base (3): Far, Mid, or Near bins.
        3. Battery Level (4): Critical, Low, Mid, or High bins.
        4. Dirt Under Feet (2): Binary flag.
        5. Proximity Dirt (2): Binary flag if dirt is in a
        3x3 surrounding area.

        :return: int - A discrete state index between 0 and 255.
        """

        x, y = self.agent_pos

        # Direction to nearest dirt (if any).
        # If no dirt exists, default offset 0.
        nearest_dx, nearest_dy = 0, 0
        found_dirt = False
        min_dist = None
        for yy in range(self.grid_size):
            for xx in range(self.grid_size):
                if self.dirt_grid[yy, xx] == 1:
                    dist = abs(xx - x) + abs(yy - y)
                    if not found_dirt or dist < min_dist:
                        found_dirt = True
                        min_dist = dist
                        nearest_dx = xx - x
                        nearest_dy = yy - y

        # Battery level bins and target focus (dirt vs base)
        battery_pct = (self.battery / self.max_battery) * 100.0
        # 4 battery bins: 0: Critical (<15), 1: Low (15-40),
        # 2: Mid (40-75), 3: High (>75)
        if battery_pct < 15.0:
            battery_bin = 0
        elif battery_pct < 40.0:
            battery_bin = 1
        elif battery_pct < 75.0:
            battery_bin = 2
        else:
            battery_bin = 3

        # Target direction: focus on base when battery is low/critical,
        # dirt otherwise
        if battery_pct < 30.0:
            target_dx, target_dy = -x, -y  # base at (0,0)
        else:
            target_dx, target_dy = nearest_dx, nearest_dy

        if target_dx != 0 or target_dy != 0:
            angle = math.atan2(target_dy, target_dx)
            sector = int(((angle + math.pi) / (2 * math.pi)) * 8) % 8
        else:
            sector = 0

        # Distance to base: 3 bins: 0: Far (>8), 1: Mid (4-8), 2: Near (<4)
        manhattan = abs(x) + abs(y)
        if manhattan > 8:
            distance_bin = 0
        elif manhattan >= 4:
            distance_bin = 1
        else:
            distance_bin = 2

        # Dirt under feet
        dirt_here = 1 if self.dirt_grid[y, x] == 1 else 0

        # Proximity dirt in 3x3 around agent (including current tile)
        proximity = 0
        for yy in range(max(0, y - 1), min(self.grid_size, y + 2)):
            for xx in range(max(0, x - 1), min(self.grid_size, x + 2)):
                if self.dirt_grid[yy, xx] == 1:
                    proximity = 1
                    break
            if proximity == 1:
                break

        # Encode into single integer
        # order: sector (8) * distance (3) * battery (4) *
        # dirt_here (2) * proximity (2)
        idx = sector
        idx = idx * 3 + distance_bin
        idx = idx * 4 + battery_bin
        idx = idx * 2 + dirt_here
        idx = idx * 2 + proximity

        # Wrap into [0,255] for the fixed 256-state Q-table
        return idx % 256

    def _apply_energy_cost(self, cost):
        """
        Reduces the agent's battery level by a specific cost.

        :param cost: The amount of battery to consume.
        """
        self.battery -= cost

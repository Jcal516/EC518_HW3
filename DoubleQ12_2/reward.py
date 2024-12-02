def reward_function(collision, speed, lane_invasion, current_position, next_waypoint,previous_distance, target_speed=15.0):
    import numpy as np
    """
    Calculate the reward for the agent based on its actions.

    Parameters:
    - collision (bool): Whether a collision occurred.
    - speed (float): The current speed of the vehicle.
    - lane_invasion (bool): Whether the vehicle has invaded a lane.
    - goal_reward
        - current_position (tuple): The current (x, y) position of the vehicle.
        - next_waypoint (tuple): The (x, y) position of the next waypoint.
    - target_speed (float): The target speed for the vehicle.

    Returns:
    - float: The calculated reward.
    """

    # Penalty for collisions
    if collision:
        return -100.0

    # Penalty for lane invasions
    if lane_invasion:
        return -10.0

    # Reward for maintaining speed
    speed_reward = 1.0 - abs(speed - target_speed) / target_speed
    speed_reward = max(speed_reward, 0)  # Ensure non-negative reward

    # Calculate Euclidean distance to next waypoint
    distance_to_waypoint = np.linalg.norm(np.array(current_position) - np.array(next_waypoint))

    # Positive reward for getting closer to the next waypoint
    goal_reward = max(0, 1 - distance_to_waypoint)  # Reward increases as distance decreases

    # Negative reward for moving away from the next waypoint
    if distance_to_waypoint > previous_distance:  # We are assuming you keep track of the previous distance
        distance_penalty = -0.5  # You can adjust this value
    else:
        distance_penalty = 0

    # Combine rewards
    total_reward = (speed_reward * 10.0) + goal_reward + distance_penalty

    return total_reward


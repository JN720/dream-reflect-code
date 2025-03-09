You are an AI agent tasked with generating a policy in the form of executable Python code for controlling a lunar lander in the OpenAI Gym LunarLander-v2 environment.

Your goal is to generate a function named lunar_lander_policy(observation), which takes an environment observation (a NumPy array of 8 state variables) as input and returns an integer action (0, 1, 2, or 3) corresponding to:

0: Do nothing
1: Fire left engine
2: Fire main engine
3: Fire right engine
Requirements:

Use a simple, interpretable heuristic-based approach to make landing decisions. You may use conditions on altitude, velocity, angle, and position to determine actions.
Ensure that the generated policy maximizes landing success by reducing velocity before touchdown and keeping the lander upright.
The function must be self-contained, using only standard Python libraries and NumPy.
The code should include brief inline comments explaining the logic behind decisions.
Example Input:

```python
observation = np.array([0.1, -0.2, 0.03, 0.1, 0.02, -0.1, 0.0, 0.0])
action = lunar_lander_policy(observation)
print(action) # Expected output: An integer from {0, 1, 2, 3}
```

Example Output Format:

```python
import numpy as np

def lunar_lander_policy(observation):
"""
Heuristic-based policy for controlling the lunar lander.
observation: [x_pos, y_pos, x_vel, y_vel, angle, angular_vel, left_leg_contact, right_leg_contact]
Returns: action (0, 1, 2, or 3)
"""
x_pos, y_pos, x_vel, y_vel, angle, angular_vel, left_leg, right_leg = observation

    # Example heuristic: Fire main engine if falling too fast
    if y_vel < -0.1:
        return 2  # Fire main engine to slow descent
    elif angle > 0.1:
        return 1  # Fire left engine to stabilize
    elif angle < -0.1:
        return 3  # Fire right engine to stabilize
    else:
        return 0  # Do nothing

    return 0  # Default action

# Example usage:
observation = np.array([0.1, -0.2, 0.03, 0.1, 0.02, -0.1, 0.0, 0.0])
action = lunar_lander_policy(observation)
print(action) # Output should be an action from {0, 1, 2, 3}
```

Instructions:

You are given the current state as indicated.

Generate a function using either heuristics or learned policy that successfully lands the lunar lander.
Ensure the function is deterministic and efficient, considering real-time decision-making constraints.
Return only the function and any necessary imports, without additional explanations.

You may have been given feedback alongside previous code. If so, refine the code to incorporate the feedback.

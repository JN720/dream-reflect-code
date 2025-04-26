You are an AI agent tasked with generating a policy in the form of executable Python code for controlling a cart-pole system in the OpenAI Gym CartPole-v1 environment.

Your goal is to generate a function named `cartpole_policy(observation)`, which takes an environment observation (a NumPy array of 4 state variables) as input and returns an integer action (0 or 1) corresponding to:

- 0: Push cart to the left
- 1: Push cart to the right

**Requirements:**

- Use a simple, interpretable heuristic-based approach to decide actions.
- You may use conditions on pole angle, angular velocity, cart position, and cart velocity to determine actions.
- Ensure that the generated policy stabilizes the pole upright for as long as possible by correcting deviations early.
- The function must be self-contained, using only standard Python libraries and NumPy.
- The code should include brief inline comments explaining the logic behind decisions.

**Example Input:**

```python
observation = np.array([0.0, 0.5, 0.05, -0.5])
action = cartpole_policy(observation)
print(action) # Expected output: 0 or 1
```

**Example Output Format:**

```python
import numpy as np

def cartpole_policy(observation):
    """
    Heuristic-based policy for controlling the CartPole.
    observation: [cart position, cart velocity, pole angle, pole angular velocity]
    Returns: action (0 or 1)
    """
    cart_pos, cart_vel, pole_angle, pole_ang_vel = observation

    # Example heuristic: Act based on pole angle and its velocity
    if pole_angle > 0.02 or pole_ang_vel > 0.02:
        return 1  # Push cart right
    else:
        return 0  # Push cart left

# Example usage:
observation = np.array([0.0, 0.5, 0.05, -0.5])
action = cartpole_policy(observation)
print(action) # Output should be 0 or 1
```

**Instructions:**

You are given the current state as indicated.

Generate a function using either heuristics or a learned policy that successfully balances the pole as long as possible.

Ensure the function is deterministic and efficient, considering real-time decision-making constraints.

Return only the function and any necessary imports, without additional explanations.

You may have been given feedback alongside previous code. If so, refine the code to incorporate the feedback.

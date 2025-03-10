import gymnasium as gym
import torch
from google import genai
import os
import numpy as np
import json
from dotenv import load_dotenv

from world_models import LunarLanderWorldModel

load_dotenv()

ai_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

world_model = LunarLanderWorldModel()
world_model.load_state_dict(torch.load("lunar_lander_world_model/best_model_1.pth"))

env = gym.make("LunarLander-v3", render_mode="human")

CODE_GENERATION_PROMPT = open(os.path.join('prompts', 'lunar_lander_code.md')).read()
CODE_CRITIC_PROMPT = open(os.path.join('prompts', 'lunar_lander_critic.md')).read()

lunar_lander_policy = None

def parse_code(code: str):
    code = code.strip()
    start = code.index("```python") + 9
    end = code.index("```", start + 9)
    if start == -1 or end == -1:
        return code
    return code[start:end]

def parse_json(json_str: str):
    text = json_str.strip()
    start = text.index("```json") + 7
    end = text.index("```", start + 7)
    if start == -1 or end == -1:
        return text
    return text[start:end]

def call_model(*contents):
    ai_response = ai_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents
    )
    return ai_response.text

def get_policy(observation, world_steps = 10, max_reflections = 3, manually_check_code = True):
    namespace = {}
    observation_str = str(observation)

    # Generate initial code
    ai_response = call_model(CODE_GENERATION_PROMPT, observation_str)
    code = parse_code(ai_response)
    print(code)
    if manually_check_code:
        print("Please check the code and write y to continue.")
        response = input()
        if response == "y":
            print("Continuing...")
        else:
            print("Stopping...")
            return lunar_lander_policy
    # Declare function
    compiled_code = compile(code, "<string>", "exec")
    exec(compiled_code, namespace)
    assert "lunar_lander_policy" in namespace, "Policy function not defined."
    lunar_lander_policy = namespace["lunar_lander_policy"]
    for _ in range(max_reflections):
        # Simulate future steps
        future_states = []
        with torch.no_grad():
            state = observation
            for _ in range(world_steps):
                assert lunar_lander_policy is not None, "Policy function not defined."
                if len(state) == 2:
                    state = state[0]
                state = torch.tensor(state, dtype = torch.float32)
                action = lunar_lander_policy(state)
                action_array = np.zeros(4, dtype = np.float32)
                action_array[action] = 1
                world_model_input = torch.tensor(np.array([*state, *action_array], dtype = np.float32))
                state = world_model(world_model_input)
                future_states.append(state)
        # Critique code
        predict_dict = {'code': code, 'future_states': str(future_states)}
        json_str = json.dumps(predict_dict)
        feedback = call_model(CODE_CRITIC_PROMPT, json_str)
        print(feedback)
        feedback = parse_json(feedback)
        feedback_json = json.loads(feedback)
        stop = feedback_json.get('stop', False)
        if stop:
            return lunar_lander_policy

        ai_response = call_model(CODE_GENERATION_PROMPT, feedback, code)
        code = parse_code(ai_response)

        print(code)
        if manually_check_code:
            print("Please check the feedback and write y to continue.")
            response = input()
            if response == "y":
                print("Continuing...")
            else:
                print("Stopping...")
                return lunar_lander_policy
        # Declare function
        compiled_code = compile(code, "<string>", "exec")
        namespace = {}
        exec(compiled_code, namespace)
        assert "lunar_lander_policy" in namespace, "Policy function not defined."
        lunar_lander_policy = namespace['lunar_lander_policy']
    assert lunar_lander_policy is not None, "Policy function not defined."
    return lunar_lander_policy

if __name__ == "__main__":
    TIMESTEPS_BETWEEN_GENERATION = 50
    rewards = []
    observation = env.reset()
    done = False
    while not done:
        cur_policy = get_policy(observation, max_reflections = 2, manually_check_code = True)
        for timestep in range(TIMESTEPS_BETWEEN_GENERATION):
            env.render()
            if len(observation) == 2:
                observation = observation[0]
            action = cur_policy(torch.tensor(observation, dtype = torch.float32))
            assert env.action_space.contains(action), "Invalid action."
            observation, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            if done or truncated:
                done = True
                break
    env.close()
    print("Rewards:\n", rewards)
    print("Total Timesteps:", len(rewards))
    print("Total Reward:", sum(rewards))
    print("Average Reward:", sum(rewards) / len(rewards))
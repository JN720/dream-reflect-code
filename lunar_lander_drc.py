import gymnasium as gym
import torch
import google.generativeai as genai
import os
import numpy as np
import json
from dotenv import load_dotenv

from world_models import LunarLanderWorldModel

load_dotenv()

client = genai.Client(api_key="GEMINI_API_KEY")

world_model = LunarLanderWorldModel()
world_model.load_state_dict(torch.load("lunar_lander_world_model/best_model_1.pth"))

env = gym.make("LunarLander-v3", render_mode="human")

CODE_GENERATION_PROMPT = open(os.path.join('prompts', 'lunar_lander_code.md')).read()
CODE_CRITIC_PROMPT = open(os.path.join('prompts', 'lunar_lander_critic.md')).read()

lunar_lander_policy = None

def get_policy(observation, world_steps = 3, max_reflections = 3, manually_check_code = True):
    observation_str = str(observation)
    # Generate initial code
    code = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[CODE_GENERATION_PROMPT, observation_str]
    )
    if manually_check_code:
        print(code)
        print("Please check the code and write y to continue.")
        response = input()
        if response == "y":
            print("Continuing...")
        else:
            print("Stopping...")
            return lunar_lander_policy
    # Declare function
    compiled_code = compile(code, "<string>", "exec")
    exec(compiled_code)
    for _ in range(max_reflections):
        # Simulate future steps
        future_states = []
        with torch.no_grad():
            state = observation
            for _ in range(world_steps):
                assert lunar_lander_policy is not None, "Policy function not defined."
                action = lunar_lander_policy(state)
                world_model_input = np.array([*state, action], dtype = np.float32)
                state = world_model(world_model_input)
                future_states.append(state)
        # Critique code
        predict_dict = {'code': code, 'future_states': future_states}
        json_str = json.dumps(predict_dict)
        feedback = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[CODE_CRITIC_PROMPT, json_str]
        )
        feedback_json = json.loads(feedback)
        if feedback_json['stop'].lower() == "true":
            return lunar_lander_policy
        code = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[CODE_GENERATION_PROMPT, feedback, code]
        )

        if manually_check_code:
            print(code)
            print("Please check the feedback and write y to continue.")
            response = input()
            if response == "y":
                print("Continuing...")
            else:
                print("Stopping...")
                return lunar_lander_policy
        # Declare function
        compiled_code = compile(code, "<string>", "exec")
        exec(compiled_code)
    return lunar_lander_policy

if __name__ == "__main__":
    TIMESTEPS_BETWEEN_GENERATION = 50
    rewards = []
    observation = env.reset()
    done = False
    while not done:
        cur_policy = get_policy(observation)
        for timestep in TIMESTEPS_BETWEEN_GENERATION:
            action = lunar_lander_policy(observation)
            assert env.action_space.contains(action), "Invalid action."
            observation, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            print(rewards)
            if done or truncated:
                done = True
                break
    env.close()
    print("Rewards:\n", rewards)
    print("Total Timesteps:", len(rewards))
    print("Total Reward:", sum(rewards))
    print("Average Reward:", sum(rewards) / len(rewards))
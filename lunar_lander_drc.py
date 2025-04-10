import gymnasium as gym
import torch
from models.groq import GroqModel
import os
import numpy as np
import json
import cv2
from dotenv import load_dotenv

from world_models import LunarLanderWorldModel
from utils import parse_code, parse_json

load_dotenv()

ai_client = GroqModel(model_name="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))

WRITE_VIDEO = True

# world_model = LunarLanderWorldModel()
# world_model.load_state_dict(torch.load("lunar_lander_world_model/best_model_1.pth"))

env = gym.make("LunarLander-v3", render_mode="rgb_array" if WRITE_VIDEO else "human")

CODE_GENERATION_PROMPT = open(os.path.join('prompts', 'lunar_lander_code.md')).read()
CODE_CRITIC_PROMPT = open(os.path.join('prompts', 'lunar_lander_critic.md')).read()

lunar_lander_policy = None

def call_model(*contents):
    return ai_client.invoke(contents)

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
    TIMESTEPS_BETWEEN_GENERATION = 70
    rewards = []
    observation = env.reset()
    done = False
    frames = []
    while not done:
        cur_policy = get_policy(observation, max_reflections = 0, manually_check_code = True)
        for timestep in range(TIMESTEPS_BETWEEN_GENERATION):
            frame = env.render()
            if WRITE_VIDEO:
                frames.append(frame)
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

    if WRITE_VIDEO:
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video = cv2.VideoWriter('lunar_lander_drc.avi', fourcc, 20.0, (width, height))

        for frame in frames:
            frame = np.array(frame, dtype = np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame)

        video.release()
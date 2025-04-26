from world_models import CartpoleWorldModel
from models.groq import GroqModel
import torch
import numpy as np
import gymnasium as gym
from drc import DreamReflectCode
import os
from dotenv import load_dotenv

load_dotenv()

CODE_GENERATION_PROMPT = open(os.path.join('prompts', 'cartpole_code.md')).read()
CODE_CRITIC_PROMPT = open(os.path.join('prompts', 'cartpole_critic.md')).read()

def postprocess_action(action):
    action_arr = np.zeros(2, dtype = np.float32)
    action_arr[action] = 1
    return action_arr

world_model = CartpoleWorldModel()
world_model.load_state_dict(torch.load("cartpole_world_model/best_model_1.pth"))

env = gym.make("CartPole-v1", render_mode="rgb_array")

llm = GroqModel(model_name="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))

drc_agent = DreamReflectCode(
    llm=llm,
    world_model=world_model,
    code_generation_prompt=CODE_GENERATION_PROMPT,
    code_critic_prompt=CODE_CRITIC_PROMPT,
    max_reflections=1,
    max_simulation_timesteps=200,
    manually_check_code=True,
    postprocess_action=postprocess_action,
    function_name = 'cartpole_policy',
)
drc_agent.infer(env, write_video=True)
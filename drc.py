import gymnasium as gym
import torch
from models.base_model import BaseModel
import numpy as np
import json
import cv2
from utils import parse_code, parse_json_with_fallback

class DreamReflectCode():
    def __init__(
            self, 
            llm: BaseModel, 
            world_model, 
            code_generation_prompt: str,
            code_critic_prompt: str,
            function_name: str,
            max_reflections=3, 
            max_simulation_timesteps=100, 
            manually_check_code=True, 
            postprocess_action=None,
    ):
        self.llm = llm
        self.world_model = world_model
        self.code_generation_prompt = code_generation_prompt
        self.code_critic_prompt = code_critic_prompt
        self.function_name = function_name
        self.max_reflections = max_reflections
        self.max_simulation_timesteps = max_simulation_timesteps
        if postprocess_action is None:
            self.postprocess_action = lambda x: x
        self.manually_check_code = manually_check_code
        self.postprocess_action = postprocess_action

        self.policy = None
        self.policy_code = ''
    
    def _call_model(self, *contents):
        for i, content in enumerate(contents):
            assert isinstance(content, str), "Content must be a string, found {}: {}\n{}".format(type(content), i, content)
        return self.llm.invoke(contents)
    
    def _simulate(self, observation: torch.tensor):
        assert self.policy is not None, "Policy function not defined."
        future_states = []
        with torch.no_grad():
            state = observation
            for _ in range(self.max_simulation_timesteps):
                # Remove info from state if it exists
                if len(state) == 2:
                    state = state[0]
                action = self.policy(state)
                action_array = self.postprocess_action(action)
                world_model_input = torch.tensor(np.array([*state, *action_array], dtype=np.float32))
                state = self.world_model(world_model_input)
                done = state[-1] > 0.5
                state = state[:-1]
                if done:
                    break
                future_states.append(state)
        return future_states
    
    def _generate_policy(self, *contents):
        ai_response = self._call_model(*contents)
        code = parse_code(ai_response)
        print(code)
        if self.manually_check_code:
            print("Please check the code and write y to continue.")
            response = input()
            if response == "y":
                print("Continuing...")
            else:
                print("Stopping...")
                return
        # Declare function
        compiled_code = compile(code, "<string>", "exec")
        namespace = {}
        exec(compiled_code, namespace)
        assert self.function_name in namespace, "Policy function not defined."
        self.policy = namespace[self.function_name]
        self.policy_code = code
    
    def _generate_critique(self, code, future_states) -> tuple[bool, str]:
        # Format the code and the future states as a JSON string
        predict_dict = {'code': code, 'future_states': str(future_states)}
        json_str = json.dumps(predict_dict)
        feedback = self._call_model(self.code_critic_prompt, json_str)
        print(feedback)
        # If valid JSON is not returned, use stop = False
        stop, feedback = parse_json_with_fallback(feedback)
        if not stop:
            feedback_json = json.loads(feedback)
            stop = feedback_json.get('stop', False)
        if stop:
            return True, ''
        return False, feedback

    def execute(self, observation) -> bool:
        observation_str = str(observation)
        # Generate initial code
        self._generate_policy(self.code_generation_prompt, observation_str)
        # Reflect
        for attempt in range(self.max_reflections):
            try:
                # Get feedback
                future_states = self._simulate(observation)
                stop, feedback = self._generate_critique(self.policy_code, future_states)
                if stop:
                    print("Code passed the critic.")
                    return True
                # Generate new code
                self._generate_policy(self.code_generation_prompt, feedback, self.policy_code)
            except Exception as e:
                print(f"Attempt {attempt+1}: Code generation failed: {str(e)}")
                return False
        print("Generated code did not pass critic, max reflections reached.")
        return True
    
    def infer(self, env, write_video = False):
        rewards = []
        observation = env.reset()
        done = False
        frames = []
        while not done:
            success = self.execute(observation)
            if not success:
                print("Code generation failed.")
                break
            for _ in range(self.max_simulation_timesteps):
                print(observation)
                frame = env.render()
                if write_video:
                    frames.append(frame)
                if len(observation) == 2:
                    observation = observation[0]
                action = self.policy(torch.tensor(observation, dtype = torch.float32))
                assert env.action_space.contains(action), "Invalid action."
                observation, reward, done, truncated, info = env.step(action)
                rewards.append(reward)
                if done or truncated:
                    done = True
                    break
        env.close()

        if rewards:
            print("Rewards:\n", rewards)
            print("Total Timesteps:", len(rewards))
            print("Total Reward:", sum(rewards))
            print("Average Reward:", sum(rewards) / len(rewards))

        if write_video:
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video = cv2.VideoWriter(self.function_name[:-6] + 'drc.avi', fourcc, 20.0, (width, height))

            for frame in frames:
                frame = np.array(frame, dtype = np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video.write(frame)

            video.release()

if __name__ == '__main__':
    from world_models import LunarLanderWorldModel
    from models.groq import GroqModel
    import os
    from dotenv import load_dotenv

    load_dotenv()

    CODE_GENERATION_PROMPT = open(os.path.join('prompts', 'lunar_lander_code.md')).read()
    CODE_CRITIC_PROMPT = open(os.path.join('prompts', 'lunar_lander_critic.md')).read()

    def postprocess_action(action):
        action_arr = np.zeros(4, dtype = np.float32)
        action_arr[action] = 1
        return action_arr

    world_model = LunarLanderWorldModel()
    world_model.load_state_dict(torch.load("lunar_lander_world_model/best_model_1.pth"))

    env = gym.make("LunarLander-v3", render_mode="rgb_array")

    llm = GroqModel(model_name="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))

    drc_agent = DreamReflectCode(
        llm=llm,
        world_model=world_model,
        code_generation_prompt=CODE_GENERATION_PROMPT,
        code_critic_prompt=CODE_CRITIC_PROMPT,
        function_name = 'lunar_lander_policy',
        max_reflections=1,
        max_simulation_timesteps=100,
        manually_check_code=True,
        postprocess_action=postprocess_action,
    )
    drc_agent.infer(env, write_video=True)
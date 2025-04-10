import gymnasium as gym
from models.groq import GroqModel
import os
import numpy as np
from dotenv import load_dotenv
import cv2

load_dotenv()

WRITE_VIDEO = True
MAX_EPISODE = 100
TIMESTEP = 100

llm = GroqModel(model_name="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))
env = gym.make("LunarLander-v3", render_mode="rgb_array" if WRITE_VIDEO else "human")

CODE_GENERATION_PROMPT = open(os.path.join('prompts', 'lunar_lander_code.md')).read()
CODE_CRITIC_PROMPT = open(os.path.join('prompts', 'lunar_lander_critic.md')).read()

class CodingAgent():
    def __init__(self, llm, max_attempts=3):
        self.llm = llm  # Use the passed LLM instance instead of creating a new one
        self.max_attempts = max_attempts
        self.policy_code = None
        self.namespace = {}
    
    def generate_policy(self, prompt, observation_example):
        observation_str = str(observation_example)
        prompt_with_obs = prompt.replace("{OBSERVATION_EXAMPLE}", observation_str)
        
        for attempt in range(self.max_attempts):
            try:
                # Generate code using LLM - use invoke instead of generate
                response = self.llm.invoke([prompt_with_obs])
                self.policy_code = self._extract_code(response)

                                # Test if code compiles
                compiled_code = compile(self.policy_code, "<string>", "exec")
                exec(compiled_code, self.namespace)
                
                # Check if lunar_lander_policy function exists
                if "lunar_lander_policy" not in self.namespace:
                    print(f"Attempt {attempt+1}: Generated code doesn't contain lunar_lander_policy function")
                    continue
                    
                return True
                
            except Exception as e:
                print(f"Attempt {attempt+1}: Code generation failed: {str(e)}")
        
        print("Failed to generate valid code after maximum attempts")
        return False
    
    def _extract_code(self, llm_response):
        """Extract code from LLM response text with better error handling."""
        try:
            # First try to extract code between markers
            if "```python" in llm_response and "```" in llm_response.split("```python", 1)[1]:
                code = llm_response.split("```python", 1)[1].split("```", 1)[0].strip()
            else:
                # Fallback to the entire response, assuming it's code
                code = llm_response.strip()
                
            # Debug output - can remove later
            print("\nExtracted code (first 100 chars):")
            print(code[:100] + "..." if len(code) > 100 else code)
            
            # Basic validation to catch obvious errors
            if code:
                # Check for balanced quotes
                single_quotes = code.count("'")
                double_quotes = code.count('"')
                if single_quotes % 2 != 0:
                    print("Warning: Unbalanced single quotes detected")
                if double_quotes % 2 != 0:
                    print("Warning: Unbalanced double quotes detected")
                    
            return code
        except Exception as e:
            print(f"Error during code extraction: {str(e)}")
            return llm_response.strip()
    
    def get_action(self, observation):
        try:
            action = self.namespace["lunar_lander_policy"](observation)
            return int(action)
        except Exception as e:
            print(f"Error executing policy: {str(e)}")
            raise  # Re-raise the exception
    
    def criticize_and_improve(self, critic_prompt, observation_examples, rewards):
        """
        Use critic prompt to improve the existing policy.
        
        Args:
            critic_prompt: The prompt template for code criticism
            observation_examples: List of example observations
            rewards: List of rewards received
            
        Returns:
            bool: Success status of code improvement
        """
        if not self.policy_code:
            return False
            
        print("\n=== PREVIOUS POLICY CODE ===")
        print(self.policy_code)
        print("===========================\n")
            
        # Prepare critic prompt with current code and performance data
        prompt = critic_prompt.replace("{CURRENT_CODE}", self.policy_code)
        prompt = prompt.replace("{OBSERVATIONS}", str(observation_examples))
        prompt = prompt.replace("{REWARDS}", str(rewards))
        
        # Generate improved code
        success = self.generate_policy(prompt, observation_examples[0])
        
        if success:
            print("\n=== POLICY CODE IMPROVED ===\n")
        else:
            print("\n=== POLICY IMPROVEMENT FAILED ===\n")
            
        return success

class ReflectionAgent():
    pass

class TrainSimulation():
    def __init__(self, write_video):
        self.env = gym.make("LunarLander-v3", render_mode="rgb_array" if write_video else "human")
        self.write_video = write_video
        self.frames = []
        self.episode_reward = 0
        self.current_observation = None
    
    def reset(self):
        """Reset the environment and return the initial observation."""
        observation, info = self.env.reset()
        self.current_observation = observation
        self.episode_reward = 0
        self.frames = []
        
        if self.write_video:
            self.frames.append(self.env.render())
            
        return observation, info
    
    def step(self, action):
        """Take an action in the environment.
        
        Args:
            action (int): The action to take (0-3 for LunarLander)
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.current_observation = observation
        self.episode_reward += reward
        
        if self.write_video:
            self.frames.append(self.env.render())
            
        return observation, reward, terminated, truncated, info
    
    def save_video(self, episode_num):
        """Save collected frames as a video file."""
        if not self.write_video or not self.frames:
            return
            
        height, width, _ = self.frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        filename = f'lunar_lander_episode_{episode_num}.avi'
        video = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))

        for frame in self.frames:
            frame = np.array(frame, dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame)

        video.release()
        print(f"Video saved as {filename}")
    
    
def main():
    try:
        sim = TrainSimulation(WRITE_VIDEO)
        agent = CodingAgent(llm)
        
        # Generate initial policy
        initial_observation, _ = sim.reset()
        success = agent.generate_policy(CODE_GENERATION_PROMPT, initial_observation)
        if not success:
            print("Failed to generate initial policy, using random actions")
        
        # Track observations and rewards for improvement
        all_observations = []
        all_rewards = []
        
        for episode in range(MAX_EPISODE):
            # Reset environment at the start of each episode
            observation, info = sim.reset()
            episode_observations = [observation]
            episode_rewards = []
            
            for _ in range(TIMESTEP):
                # Get action from agent policy
                action = agent.get_action(observation)
                
                # Take step in environment with agent's action
                observation, reward, terminated, truncated, info = sim.step(action)
                
                # Track data for improvement
                episode_observations.append(observation)
                episode_rewards.append(reward)
                
                # Check if episode is done
                if terminated or truncated:
                    break
            
            # Store episode data for agent improvement
            all_observations.append(episode_observations)
            all_rewards.append(sim.episode_reward)
            
            # Print episode results
            print(f"Episode {episode+1}: Total Reward = {sim.episode_reward:.2f}")
            
            # Every few episodes, try to improve the policy
            if (episode + 1) % 5 == 0 and episode > 0:
                print(f"Improving policy after episode {episode+1}...")
                agent.criticize_and_improve(CODE_CRITIC_PROMPT, 
                                           [obs[0] for obs in all_observations[-5:]], 
                                           all_rewards[-5:])
            
            # Save video of the episode
            if episode % 10 == 0:
                sim.save_video(episode + 1)
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")


if __name__ == "__main__":
    main()
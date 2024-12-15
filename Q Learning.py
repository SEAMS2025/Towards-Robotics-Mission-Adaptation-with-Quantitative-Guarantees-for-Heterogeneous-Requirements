import numpy as np
import subprocess
import tempfile
import os
from typing import Dict, Tuple, List, Any
from fractions import Fraction
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
import traceback

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.FileHandler("debug.log"),
                       logging.StreamHandler()
                   ])
logger = logging.getLogger(__name__)

class GridWorldEnv:
    """Grid World Environment with Multiple Obstacles"""
    def __init__(self, size: int):
        self.size = size
        self.goals = {
            1: (5, 8),  # First intermediate goal
            2: (4, 2),  # Second intermediate goal
            3: (9, 9)   # Final goal
        }
        self.obstacles = [(6, 6), (3, 4)]  
        self.max_steps = 200  
        self.steps = 0
        self.reset()

    def reset(self) -> Tuple[int, int, bool, bool, bool]:
        self.current_pos = (0, 0)
        self.visited_goals = set()
        self.steps = 0
        return self.get_state()

    def get_state(self) -> Tuple[int, int, bool, bool, bool]:
        x, y = self.current_pos
        g1_reached = 1 in self.visited_goals
        g2_reached = 2 in self.visited_goals
        g3_reached = 3 in self.visited_goals
        return (x, y, g1_reached, g2_reached, g3_reached)

    def step(self, action: int) -> Tuple[Tuple[int, int, bool, bool, bool], float, bool]:
        """Take action and return (state, reward, done)"""
        self.steps += 1
        x, y = self.current_pos
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left
        dx, dy = moves[action]

        # Apply movement
        new_x = max(0, min(self.size - 1, x + dx))
        new_y = max(0, min(self.size - 1, y + dy))
        
        # Check if new position is any obstacle
        if (new_x, new_y) in self.obstacles:
            new_x, new_y = x, y 
            reward = -10.0  
        else:
            self.current_pos = (new_x, new_y)
            reward = -0.1  

        if self.steps >= self.max_steps:
            return self.get_state(), reward, True

        done = False

        current_goal = len(self.visited_goals) + 1
        if self.current_pos == self.goals.get(current_goal):
            self.visited_goals.add(current_goal)
            if current_goal == 3:  
                reward = 100.0
                done = True
            else:
                reward = 10.0
        elif self.current_pos in self.goals.values():
            reward = -5.0

        if not done and len(self.visited_goals) < 3:
            next_goal = self.goals[len(self.visited_goals) + 1]
            distance = abs(new_x - next_goal[0]) + abs(new_y - next_goal[1])
            reward -= 0.01 * distance 

        return self.get_state(), reward, done

class PrismModelGenerator:
    """Generate PRISM models from policies """
    def __init__(self, size: int):
        self.size = size
        self.logger = logging.getLogger(__name__)
        self.obstacles = [(6, 6), (3, 4)]  
        self.goals = {
            1: (5, 8),  # First intermediate goal
            2: (4, 2),  # Second intermediate goal
            3: (9, 9)   # Final goal
        }
        self.initial_hit_prob = 0.2  
        self.debug_transitions = set()  

    def _generate_transitions(self, x: int, y: int, g1: bool, g2: bool, g3: bool, 
                           q_values: np.ndarray) -> list:
        """Generate transitions"""
        q_exp = np.exp((q_values - np.max(q_values)) / 0.1)
        probs = q_exp / np.sum(q_exp)

        int_probs = np.round(probs * 1000).astype(int)
        diff = 1000 - int_probs.sum()
        int_probs[np.argmax(probs)] += diff
        
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left
        transitions = []
        
        if (x, y) in self.goals.values():
            self.logger.debug(f"Generating transitions for goal state ({x}, {y})")
            self.logger.debug(f"Current goals reached: g1={g1}, g2={g2}, g3={g3}")
            self.logger.debug(f"Q-values: {q_values}")
            self.logger.debug(f"Action probabilities: {probs}")
        
        guard = (f"  [] (x={x} & y={y}"
                f" & g1={str(g1).lower()}"
                f" & g2={str(g2).lower()}"
                f" & g3={str(g3).lower()}) ->")
        
        updates = []
        
        for i, (dx, dy) in enumerate(moves):
            if int_probs[i] > 0:
                new_x = max(0, min(self.size - 1, x + dx))
                new_y = max(0, min(self.size - 1, y + dy))
                
                # Debug reachability for goal states
                if (new_x, new_y) in self.goals.values():
                    self.logger.debug(f"Found transition to goal: ({new_x}, {new_y})")
                    self.logger.debug(f"From state: ({x}, {y})")
                    self.logger.debug(f"Action: {moves[i]}")
                    self.logger.debug(f"Probability: {int_probs[i]/1000}")
                    self.debug_transitions.add((x, y, new_x, new_y))
                
                if (new_x, new_y) in self.obstacles:
                    if (x, y) not in self.obstacles + list(self.goals.values()):
                        avoid_prob = 1 - self.initial_hit_prob
                        prob_hit = (int_probs[i] / 1000) * self.initial_hit_prob
                        prob_avoid = (int_probs[i] / 1000) * avoid_prob

                        updates.append(f"{prob_hit}:(x'={x})"
                                    f" & (y'={y})"
                                    f" & (g1'={str(g1).lower()})"
                                    f" & (g2'={str(g2).lower()})"
                                    f" & (g3'={str(g3).lower()})")
                        
                        updates.append(f"{prob_avoid}:(x'={new_x})"
                                    f" & (y'={new_y})"
                                    f" & (g1'={str(g1).lower()})"
                                    f" & (g2'={str(g2).lower()})"
                                    f" & (g3'={str(g3).lower()})")
                        continue
                
                new_g1 = g1 or ((new_x, new_y) == self.goals[1])
                new_g2 = g2 or (g1 and (new_x, new_y) == self.goals[2])
                new_g3 = g3 or (g2 and (new_x, new_y) == self.goals[3])
                
                prob = int_probs[i] / 1000
                update = (f"{prob}:(x'={new_x})"
                         f" & (y'={new_y})"
                         f" & (g1'={str(new_g1).lower()})"
                         f" & (g2'={str(new_g2).lower()})"
                         f" & (g3'={str(new_g3).lower()})")
                updates.append(update)
        
        if updates:
            transition = f"{guard} " + " + ".join(updates) + ";"
            transitions.append(transition)
            
            if (x, y) in self.goals.values():
                self.logger.debug(f"Generated transition: {transition}")
        
        return transitions

    def generate_prism_model(self, policy: Dict[Tuple[int, int, bool, bool, bool], int],
                           q_table: Dict[Tuple[int, int, bool, bool, bool], np.ndarray]) -> str:
        """Generate PRISM model"""
        # Clear debug transitions
        self.debug_transitions = set()
        
        model = ["dtmc", "", f"const int N = {self.size};", ""]
        
        # Module definition
        model.append("module gridworld")
        model.append(f"  x : [0..{self.size-1}] init 0;")
        model.append(f"  y : [0..{self.size-1}] init 0;")
        model.append("  g1 : bool init false;")
        model.append("  g2 : bool init false;")
        model.append("  g3 : bool init false;")
        model.append("")
        
        # Generate transitions
        total_states = 0
        goal_transitions = 0
        
        for x in range(self.size):
            for y in range(self.size):
                for g1 in [False, True]:
                    for g2 in [False, True]:
                        for g3 in [False, True]:
                            state = (x, y, g1, g2, g3)
                            if state in q_table:
                                total_states += 1
                                transitions = self._generate_transitions(
                                    x, y, g1, g2, g3, q_table[state]
                                )
                                model.extend(transitions)
        
        model.append("endmodule")
        model.append("")
        
        model.append("// Labels for properties")
        model.append(f'label "at_goal1" = x={self.goals[1][0]} & y={self.goals[1][1]};')
        model.append(f'label "at_goal2" = x={self.goals[2][0]} & y={self.goals[2][1]};')
        model.append(f'label "at_goal3" = x={self.goals[3][0]} & y={self.goals[3][1]};')
        model.append(f'label "at_obstacle" = (x={self.obstacles[0][0]} & y={self.obstacles[0][1]}) | (x={self.obstacles[1][0]} & y={self.obstacles[1][1]});')
        
        self.logger.debug(f"Generated PRISM model with {total_states} states")
        self.logger.debug(f"Found {len(self.debug_transitions)} transitions to goal states")
        self.logger.debug("Goal transitions:")
        for x, y, new_x, new_y in self.debug_transitions:
            self.logger.debug(f"({x}, {y}) -> ({new_x}, {new_y})")
        
        model_str = "\n".join(model)
        return model_str
class PrismVerifier:
    """PRISM model checking interface"""
    def __init__(self, prism_bin_path: str):
        self.prism_bin_path = prism_bin_path
        self.logger = logging.getLogger(__name__)
        self.temp_files = [] 

    def verify_property(self, model_str: str, property_str: str) -> List[float]:
        model_path = None
        prop_path = None
        try:
            # Save model to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.nm', delete=False) as model_file:
                model_file.write(model_str)
                model_path = model_file.name
                self.temp_files.append(model_path)

            # Save properties to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.props', delete=False) as prop_file:
                prop_file.write(property_str)
                prop_path = prop_file.name
                self.temp_files.append(prop_path)

            # Debug log the model and properties
            self.logger.debug("PRISM Model:")
            self.logger.debug(model_str)
            self.logger.debug("Properties:")
            self.logger.debug(property_str)

            # Construct PRISM command
            cmd = [
                self.prism_bin_path,
                model_path,
                prop_path,
                "-explicit",
                "-javamaxmem", "4g",
                "-maxiters", "1000000", 
                "-power", 
                "-verbose",
                "-exportstates", "states.txt"  
            ]

            # Run PRISM
            self.logger.debug(f"Running PRISM command: {' '.join(cmd)}")
            result = subprocess.run(cmd, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE, 
                                 text=True)

            # Log PRISM output for debugging
            self.logger.debug("PRISM stdout:")
            self.logger.debug(result.stdout)
            self.logger.debug("PRISM stderr:")
            self.logger.debug(result.stderr)

            # Parse results
            probabilities = []
            for line in result.stdout.split('\n'):
                if "Result:" in line:
                    try:
                        value_str = line.split(':')[1].strip().split()[0]
                        prob = float(value_str)
                        probabilities.append(prob)
                        self.logger.debug(f"Parsed probability: {prob}")
                    except (IndexError, ValueError) as e:
                        self.logger.error(f"Failed to parse line: {line}")
                        self.logger.error(f"Error: {str(e)}")
                        continue

            if not probabilities:
                self.logger.error("No probabilities found in PRISM output")
                return [0.0] * 8  # Return zeros for all properties

            return probabilities

        except Exception as e:
            self.logger.error(f"Verification error: {str(e)}")
            self.logger.error(traceback.format_exc())
            return [0.0] * 8

        finally:
            # Cleanup temporary files
            for temp_file in self.temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception as e:
                    self.logger.error(f"Failed to delete temporary file {temp_file}: {str(e)}")
            self.temp_files = []

class SimplifiedVerifier:
    """Verifier"""
    def __init__(self, prism_verifier: PrismVerifier):
        self.prism_verifier = prism_verifier
        self.logger = logging.getLogger(__name__)
        self.ltl_probabilities = []
        self.verification_history = []

    def verify_policy(self, model_str: str) -> float:
        """Fix the obstacle avoidance property"""
        property_str = (
        "P=? [ F \"at_goal1\" ];\n"
        "P=? [ F \"at_goal2\" ];\n"
        "P=? [ F \"at_goal3\" ];\n"
        "P=? [ !\"at_goal2\" U \"at_goal1\" ];\n"
        "P=? [ !\"at_goal3\" U (\"at_goal1\" & (!\"at_goal3\" U \"at_goal2\")) ];\n"
        "P=? [ (!\"at_goal2\" U \"at_goal1\") & (!\"at_goal3\" U (\"at_goal1\" & (!\"at_goal3\" U \"at_goal2\"))) ];\n"
        "P=? [ G<=20 !\"at_obstacle\" ];\n" 
        )
        
        probabilities = self.prism_verifier.verify_property(model_str, property_str)
        self.ltl_probabilities.append(probabilities)
        
        score = self._calculate_score(probabilities)
        self.logger.info(f"Goal Sequence Probabilities: {probabilities}")
        self.logger.info(f"Combined Score: {score:.4f}")
        return score

    def _calculate_score(self, results: List[float]) -> float:
        """Calculate policy score with balanced weights"""
        if len(results) >= 7:
            goal1_reach = results[0]
            goal2_reach = results[1] 
            goal3_reach = results[2]
            seq1 = results[3]  # G1 before G2
            seq2 = results[4]  # G1G2 before G3
            seq3 = results[5]  # Complete sequence
            avoid_obstacle = results[6]  # Obstacle avoidance
            
            score = (0.1 * goal1_reach + 
                    0.1 * goal2_reach + 
                    0.1 * goal3_reach + 
                    0.15 * seq1 +
                    0.15 * seq2 + 
                    0.2 * seq3 +
                    0.2 * avoid_obstacle) 
            
            return score
        return 0.0


    def save_probabilities_to_file(self, agent, save_dir: str = "results"):
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, 'ltl_probabilities_and_rewards.txt')
        
        try:
            with open(output_path, 'w') as f:
                f.write("Episode,Reach_G1,Reach_G2,Reach_G3,G1_before_G2,G1G2_before_G3,"
                    "Complete_Sequence,Avoid_Obstacle,Episode_Reward,Path_Exists,Score\n")
                
                for episode, probs in enumerate(self.ltl_probabilities):
                    prob_str = ','.join(f"{p:.4f}" for p in probs)
                    reward = agent.episode_rewards[episode] if episode < len(agent.episode_rewards) else 0.0
                    score = self._calculate_score(probs)
                    f.write(f"{episode},{prob_str},{reward:.4f},1.0,{score:.4f}\n")  # Added Path_Exists=1.0
                    
                self.logger.info(f"Saved LTL probabilities and rewards to {output_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving probabilities to file: {str(e)}")
class LTLGuidedQLearning:
    """Q-Learning with probability verification for grid world navigation"""
    def __init__(self, size: int, prism_verifier: PrismVerifier):
        self.size = size
        self.env = GridWorldEnv(size)
        self.model_generator = PrismModelGenerator(size)
        self.verifier = SimplifiedVerifier(prism_verifier)
        self.logger = logging.getLogger(__name__)
        
        # Initialize Q-table and parameters
        self.q_table = {}
        self.action_space = 4
        self._initialize_q_table()
        
        # Learning parameters
        self.epsilon = 0.8      
        self.epsilon_min = 0.1  
        self.epsilon_decay = 0.995  
        self.alpha = 0.2       
        self.alpha_min = 0.05  
        self.gamma = 0.99   
        
        # Store PRISM probabilities (only for monitoring)
        self.prism_probs = {
            'goal1': 0.0,
            'goal2': 0.0,
            'goal3': 0.0,
            'seq1': 0.0,
            'seq2': 0.0,
            'seq3': 0.0,
            'avoid_obstacle': 0.0,
            'path_exists': 0.0
        }
        
        # Performance tracking
        self.episode_rewards = []
        self.ltl_scores = []
        self.best_policy = None
        self.best_ltl_score = 0.0
        self.training_stats = []

    def _initialize_q_table(self):
        """Initialize Q-table"""
        for x in range(self.size):
            for y in range(self.size):
                for g1 in [False, True]:
                    for g2 in [False, True]:
                        for g3 in [False, True]:
                            state = (x, y, g1, g2, g3)
                            self.q_table[state] = np.zeros(self.action_space)  

    def _update_prism_probabilities(self, probabilities: List[float]):
        """Update stored PRISM verification probabilities (for monitoring only)"""
        if len(probabilities) >= 8:
            self.prism_probs = {
                'goal1': probabilities[0],
                'goal2': probabilities[1],
                'goal3': probabilities[2],
                'seq1': probabilities[3],
                'seq2': probabilities[4],
                'seq3': probabilities[5],
                'avoid_obstacle': probabilities[6],
                'path_exists': probabilities[7]
            }

    def _calculate_reward(self, state: tuple, next_state: tuple) -> float:
        x, y, g1, g2, g3 = state
        next_x, next_y, next_g1, next_g2, next_g3 = next_state
        
        reward = -0.1
        

        if np.random.random() < 0.3:  
            reward -= np.random.uniform(0, 1.0)
        
        # Obstacle penalty
        if (next_x, next_y) in [(6, 6), (3, 4)]: 
            reward += -10.0
            
        if not g1 and next_x == 5 and next_y == 8: 
            if np.random.random() < 0.7:  
                reward += 100.0
        elif g1 and not g2 and next_x == 4 and next_y == 2:  
            if np.random.random() < 0.7:  
                reward += 100.0
        elif g1 and g2 and not g3 and next_x == 9 and next_y == 9:  
            if np.random.random() < 0.7:  
                reward += 1000.0
            
        elif ((not g1 and (next_x == 4 and next_y == 2 or next_x == 9 and next_y == 9)) or
              (not g2 and next_x == 9 and next_y == 9)):
            reward -= 10.0
            
        return reward

    def _get_action(self, state: tuple) -> int:
        """Select action using simple epsilon-greedy policy with higher exploration"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space)
            
        noise = np.random.normal(0, 0.1, self.action_space)
        noisy_q_values = self.q_table[state] + noise
        return int(np.argmax(noisy_q_values))

    def _adjust_parameters(self, ltl_score: float):
        if ltl_score > self.best_ltl_score:
            self.best_ltl_score = ltl_score
            self.best_policy = self.get_current_policy()
            
        self.epsilon = max(self.epsilon_min, 
                         self.epsilon * self.epsilon_decay)
        self.alpha = max(self.alpha_min,
                       self.alpha * 0.998)
            
        # Track training statistics
        self.training_stats.append({
            'ltl_score': ltl_score,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'best_score': self.best_ltl_score
        })

    def get_current_policy(self) -> Dict[Tuple[int, int, bool, bool, bool], int]:
        """Extract current policy from Q-table"""
        return {state: int(np.argmax(q_values)) 
                for state, q_values in self.q_table.items()}

    def save_training_stats(self, save_dir: str = "results"):
        """Save training statistics"""
        os.makedirs(save_dir, exist_ok=True)
        stats_file = os.path.join(save_dir, 'training_statistics.txt')
        
        try:
            with open(stats_file, 'w') as f:
                f.write("Episode,LTL_Score,Epsilon,Alpha,Best_Score\n")
                for episode, stats in enumerate(self.training_stats):
                    f.write(f"{episode},"
                           f"{stats['ltl_score']:.4f},"
                           f"{stats['epsilon']:.4f},"
                           f"{stats['alpha']:.4f},"
                           f"{stats['best_score']:.4f}\n")
                           
            self.logger.info(f"Training statistics saved to {stats_file}")
        except Exception as e:
            self.logger.error(f"Failed to save training statistics: {str(e)}")

    def train(self, episodes: int, verify_interval: int = 100):
        """Training loop with probability verification"""
        try:
            start_time = time.time()
            best_episode = 0
            
            for episode in range(episodes):
                state = self.env.reset()
                total_reward = 0
                done = False
                steps = 0
                
                # Verification step (only for monitoring)
                if episode % verify_interval == 0:
                    try:
                        model_str = self.model_generator.generate_prism_model(
                            self.get_current_policy(), self.q_table)
                        ltl_score = self.verifier.verify_policy(model_str)
                        
                        if self.verifier.ltl_probabilities:
                            latest_probs = self.verifier.ltl_probabilities[-1]
                            self._update_prism_probabilities(latest_probs)
                        
                        self.ltl_scores.append(ltl_score)
                        self._adjust_parameters(ltl_score)
                        
                        # Log progress
                        self.logger.info(
                            f"Episode {episode}: LTL Score = {ltl_score:.4f}, "
                            f"Epsilon = {self.epsilon:.3f}, "
                            f"Alpha = {self.alpha:.3f}, "
                            f"Best Score = {self.best_ltl_score:.4f}, "
                            f"Time = {(time.time() - start_time)/60:.1f}m"
                        )
                        
                        if ltl_score > self.best_ltl_score:
                            best_episode = episode
                            
                    except Exception as e:
                        self.logger.error(f"Verification error in episode {episode}: {str(e)}")
                        continue
                
                while not done:
                    action = self._get_action(state)
                    next_state, reward, done = self.env.step(action)
                    
                    # Calculate basic reward without LTL guidance
                    basic_reward = self._calculate_reward(state, next_state)
                    
                    # Standard Q-learning update
                    current_q = self.q_table[state][action]
                    next_max_q = np.max(self.q_table[next_state])
                    new_q = current_q + self.alpha * (
                        basic_reward + self.gamma * next_max_q - current_q
                    )
                    self.q_table[state][action] = new_q
                    
                    state = next_state
                    total_reward += reward
                    steps += 1
                    
                    if steps >= self.env.max_steps:
                        done = True
                
                self.episode_rewards.append(total_reward)
                
                # Log episode results
                if episode % 10 == 0:
                    self.logger.info(f"Episode {episode}: Steps = {steps}, "
                                   f"Total Reward = {total_reward:.2f}")
            
            # Final verification
            try:
                model_str = self.model_generator.generate_prism_model(
                    self.get_current_policy(), self.q_table)
                final_score = self.verifier.verify_policy(model_str)
                self.logger.info(f"\nFinal LTL Score: {final_score:.4f}")
            except Exception as e:
                self.logger.error(f"Final verification error: {str(e)}")
            
            # Save results
            self.verifier.save_probabilities_to_file(self)
            
            # Training summary
            training_time = time.time() - start_time
            self.logger.info(f"\nTraining completed in {training_time/60:.1f} minutes")
            self.logger.info(f"Best LTL Score: {self.best_ltl_score:.4f} "
                           f"(Episode {best_episode})")
            
            return self.best_policy, self.best_ltl_score
            
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def plot_training_progress(self, save_dir: str = "results"):
        """Plot training progress metrics"""
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 10))
        
        # Plot episode rewards
        plt.subplot(2, 1, 1)
        episodes = range(len(self.episode_rewards))
        plt.plot(episodes, self.episode_rewards, 'b-', alpha=0.3, label='Rewards')
        
        window = min(10, len(self.episode_rewards))
        if len(self.episode_rewards) >= window:
            smoothed = np.convolve(self.episode_rewards,
                                np.ones(window)/window,
                                mode='valid')
            plt.plot(range(window-1, len(self.episode_rewards)),
                    smoothed, 'r-', label=f'Moving Average (w={window})')
        
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True)
        
        # Plot LTL scores and training parameters
        plt.subplot(2, 1, 2)
        verify_episodes = range(0, len(self.training_stats))
        
        plt.plot(verify_episodes,
                [s['ltl_score'] for s in self.training_stats],
                'g-', label='LTL Score')
        plt.plot(verify_episodes,
                [s['epsilon'] for s in self.training_stats],
                'b--', label='Epsilon')
        plt.plot(verify_episodes,
                [s['alpha'] for s in self.training_stats],
                'r--', label='Alpha')
        
        plt.title('LTL Scores and Parameters')
        plt.xlabel('Verification Episode')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
def get_prism_path():
    """Get PRISM executable path"""
    prism_path = "Your PRISM PATH"  # Update this path for your system
    if os.path.exists(prism_path) and os.access(prism_path, os.X_OK):
        return prism_path
    raise FileNotFoundError(f"PRISM executable not found at {prism_path}")

def main():
    try:
        # Set up logging
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("logs/training.log"),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        
        # Initialize PRISM
        try:
            prism_path = get_prism_path()
            logger.info(f"Successfully initialized PRISM at: {prism_path}")
        except FileNotFoundError as e:
            logger.error(f"PRISM initialization failed: {e}")
            return
        
        # Training parameters
        verify_interval = 1     
        total_episodes = 200    
        grid_size = 10          
        
        # Create results directory
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize verifier and agent
        logger.info("Initializing verifier and agent...")
        prism_verifier = PrismVerifier(prism_path)
        agent = LTLGuidedQLearning(size=grid_size, 
                                  prism_verifier=prism_verifier)
        
        # Train agent
        logger.info(f"Starting training on {grid_size}x{grid_size} grid")
        logger.info(f"Goals: G1(5,8), G2(4,2), G3(9,9)")
        logger.info(f"Obstacles: (6,6), (3,4)")
        logger.info(f"Training for {total_episodes} episodes")
        
        start_time = time.time()
        best_policy, best_ltl_score = agent.train(
            episodes=total_episodes, 
            verify_interval=verify_interval
        )
        training_time = time.time() - start_time
        
        # Save results
        agent.save_training_stats(results_dir)
        agent.verifier.save_probabilities_to_file(agent, results_dir)
        
        # Plot training progress
        agent.plot_training_progress(results_dir)
        
        # Log final results
        logger.info("\nTraining Summary:")
        logger.info(f"Training time: {training_time/60:.1f} minutes")
        logger.info(f"Best LTL Score: {best_ltl_score:.4f}")
        logger.info(f"Results saved to {results_dir}")
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
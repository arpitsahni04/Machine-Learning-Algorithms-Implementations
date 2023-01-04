import argparse
import numpy as np

from environment import MountainCar, GridWorld
import sys
""" The environment returns states, represented as 1D numpy arrays, rewards, 
and a Boolean flag indicating whether the episode has terminated. The environment 
accepts actions, represented as integers.

class Environment: # either MountainCar or GridWorld

    def __init__(self, mode, debug=False):
        Initialize the environment with the mode, which can be either "raw" 
        (for the raw state representation) or "tile" (for the tiled state 
        representation). The raw state representation contains the position and 
        velocity; the tile representation contains zeroes for the non-active 
        tile indices and ones for the active indices. GridWorld must be used in 
        tile mode. The debug flag will log additional information for you; 
        make sure that this is turned off when you submit to the autograder.

        self.state_space = an integer representing the size of the state vector
        self.action_space = an integer representing the range for the valid actions

        You should make use of env.state_space and env.action_space when creating 
        your weight matrix.

    def reset(self):
        Resets the environment to initial conditions. Returns:

            (1) state : A numpy array of size self.state_space, representing 
                        the initial state.
    
    def step(self, action):
        Updates itself based on the action taken. The action parameter is an 
        integer in the range [0, 1, ..., self.action_space). Returns:

            (1) state : A numpy array of size self.state_space, representing 
                        the new state that the agent is in after taking its 
                        specified action.
            
            (2) reward : A float indicating the reward received at this step.

            (3) done : A Boolean flag indicating whether the episode has 
                        terminated; if this is True, you should reset the 
                        environment and move on to the next episode.
    
    def render(self, mode="human"):
        Renders the environment at the current step. Only supported for MountainCar.


For example, for the GridWorld environment, you could do:

    env = GridWorld(mode="tile")

"""

def parse_args() -> tuple:
    """
    Parses all args and returns them. Returns:

        (1) env_type : A string, either "mc" or "gw" indicating the type of 
                    environment you should use
        (2) mode : A string, either "raw" or "tile"
        (3) weight_out : The output path of the file containing your weights
        (4) returns_out : The output path of the file containing your returns
        (5) episodes : An integer indicating the number of episodes to train for
        (6) max_iterations : An integer representing the max number of iterations 
                    your agent should run in each episode
        (7) epsilon : A float representing the epsilon parameter for 
                    epsilon-greedy action selection
        (8) gamma : A float representing the discount factor gamma
        (9) lr : A float representing the learning rate
    
    Usage:
        env_type, mode, weight_out, returns_out, episodes, max_iterations, epsilon, gamma, lr = parse_args()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, choices=["mc", "gw"])
    parser.add_argument("mode", type=str, choices=["raw", "tile"])
    parser.add_argument("weight_out", type=str)
    parser.add_argument("returns_out", type=str)
    parser.add_argument("episodes", type=int)
    parser.add_argument("max_iterations", type=int)
    parser.add_argument("epsilon", type=float)
    parser.add_argument("gamma", type=float)
    parser.add_argument("learning_rate", type=float)

    args = parser.parse_args()

    return args.env, args.mode, args.weight_out, args.returns_out, args.episodes, args.max_iterations, args.epsilon, args.gamma, args.learning_rate


if __name__ == "__main__":

    env_type, mode, weight_out, returns_out, episodes, max_iterations, epsilon, gamma, lr = parse_args()

    
    if env_type == "mc":
        env = MountainCar(mode = mode)#None # Replace me
    elif env_type == "gw":
        env = GridWorld(mode = mode,debug = True)#None # Replace me!
    else: raise Exception(f"Invalid environment type {env_type}")

    # # GridWorld must be used in tile mode.
    # if mode == "tile":
    #     env = GridWorld(mode = mode)#None # Replace me!
    reward_line = []
    print("state space",env.action_space)
    print("state space",env.state_space)
    W = np.zeros((env.action_space, env.state_space + 1), dtype=np.float64)
    
    with open(returns_out,"w") as ofile:
        for episode in range(episodes):

            # Get the initial state by calling env.reset()
            initial  = env.reset()
            #hstack
            initial = initial.reshape(initial.shape[0],1)
            # bias initialize
            initial =  np.vstack(([1], initial))
            total_reward= 0
            for iteration in range(max_iterations):
                # Select an action based on the state via the epsilon-greedy strategy
                # pick the action represented by the smallest number if there is a 
                # draw in the greedy action selection process
                computed_Q = np.dot(W,initial) 
                greedy_action = np.argmax(computed_Q)
                random_action = np.random.choice(env.action_space)
                # print(greedy_action,random_action)
                action  = np.random.choice([greedy_action,random_action],p=[1-epsilon,epsilon])
                # Take a step in the environment with this action, and get the 
                # returned next state, reward, and done flag
                new_state,reward,done = env.step(action) 
                total_reward+=reward
                new_state =  np.vstack(([1], new_state.reshape(new_state.shape[0],1)))
                # Using the original state, the action, the next state, and 
                # the reward, update the parameters. Don't forget to update the 
                # bias term!
                initial = np.squeeze(initial.T)
                W[action] -= lr*(computed_Q[action] - (reward + gamma*(np.max(np.dot(W,new_state) )) ) )*initial
                # Remember to break out of this inner loop if the environment signals done!
                if done:
                    break
                initial = new_state
            reward_line.append(total_reward)
            ofile.write(str(total_reward))
            ofile.write("\n")
    # np.savetxt("rewards "+mode,reward_line)
    print(reward_line)
    # Save your weights and returns. The reference solution uses 
    np.savetxt(weight_out,W,fmt="%.18e", delimiter=" ")

    

import random
import torch


def select_greedy_action(state, policy_net, action_size):
    """ Select the greedy action
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    action_size: int
        number of possible actions
    Returns
    -------
    int
        ID of selected action
    """

    # TODO: Select greedy action
     # Convert state to tensor and ensure it has the correct shape
    #state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # Add batch dimension (1, *state_shape)
    

    # Get Q-values from policy network for this state
    with torch.no_grad():  # No need for gradients during action selections
        q_values = policy_net(state)

    # Select the action with the highest Q-value
    action = q_values.argmax(dim=1).item()

    return action

def select_exploratory_action(state, policy_net, action_size, exploration, t):
    """ Select an action according to an epsilon-greedy exploration strategy
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    action_size: int
        number of possible actions
    exploration: LinearSchedule
        linear exploration schedule
    t: int
        current time-step
    Returns
    -------
    int
        ID of selected action
    """

    # TODO: Select exploratory action

    # Get the current value of epsilon
    epsilon = exploration.value(t)

    # Generate a random number to decide between exploration and exploitation
    if random.random() < epsilon:
        # Exploration: Choose a random action
        action = random.randint(0, action_size - 1)
    else:
        # Exploitation: Select the greedy action
        action = select_greedy_action(state, policy_net, action_size)

    return action

def get_action_set():
    """ Get the list of available actions
    Returns
    -------
    list
        list of available actions
    """
    return [[-1.0, 0.05, 0], [1.0, 0.05, 0], [0, 0.5, 0], [0, 0, 1.0]]

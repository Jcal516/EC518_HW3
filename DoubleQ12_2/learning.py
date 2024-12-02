import numpy as np
import torch
import torch.nn.functional as F


def perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device):
    """ Perform a deep Q-learning step
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    optimizer: torch.optim.Adam
        optimizer
    replay_buffer: ReplayBuffer
        replay memory storing transitions
    batch_size: int
        size of batch to sample from replay memory 
    gamma: float
        discount factor used in Q-learning update
    device: torch.device
        device on which to the models are allocated
    Returns
    -------
    float
        loss value for current learning step
    """

    # TODO: Run single Q-learning step
    """ Steps: 
        1. Sample transitions from replay_buffer
        2. Compute Q(s_t, a)
        3. Compute \max_a Q(s_{t+1}, a) for all next states.
        4. Mask next state values where episodes have terminated
        5. Compute the target
        6. Compute the loss
        7. Calculate the gradients
        8. Clip the gradients
        9. Optimize the model
    """
    #Step 1
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    
    #convert to tensors if necessary
    # Convert numpy arrays to PyTorch tensors
    states = torch.tensor(states, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.float32, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)

    
    #Step 2
    states_action_values = policy_net(states)

    # Step 3: Compute next state Q-values using Double Q-learning logic
    with torch.no_grad():
        # Use policy_net to select the best action in next_states
        next_action_indices = policy_net(next_states).argmax(1, keepdim=True)
        # Use target_net to evaluate the value of those actions
        next_state_values = target_net(next_states).gather(1, next_action_indices).squeeze(1)

    #Step 4
    next_state_values = next_state_values * (1 - dones)

    #Step 5
    target_q_values = rewards + (gamma * next_state_values)

    #Step 6
    #print(states_action_values.shape)
    #print(target_q_values.shape)
    loss = F.mse_loss(states_action_values, target_q_values.unsqueeze(1))

    # Step 7
    optimizer.zero_grad()
    loss.backward()

    #Step 8
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)

    # Step 10: Perform a gradient descent step
    optimizer.step()

    return loss.item()

def update_target_net(policy_net, target_net):
    """ Update the target network
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    """

    # TODO: Update target network
    target_net.load_state_dict(policy_net.state_dict())

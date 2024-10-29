import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, action_size, device):
        """ Create Q-network
        Parameters
        ----------
        action_size: int
            number of actions
        device: torch.device
            device on which to the model will be allocated
        """
        super().__init__()

        self.device = device 
        self.action_size = action_size

        # TODO: Create network
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)  # (N, 32, 78, 58)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)  # (N, 64, 38, 28)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)  # (N, 64, 36, 26)

         # Fully connected layers
        self.fc1 = nn.Linear(64 * 36 * 26, 512)  # Flatten the output of the conv3 layer to feed into fully connected
        self.fc2 = nn.Linear(512, action_size)   # Output layer for action values (Q-values)

    def forward(self, observation):
        """ Forward pass to compute Q-values
        Parameters
        ----------
        observation: np.array
            array of state(s)
        Returns
        ----------
        torch.Tensor
            Q-values  
        """

        # TODO: Forward pass through the network

        #Transform observation from np.array to Tensor
        if(torch.is_tensor(observation)):
            observation_tensor = observation
        else: 
            observation_tensor = torch.from_numpy(observation).float()
    
        #Transpose the dimensions  
        if observation_tensor.dim() == 3:
            observation_tensor = observation_tensor.unsqueeze(0)
        elif observation_tensor.dim() == 4:
            observation_tensor = observation_tensor.permute(0, 3, 1, 2)  # Shape: (N, C, H, W)
        
        #Move to device
        observation_tensor = observation_tensor.to(self.device)

        # Apply convolutional layers with ReLU activations
        x = F.relu(self.conv1(observation_tensor))  # Shape: (batch_size, 32, 78, 58)
        x = F.relu(self.conv2(x))            # Shape: (batch_size, 64, 38, 28)
        x = F.relu(self.conv3(x))            # Shape: (batch_size, 64, 36, 26)

        # Flatten the output from the conv layers
        x = x.reshape(x.size(0), -1)  # Change to reshape  

        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))    # Shape: (batch_size, 512)
        q_values = self.fc2(x)     # Output layer for Q-values, shape: (batch_size, action_size)

        return q_values

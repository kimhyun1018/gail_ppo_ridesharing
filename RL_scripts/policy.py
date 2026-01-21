import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F

class ShuttlePPOPolicy(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=128, lstm_hidden_dim=128, device='cpu'):
        super(ShuttlePPOPolicy, self).__init__()
        self.device = device  # Set the device attribute

        # LSTM hidden dimensions
        self.lstm_hidden_dim = lstm_hidden_dim

        # Shared input layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # LSTM layer for temporal information
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)

        # Policy output layer for actions (3 actions for a single shuttle: move, pick up, drop off)
        self.policy_head = nn.Linear(lstm_hidden_dim, 3)

        # Value function output layer
        self.value_head = nn.Linear(lstm_hidden_dim, 1)

        # Initialize the weights using Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.policy_head.weight)
        nn.init.xavier_uniform_(self.value_head.weight)

        # Move the model to the specified device
        self.to(self.device)

    def reset_lstm(self, batch_size=1):
        """
        Reset the hidden state for the LSTM with the correct batch size.
        """
        return (torch.zeros(1, batch_size, self.lstm_hidden_dim, device=self.device),
                torch.zeros(1, batch_size, self.lstm_hidden_dim, device=self.device))

    def forward(self, state, hidden_state=None):
        # Ensure the input state is on the correct device
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state = state.to(self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension if it's a single state
            is_single_state = True
        else:
            is_single_state = False

        # Input normalization to improve stability
        state = (state - state.mean(dim=-1, keepdim=True)) / (state.std(dim=-1, keepdim=True) + 1e-8)

        # Process the input through the fully connected layer
        x = F.relu(self.fc1(state))

        if hidden_state is None:
            hidden_state = self.reset_lstm(batch_size=x.size(0))  # Initialize hidden state based on batch size

        # Pass the features through the LSTM layer
        lstm_out, hidden_state = self.lstm(x.unsqueeze(1), hidden_state)
        lstm_out = lstm_out.squeeze(1)  # Remove the sequence length dimension after LSTM

        # Policy logits for actions
        action_logits = self.policy_head(lstm_out)

        # Value function output
        state_value = self.value_head(lstm_out)

        if is_single_state:
            action_logits = action_logits.squeeze(0)
            state_value = state_value.squeeze(0)

        return action_logits, state_value, hidden_state

    def select_actions(self, state, hidden_state=None):
        # Ensure the state is a tensor and on the correct device
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
        else:
            state = state.to(self.device)

        # Ensure state has batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension if it's a single state

        # Forward pass to get logits and values
        action_logits, _, hidden_state = self.forward(state, hidden_state)

        # Apply softmax to get probabilities for actions
        action_probs = F.softmax(action_logits, dim=-1)  # Shape: (batch_size, 3)

        # Sample actions
        m = Categorical(action_probs)
        actions = m.sample()  # Shape: (batch_size,)

        # If state was a single sample, we need to remove the batch dimension
        if state.shape[0] == 1:
            actions = actions.squeeze(0)  # Remove batch dimension if we had a single sample

        # Convert actions to list
        return actions.item(), hidden_state

    def evaluate(self, states, actions, hidden_state=None):
        """
        Evaluate the policy on a batch of states and actions. This is used during training for policy updates.
        """
        # Ensure the inputs are on the correct device
        states = states.to(self.device)
        actions = actions.to(self.device)

        # Forward pass through the policy network
        action_logits, state_values, hidden_state = self.forward(states, hidden_state)

        # Create action distributions
        action_probs = F.softmax(action_logits, dim=-1)
        action_log_probs = F.log_softmax(action_logits, dim=-1)

        # Gather the log probabilities of the selected actions
        selected_action_log_probs = action_log_probs[range(len(actions)), actions]

        # Return the log probabilities and the state values
        return selected_action_log_probs, state_values.squeeze()


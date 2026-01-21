import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
import joblib

class ShuttleDataset(Dataset):
    def __init__(self, states, actions):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.long)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

class BehavioralCloningModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BehavioralCloningModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

def load_expert_data_from_folder(folder_path):
    """
    Load all expert data from the .h5 files in a folder.
    Combine the states and actions from each file into a single dataset.
    """
    all_states = []
    all_actions = []

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.h5'):
            file_path = os.path.join(folder_path, file_name)
            with h5py.File(file_path, 'r') as f:
                for episode in f.keys():
                    states = np.array(f[episode]['states'])
                    actions = np.array(f[episode]['actions'])
                    all_states.append(states)
                    all_actions.append(actions)

    # Concatenate all the states and actions
    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    
    return all_states, all_actions

def train_behavioral_cloning(train_folder, val_folder, scaler_folder, output_model_path, input_dim, output_dim):
    # Load all train and validation data
    train_states, train_actions = load_expert_data_from_folder(train_folder)
    val_states, val_actions = load_expert_data_from_folder(val_folder)

    # Optionally, load the scaler (if needed for evaluation)
    scaler_files = [f for f in os.listdir(scaler_folder) if f.endswith('.pkl')]
    for scaler_file in scaler_files:
        scaler_path = os.path.join(scaler_folder, scaler_file)
        scaler = joblib.load(scaler_path)
        train_states = scaler.transform(train_states)
        val_states = scaler.transform(val_states)

    # Create datasets and loaders
    train_dataset = ShuttleDataset(train_states, train_actions)
    val_dataset = ShuttleDataset(val_states, val_actions)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Initialize model, loss, optimizer
    model = BehavioralCloningModel(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 2000
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for batch_states, batch_actions in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_states)
            loss = criterion(outputs, batch_actions)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_states, batch_actions in val_loader:
                outputs = model(batch_states)
                loss = criterion(outputs, batch_actions)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_model_path)
            print(f"Best model saved at epoch {epoch+1}")

    print("Behavioral Cloning training completed.")

if __name__ == "__main__":
    # Define paths
    train_folder = '/home/hyun/AS_IL_GAIL_RL/data/preprocess_BC/session_9_28/train/'
    val_folder = '/home/hyun/AS_IL_GAIL_RL/data/preprocess_BC/session_9_28/val/'
    scaler_folder = '/home/hyun/AS_IL_GAIL_RL/data/preprocess_BC/session_9_28/scaler/'
    output_model_path = '/home/hyun/AS_IL_GAIL_RL/IL_models/BC_models/session_9_28/behavioral_cloning_2.pth'

    # Example input and output dimensions (adjust as needed)
    input_dim = 27  # Number of features in the state (adjust based on your actual input size)
    output_dim = 3  # Number of possible actions

    # Train BC model
    train_behavioral_cloning(train_folder, val_folder, scaler_folder, output_model_path, input_dim, output_dim)

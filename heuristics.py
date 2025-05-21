import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

class BaseHeuristic:
    def __init__(self, n=11, k=4):
        self._n = n
        self._k = k

    def get_h_values(self, states):
        states_as_list = [state.get_state_as_list() for state in states]
        gaps = []

        for state_as_list in states_as_list:
            gap = 0
            if state_as_list[0] != 1:
                gap = 1

            for i in range(len(state_as_list) - 1):
                if abs(state_as_list[i] - state_as_list[i + 1]) != 1:
                    gap += 1

            gaps.append(gap)

        return gaps
    

class DeepCubeA(nn.Module):
    def __init__(self, input_dim):
        super(DeepCubeA, self).__init__()
        self.fc1 = nn.Linear(input_dim, 5000)
        self.bn1 = nn.BatchNorm1d(5000)

        self.fc2 = nn.Linear(5000, 5000)
        self.bn2 = nn.BatchNorm1d(5000)

        self.fc3 = nn.Linear(5000, 1000)
        self.bn3 = nn.BatchNorm1d(1000)
        self.match_dim1 = nn.Linear(5000, 1000)  # Transformation layer for residual

        self.fc4 = nn.Linear(1000, 1000)
        self.bn4 = nn.BatchNorm1d(1000)

        self.fc5 = nn.Linear(1000, 1000)
        self.bn5 = nn.BatchNorm1d(1000)

        self.fc6 = nn.Linear(1000, 1000)
        self.bn6 = nn.BatchNorm1d(1000)

        self.fc7 = nn.Linear(1000, 1000)
        self.bn7 = nn.BatchNorm1d(1000)

        self.fc8 = nn.Linear(1000, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))

        residual1 = self.match_dim1(x)  # Adjusting residual to match output of fc3
        x = F.relu(self.bn3(self.fc3(x)))
        x = x + residual1  # Adding the transformed residual using out-of-place operation

        residual2 = x  # No need to transform since dimensions match
        x = F.relu(self.bn4(self.fc4(x)))
        x = x + residual2  # Adding the residual using out-of-place operation

        residual3 = x  # No need to transform since dimensions match
        x = F.relu(self.bn5(self.fc5(x)))
        x = x + residual3  # Adding the residual using out-of-place operation

        x = F.relu(self.bn6(self.fc6(x)))
        x = F.relu(self.bn7(self.fc7(x)))
        x = self.fc8(x)

        return x
    
class HeuristicModel(nn.Module):
    def __init__(self, input_dim):
        super(HeuristicModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class LearnedHeuristic:
    def __init__(self, n=11, k=4, model=None):
        self._n = n
        self._k = k
        self._model = HeuristicModel(n)
        if model!=None:
            self._model= DeepCubeA(n)
        self._criterion = nn.MSELoss()
        self._optimizer = optim.Adam(self._model.parameters(), lr=0.001)

    def get_h_values(self, states):
        states_as_list = [state.get_state_as_list() for state in states]
        states = np.array(states_as_list, dtype=np.float32)
        states_tensor = torch.tensor(states)
        if(type(self._model)==DeepCubeA):
            self._model.eval()
        with torch.no_grad():
            predictions = self._model(states_tensor).numpy()
        return predictions.flatten()

    def train_model(self, input_data, output_labels, epochs=100):
        input_as_list = [state.get_state_as_list() for state in input_data]
        inputs = np.array(input_as_list, dtype=np.float32)
        outputs = np.array(output_labels, dtype=np.float32)

        inputs_tensor = torch.tensor(inputs)
        outputs_tensor = torch.tensor(outputs).unsqueeze(1)  # Adding a dimension for the output

        for epoch in range(epochs):
            self._model.train()
            self._optimizer.zero_grad()

            predictions = self._model(inputs_tensor)
            loss = self._criterion(predictions, outputs_tensor)
            loss.backward()
            self._optimizer.step()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    def save_model(self, path):
        torch.save(self._model.state_dict(), path)

    def load_model(self, path):
        self._model.load_state_dict(torch.load(path))
        self._model.eval()

class BellmanUpdateHeuristic(LearnedHeuristic):
    def __init__(self, n=11, k=4):
        super().__init__(n, k, "DeepCubeA")
        self.load_model()

    def save_model(self):
        super().save_model('bellman_update_heuristic.pth')

    def load_model(self):
        super().load_model('bellman_update_heuristic.pth')
        self._model.eval()


class BootstrappingHeuristic(LearnedHeuristic): 
    def __init__(self, n=11, k=4):
        super().__init__(n, k)
        self.load_model()
        # 
    def save_model(self):
        super().save_model('bootstrapping_heuristic.pth')

    def load_model(self):
        super().load_model('bootstrapping_heuristic.pth')

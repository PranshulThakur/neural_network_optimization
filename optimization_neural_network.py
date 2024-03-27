import matplotlib.pyplot as plt
import torch
from objective_function import obj_func
import numpy as np

n_samples_1d = 20;
x_1d =  torch.linspace(0,1, n_samples_1d);
x_tensor = torch.Tensor(n_samples_1d**2, 2);
index = 0;
for i in range(n_samples_1d):
    for j in range(n_samples_1d):
        x_tensor[index][0] = x_1d[i].item();
        x_tensor[index][1] = x_1d[j].item();
        index += 1;
print(x_tensor);
print(x_tensor.unsqueeze(1));

# Define the neural network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(2, 64) # 2 input, 64 hidden neurons in layer 1
        self.hidden2 = torch.nn.Linear(64, 128) # 64 input, 128 hidden neurons in layer 2
        self.output = torch.nn.Linear(128, 1) # 128 input, 1 output

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

net = Net()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)  # Using Adam optimizer

for epoch in range(1000):
    running_loss = 0.0
    optimizer.zero_grad()
    outputs = net(x_tensor)
    print("Reached here.");
    print(outputs);
    loss = criterion(outputs, obj_func(x))
    loss.backward()
    optimizer.step()
    running_loss += loss.item()

    if epoch % 100 == 0:
        print("Epoch {}: Loss = {}".format(epoch, loss.detach().numpy()))


# Plot the actual function and predicted function
x_plot = torch.linspace(-1,1, 100)
actual_y = torch.tensor([obj_func(p) for p in x_plot])
predicted_y = net(x);
plt.plot(x_plot, actual_y, 'g*', label='Actual Function')
plt.plot(x, predicted_y.detach().numpy(), 'b', label='Predicted Function')
plt.legend()
plt.show()

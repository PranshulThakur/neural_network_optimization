import matplotlib.pyplot as plt
import torch

f = lambda x: torch.sin(3*x)

x = torch.linspace(-1,1, 35)

# Define the neural network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(1, 64) # 1 input, 64 hidden neurons in layer 1
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
    outputs = net(x.unsqueeze(1))
    loss = criterion(outputs.squeeze(), f(x))
    loss.backward()
    optimizer.step()
    running_loss += loss.item()

    if epoch % 100 == 0:
        print("Epoch {}: Loss = {}".format(epoch, loss.detach().numpy()))


# Plot the actual function and predicted function
x_plot = torch.linspace(-1,1, 100)
actual_y = torch.tensor([f(p) for p in x_plot])
predicted_y = net(x.unsqueeze(1)).squeeze()
plt.plot(x_plot, actual_y, 'g', label='Actual Function')
plt.plot(x, predicted_y.detach().numpy(), 'b', label='Predicted Function')
plt.legend()
plt.show()

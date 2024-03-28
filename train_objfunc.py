import numpy as np
import torch
from objective_function import obj_func

# Define the neural network for objective function
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        n_inputs = 2;
        n_layer1 = 50;
        n_layer2 = 100;
        n_layer3 = 50;
        n_outputs = 1;
        self.hidden1 = torch.nn.Linear(n_inputs, n_layer1) # 2 input, 64 hidden neurons in layer 1
        self.hidden2 = torch.nn.Linear(n_layer1, n_layer2) # 64 input, 128 hidden neurons in layer 2
        self.hidden3 = torch.nn.Linear(n_layer2, n_layer3) # 64 input, 128 hidden neurons in layer 2
        self.output = torch.nn.Linear(n_layer3, n_outputs) # 128 input, 1 output

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = self.output(x)
        return x


def get_trained_objfunc(): 

    # Samples to train the neural network.
    n_samples_1d = 20;
    x_1d =  torch.linspace(0.0,1.2, n_samples_1d);
    x_samples,y_samples = np.meshgrid(x_1d,x_1d);
    xy_tensor = torch.from_numpy(np.concatenate((x_samples.reshape(n_samples_1d**2,1), y_samples.reshape(n_samples_1d**2,1)),axis=1));
    f_exact = obj_func(xy_tensor).unsqueeze(1);

    f_neural_net = Net()
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(f_neural_net.parameters(), lr=0.001)  # Using Adam optimizer

    print("Training neural network of objective function");
    for epoch in range(10000):
        running_loss = 0.0
        optimizer.zero_grad()
        outputs = f_neural_net(xy_tensor)
        loss = loss_function(outputs, f_exact)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if epoch % 100 == 0:
            print("Epoch {}: Loss = {}".format(epoch, loss.detach().numpy()))

    print("Finished training neural netork of objective function");

    return f_neural_net;

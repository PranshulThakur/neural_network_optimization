import numpy as np
import torch
from objective_function import obj_func

# Neural network (NN) for objective function. 
# The NN has two inputs, three hidden layers and one output.
# torch.nn.Linear stores weights and biases, which are trained to minimize error with respect to the objective function.
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        n_inputs = 2;
        n_layer1 = 50;
        n_layer2 = 100;
        n_layer3 = 50;
        n_outputs = 1;
        # Specify layers of the neural network.
        self.hidden1 = torch.nn.Linear(n_inputs, n_layer1) # 2 inputs, n_layer1 neurons
        self.hidden2 = torch.nn.Linear(n_layer1, n_layer2) # n_layer1 inputs, n_layer2 neurons
        self.hidden3 = torch.nn.Linear(n_layer2, n_layer3) # n_layer2 inputs, n_layer3 neurons
        self.output_layer = torch.nn.Linear(n_layer3, n_outputs) # n_layer3 inputs, 1 output
        
        # Initialize weights and biases.
        torch.nn.init.eye_(self.hidden1.weight);
        torch.nn.init.ones_(self.hidden1.bias);
        torch.nn.init.eye_(self.hidden2.weight);
        torch.nn.init.ones_(self.hidden2.bias);
        torch.nn.init.eye_(self.hidden3.weight);
        torch.nn.init.ones_(self.hidden3.bias);
        torch.nn.init.eye_(self.output_layer.weight);
        torch.nn.init.ones_(self.output_layer.bias);

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = self.output_layer(x)
        return x


def get_trained_objfunc(): 

    # Data to train the neural network.
    n_samples_1d = 20;
    x_1d =  torch.linspace(0.0,1.2, n_samples_1d);
    x_samples,y_samples = np.meshgrid(x_1d,x_1d);
    xy_tensor = torch.from_numpy(np.concatenate((x_samples.reshape(n_samples_1d**2,1), y_samples.reshape(n_samples_1d**2,1)),axis=1));
    f_exact = obj_func(xy_tensor).unsqueeze(1);

    f_neural_net = Net()
    loss_function = torch.nn.MSELoss() # Using squared L2 norm of the error.
    optimizer = torch.optim.Adam(f_neural_net.parameters(), lr=0.005)  # Using Adam optimizer.

    print("Training neural network of objective function");
    for epoch in range(10000):
        optimizer.zero_grad()
        outputs = f_neural_net(xy_tensor)
        loss = loss_function(outputs, f_exact)
        loss.backward() # Backpropagation and computing gradients w.r.t. weights and biases.
        optimizer.step() # Update weights and biases.

        if epoch % 500 == 0:
            print("Epoch {}: Loss = {}".format(epoch, loss.detach().numpy()))

    print("Finished training neural network of objective function");

    return f_neural_net;

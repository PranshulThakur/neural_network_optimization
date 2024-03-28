import matplotlib.pyplot as plt
import torch
import numpy as np
from train_objfunc import get_trained_objfunc
from constraints import constraint_1, constraint_2
from objective_function import obj_func

# Define the neural network for optimization problem
class Net_optimizationproblem(torch.nn.Module):
    def __init__(self,f_neural_net, constraint):
        super(Net_optimizationproblem, self).__init__()
        self.f_neural_net = f_neural_net;
        for param in self.f_neural_net.parameters():
            param.requires_grad = False;
        self.constraint = constraint;
        self.linear = torch.nn.Linear(2,2);
        torch.nn.init.zeros_(self.linear.weight);
        torch.nn.init.zeros_(self.linear.bias);


    def forward(self, x):
        with torch.no_grad():
            self.linear.weight.data.copy_(torch.eye(2));

        x = self.linear(x);
        f_val = self.f_neural_net(x).squeeze();
        #f_val = obj_func(x);
        constraints_val = self.constraint(x); 

        output = f_val + 10*(torch.relu(constraints_val) + torch.relu(-constraints_val));
        #output = f_val + 5.0*(torch.tanh((constraints_val/2.0)))**2;
        return output;

#Custom loss function for optimization
class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predictions):
        return predictions;

objfunc_neural_net = get_trained_objfunc();

optimization_problem = Net_optimizationproblem(objfunc_neural_net, constraint_1);
loss_function = CustomLoss()
optimizer = torch.optim.NAdam(optimization_problem.parameters(), lr=0.01)  # Using Adam optimizer


print("Running optimization using neural network..");

x_initial = torch.Tensor(1,2);
x_initial[:,0] = 0.5;
x_initial[:,1] = 0.5;
print(x_initial.size());
for epoch in range(10000):
    running_loss = 0.0
    optimizer.zero_grad()
    outputs = optimization_problem(x_initial)
    loss = loss_function(outputs)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()

    if epoch % 100 == 0:
        print("Epoch {}: Loss = {}".format(epoch, loss.detach().numpy()))

print("Optimal point = ");
x_optimal = optimization_problem.linear(x_initial);
print(x_optimal);
"""
# Plot the actual function and predicted function
x_plot,y_plot = np.meshgrid(x_1d,x_1d);
actual_z = f_exact.squeeze();
predicted_z = f_neural_net(xy_tensor).squeeze();
actual_z = torch.reshape(actual_z, (n_samples_1d,n_samples_1d));
predicted_z = torch.reshape(predicted_z, (n_samples_1d,n_samples_1d));

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(x_plot,y_plot,actual_z);
ax.plot_surface(x_plot,y_plot,predicted_z.detach().numpy());
plt.show()
"""

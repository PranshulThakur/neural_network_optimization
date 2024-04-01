import matplotlib.pyplot as plt
import torch
import numpy as np
from train_objfunc import get_trained_objfunc
from constraints import constraint_1, constraint_2
from objective_function import obj_func

# Modified implementation of: Chen J, Liu Y. 2023 Neural optimization machine: a neural network approach for optimization and its application in additive manufacturing with physics-guided learning. Phil. Trans. R. Soc. A.

# Define the neural network for optimization problem
class Net_optimizationproblem(torch.nn.Module):
    def __init__(self,f_neural_net, constraint):
        super(Net_optimizationproblem, self).__init__()
        self.f_neural_net = f_neural_net;
        for param in self.f_neural_net.parameters():
            param.requires_grad = False;
        self.constraint = constraint;
        self.linear = torch.nn.Linear(2,2);
        torch.nn.init.eye_(self.linear.weight);
        torch.nn.init.zeros_(self.linear.bias);


    def forward(self, x):
        x = self.linear(x);
        f_val = self.f_neural_net(x).squeeze();
        #f_val = obj_func(x); # Can also use the analytical objective function.
        constraints_val = self.constraint(x); 

        output = f_val + 10*(torch.relu(constraints_val) + torch.relu(-constraints_val));
        return output;

#Custom loss function for optimization.
class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predictions):
        return predictions;


def run_optimizer(objfunc_neural_net, constraint):
    optimization_problem = Net_optimizationproblem(objfunc_neural_net, constraint);
    loss_function = CustomLoss()
    optimizer = torch.optim.NAdam(optimization_problem.parameters(), lr=0.01);

    x_initial = torch.Tensor(1,2);
    x_initial[:,0] = 0.5;
    x_initial[:,1] = 0.5;
    for epoch in range(10000):
        optimizer.zero_grad()
        outputs = optimization_problem(x_initial)
        loss = loss_function(outputs)
        loss.backward() # Backward propagation with automatic differentiation.
        optimizer.step()
        if epoch % 1000 == 0:
            print("Epoch {}: Loss = {}".format(epoch, loss.detach().numpy()))

    x_optimal = optimization_problem.linear(x_initial);
    print("Optimal point = ", x_optimal);
    return x_optimal;



print("===================================================================");
print("                 Training NN objective function                       ");
print("===================================================================");
objfunc_neuralnet = get_trained_objfunc();
print("===================================================================");
print('\n\n');
print("===================================================================");
print("                 Optimization using Neural Network                 ");
print("===================================================================");
print('\n');
print("===================================================================");
print("Running optimization for f(x,y) = (1-x)^2 + 100*(y-x^2)^2");
print("With constraint c1(x,y) = 1-x-y = 0");
print("===================================================================");
run_optimizer(objfunc_neuralnet, constraint_1);
print("===================================================================");
print('\n\n');
print("===================================================================");
print("Running optimization for f(x,y) = (1-x)^2 + 100*(y-x^2)^2");
print("With constraint c2(x,y) = 1-x^2-y^2 = 0");
print("===================================================================");
run_optimizer(objfunc_neuralnet, constraint_2);
print("===================================================================");


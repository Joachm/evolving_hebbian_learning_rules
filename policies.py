import torch
import torch.nn as nn

class MLP_heb(nn.Module):
    "MLP, no bias"
    def __init__(self, input_space, action_space):
        super(MLP_heb, self).__init__()

        self.fc1 = nn.Linear(input_space, 128, bias=False)
        self.fc2 = nn.Linear(128, 64, bias=False)
        self.fc3 = nn.Linear(64, action_space, bias=False)

    def forward(self, ob):

        state = torch.as_tensor(ob).float().detach()
       
        x1 = torch.tanh(self.fc1(state))   
        x2 = torch.tanh(self.fc2(x1))
        o = torch.tanh(self.fc3(x2))
         
        return state, x1, x2, o
        # return state, self.fc1(state), self.fc2(x1), self.fc3(x2)  

    def get_weights(self):
        return  nn.utils.parameters_to_vector(self.parameters()).detach()

    
class MLP_heb_med(nn.Module):
    "MLP, no bias"
    def __init__(self, input_space, action_space):
        super(MLP_heb_med, self).__init__()

        self.fc1 = nn.Linear(input_space, 64, bias=False)
        self.fc2 = nn.Linear(64, 32, bias=False)
        self.fc3 = nn.Linear(32, action_space, bias=False)

    def forward(self, ob):

        state = torch.as_tensor(ob).float().detach()
       
        x1 = torch.tanh(self.fc1(state))   
        x2 = torch.tanh(self.fc2(x1))
        o = torch.tanh(self.fc3(x2))
         
        return state, x1, x2, o
        # return state, self.fc1(state), self.fc2(x1), self.fc3(x2)  

    def get_weights(self):
        return  nn.utils.parameters_to_vector(self.parameters()).detach()

 

class MLP_heb_small(nn.Module):
    "MLP, no bias"
    def __init__(self, input_space, action_space):
        super(MLP_heb_small, self).__init__()

        self.fc1 = nn.Linear(input_space, 32, bias=False)
        self.fc2 = nn.Linear(32, 16, bias=False)
        self.fc3 = nn.Linear(16, action_space, bias=False)

    def forward(self, ob):

        state = torch.as_tensor(ob).float().detach()
       
        x1 = torch.tanh(self.fc1(state))   
        x2 = torch.tanh(self.fc2(x1))
        o = torch.tanh(self.fc3(x2))
         
        return state, x1, x2, o
        # return state, self.fc1(state), self.fc2(x1), self.fc3(x2)  

    def get_weights(self):
        return  nn.utils.parameters_to_vector(self.parameters()).detach()

 

class MLP_heb_tiny(nn.Module):
    "MLP, no bias"
    def __init__(self, input_space, action_space):
        super(MLP_heb_tiny, self).__init__()

        self.fc1 = nn.Linear(input_space, 8, bias=False)
        self.fc2 = nn.Linear(8, 8, bias=False)
        self.fc3 = nn.Linear(8, action_space, bias=False)

    def forward(self, ob):

        state = torch.as_tensor(ob).float().detach()
       
        x1 = torch.tanh(self.fc1(state))   
        x2 = torch.tanh(self.fc2(x1))
        o = torch.tanh(self.fc3(x2))
         
        return state, x1, x2, o
        # return state, self.fc1(state), self.fc2(x1), self.fc3(x2)  

    def get_weights(self):
        return  nn.utils.parameters_to_vector(self.parameters()).detach()

 


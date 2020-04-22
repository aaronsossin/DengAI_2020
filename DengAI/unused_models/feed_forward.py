import torch


class feed_forward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(feed_forward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.linear1 = torch.nn.Linear()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.linear2 = torch.nn.Linear()

    def forward(self, x):
        hidden = self.fc1(x)
        linear = self.linear1(hidden)
        output = self.fc2(linear)
        output = self.linear2(output)
        return output


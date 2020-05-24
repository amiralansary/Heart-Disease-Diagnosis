"""This file contains different pytorch models"""
# Copyright (C) 2020 Amir Alansary <amiralansary@gmail.com>
#

import torch.nn as nn
import torch.nn.functional as F


###############################################################################
# Linear Regression
###############################################################################
class LinearRegression(nn.Module):
    def __init__(self, input_size, n_classes):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, n_classes)

    def forward(self, x):
        x = self.linear(x)
        return x


###############################################################################
# Logistic Regression
###############################################################################
class LogisticRegression(nn.Module):
    def __init__(self, input_size, n_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, n_classes)

    def forward(self, x):
        x = self.linear(x)
        # x = F.sigmoid(x)
        return x


###############################################################################
# Neural network
###############################################################################
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], n_classes)

        self.drop_layer = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = self.drop_layer(x)
        x = F.relu(self.fc2(x))
        # x = self.drop_layer(x)
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=1)
        return x


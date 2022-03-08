import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#SHOW DEVICE GPU/CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class CNN_classifier(nn.Module):


    def __init__(self):
        super(CNN_classifier,self).__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
        )

    def forward(self,x ):
        x = self.flatten(x)
        prediction = self.stack(x)
        return prediction

model = CNN_classifier()
# print(model)

# X = torch.rand(1, 28, 28)
# logits = model(X)
# pred_probab = nn.Softmax(dim=1)(logits)
# y_pred = pred_probab.argmax(1)
# print(f"Predicted class: {y_pred}")


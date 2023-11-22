import torch
from torch import nn 

class tinyVGGModel(nn.Module):
    def __init__(self, input_features:int, hidden_units:int, output_features:int):
        super().__init__()

        self.cnn_block1 = nn.Sequential(
            nn.Conv2d(in_channels= input_features, out_channels= hidden_units, stride= 1, 
                    padding= 0, kernel_size= 3),
            nn.ReLU(),
            nn.Conv2d(in_channels= input_features, out_channels= hidden_units, stride= 1, 
                    padding= 0, kernel_size= 3),
            nn.ReLU(),
            
            nn.MaxPool2d(kernal_size = 2, stride = 2)
        )

        self.cnn_block2 = nn.Sequential(
            nn.Conv2d(in_channels= input_features, out_channels= hidden_units, stride= 1, 
                    padding= 0, kernel_size= 3),
            nn.ReLU(),
            nn.Conv2d(in_channels= input_features, out_channels= hidden_units, stride= 1, 
                    padding= 0, kernel_size= 3),
            nn.ReLU(),
            nn.MaxPool2d(kernal_size = 2, stride = 2)
        )

        self.linear_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features= hidden_units*13*13, out_features= output_features)
        )
    
    def forward(self, x: torch.Tensor)->torch.Tensor:
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        x = self.linear_layer(x)
        return x
        
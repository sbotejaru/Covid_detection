from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Layers
        self.conv_layer1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv_layer2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv_layer3 = nn.Conv2d(16, 32, 3, padding=1)
        self.pooling = nn.MaxPool2d(2)

        self.final_layer1 = nn.Linear(2048, 128)
        self.final_layer2 = nn.Linear(128, 3)

        # Activations
        self.activation_hidden = nn.LeakyReLU()
        self.activation_final = nn.LogSoftmax(dim=1)

        # Batch norm
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)

        # Dropout
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        # Feature extraction
        x = self.conv_layer1(x)
        x = self.bn1(x)
        x = self.activation_hidden(x)
        x = self.pooling(x)

        x = self.conv_layer2(x)
        x = self.bn2(x)
        x = self.activation_hidden(x)
        x = self.pooling(x)

        x = self.conv_layer3(x)
        x = self.bn3(x)
        x = self.activation_hidden(x)
        x = self.pooling(x)

        # Classification
        x = x.view(x.shape[0], -1)

        x = self.final_layer1(x)
        x = self.activation_hidden(x)
        x = self.drop(x)

        x = self.final_layer2(x)
        x = self.activation_final(x)

        return x
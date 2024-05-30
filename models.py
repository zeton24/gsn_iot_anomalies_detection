import torch
nn = torch.nn


class Model1D(nn.Module):
    def __init__(self, binary_classification=False):
        super().__init__()
        self.num_classes = 2 if binary_classification else 5
        self.convblock1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.AvgPool1d(2),
            nn.Dropout(p=0.05)
        )

        self.c1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding='same')
        self.relu1 = nn.ReLU()
        self.norm1 = nn.BatchNorm1d(32)
        self.pool1 = nn.AvgPool1d(2)
        self.drop1 = nn.Dropout(p=0.05)

        self.convblock2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.AvgPool1d(2),
            nn.Dropout(p=0.05)
        )

        self.convblock3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.AvgPool1d(2),
            nn.Dropout(p=0.05)
        )

        self.convblock4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.AvgPool1d(2),
            nn.Dropout(p=0.05)
        )

        self.evaluator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.evaluator(x)
        return x


class Model2D(nn.Module):
    def __init__(self, binary_classification=False):
        super().__init__()
        self.num_classes = 2 if binary_classification else 5
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.AvgPool2d(2),
            nn.Dropout(p=0.05)
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(2),
            nn.Dropout(p=0.05)
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(2),
            nn.Dropout(p=0.05)
        )

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Dropout(p=0.05)
        )

        self.evaluator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], 1, 8, 8))
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.evaluator(x)
        return x


class Model3D(nn.Module):
    def __init__(self, binary_classification=False):
        super().__init__()
        self.num_classes = 2 if binary_classification else 5
        self.convblock1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=5, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(32),
            nn.AvgPool3d(2),
            nn.Dropout(p=0.05)
        )

        self.convblock2 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=5, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(64),
            nn.AvgPool3d(2),
            nn.Dropout(p=0.05)
        )

        self.convblock3 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=5, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(128),
            nn.Dropout(p=0.05)
        )

        self.convblock4 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=5, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(256),
            nn.Dropout(p=0.05)
        )

        self.evaluator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], 1, 4, 4, 4))
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.evaluator(x)
        return x

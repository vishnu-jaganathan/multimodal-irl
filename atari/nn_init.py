import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class LinearNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(LinearNN, self).__init__()

        self.weights = nn.Sequential()
        if len(hidden_dims):
            for i in range(len(hidden_dims)+1):
                if i==0:
                    self.weights.append(nn.Linear(input_dim, hidden_dims[i]))
                elif i==len(hidden_dims):
                    self.weights.append(nn.Linear(hidden_dims[i-1],output_dim))
                else:
                    self.weights.append(nn.Linear(hidden_dims[i-1],hidden_dims[i]))

    def forward(self, x):
        reward = self.weights(x)
        return reward


# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # Input: 2 x 250 x 160
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1, stride=2),
            nn.ReLU(),

            # 64 x 125 x 80
            nn.Conv2d(64, 64, 3, padding=1, stride=2),
            nn.ReLU(),

            # 64 x 62 x 40
            nn.Conv2d(64, 64, 3, padding=1, stride=2),
            nn.ReLU(),

            # 64 x 31 x 20
            nn.Conv2d(64, 1, 3, padding=1, stride=2),
            nn.ReLU(),

            # 1 x 15 x 10
            nn.Flatten(),
        )
        # Output: 150 x 1

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        # Input: 150 x 1
        self.decoder = nn.Sequential(
            nn.Unflatten(0,(1,15,10)),

            # 1 x 15 x 10
            nn.ConvTranspose2d(1, 64, 3, output_padding=(2,1), padding=1, stride=2),
            nn.ReLU(),

            # 64 x 31 x 20
            nn.ConvTranspose2d(1, 64, 3, output_padding=1, padding=1, stride=2),
            nn.ReLU(),

            # 64 x 62 x 40
            nn.ConvTranspose2d(64, 64, 3, output_padding=(2,1), padding=1, stride=2),
            nn.ReLU(),

            # 64 x 125 x 80
            nn.ConvTranspose2d(64, 2, 3, output_padding=1, padding=1, stride=2),
            nn.Tanh(),
        )

        # Output: 2 x 250 x 160

    def forward(self, x):
        return 255*self.decoder(x)


class Autoencoder(pl.LightningModule):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.save_hyperparameters()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded_state = self.encoder(x)
        x_hat = self.decoder(encoded_state)
        return x_hat
    
    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x, _ = batch # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.2,
                                                         patience=20,
                                                         min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss)

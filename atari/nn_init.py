import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

class LinearNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(LinearNN, self).__init__()

        self.weights = nn.Sequential()
        if len(hidden_dims) > 0:
            for i in range(len(hidden_dims)+1):
                if i==0:
                    self.weights.append(nn.Linear(input_dim, hidden_dims[i]))
                elif i==len(hidden_dims):
                    self.weights.append(nn.Linear(hidden_dims[i-1], output_dim))
                else:
                    self.weights.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        else:
            self.weights.append(nn.Linear(input_dim, output_dim))

    def forward(self, x):
        reward = self.weights(x)
        return reward



# Autoencoder
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # Input: 2 x 250 x 160
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),

            # 64 x 125 x 80
            nn.Conv2d(64, 64, kernel_size=3, padding=(0,1), stride=2),
            nn.ReLU(),

            # 64 x 62 x 40
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),

            # 128 x 31 x 20
            nn.Conv2d(64, 1, kernel_size=3, padding=(0,1), stride=2),
            nn.ReLU(),

            # 1 x 15 x 10
            nn.Flatten(),
        )
        # Output: 1 x 150

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        # Input: 1 x 150
        self.decoder = nn.Sequential(
            nn.Unflatten(-1,(15,10)),

            # 1 x 15 x 10
            nn.ConvTranspose2d(1, 64, kernel_size=3, output_padding=(0,1), padding=(0,1), stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            # 64 x 31 x 20
            nn.ConvTranspose2d(64, 64, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            # 64 x 62 x 40
            nn.ConvTranspose2d(64, 64, kernel_size=3, output_padding=(0,1), padding=(0,1), stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            # 64 x 125 x 80
            nn.ConvTranspose2d(64, 2, kernel_size=3, output_padding=1, padding=1, stride=2),
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

def main():
    encoder = Encoder()
    decoder = Decoder()

    input = torch.randn(2,260,150)
    print("\n\nInput:", tuple(input.size()), "\n\n")
    summary(encoder, input_size=(2, 250, 160))
    encoded_state = encoder(input)
    print("\n\nEncoded State:", tuple(encoded_state.size()), "\n\n")
    summary(decoder, input_size=encoded_state.size())
    decoded_state = decoder(encoded_state)
    print("\n\nDecoded State:", tuple(decoded_state.size()), "\n\n")

if __name__ == '__main__':
    main()
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
import torchvision
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


        # Input: 2 x 210 x 160
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),

            # 64 x 105 x 80
            nn.Conv2d(32, 32, kernel_size=3, padding=(0,1), stride=2),
            nn.ReLU(),

            # 64 x 52 x 40
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),

            # 128 x 26 x 20
            nn.Conv2d(32, 1, kernel_size=3, padding=(0,1), stride=2),
            nn.ReLU(),

            # 1 x 12 x 10
            nn.Flatten(),
        )
        # Output: 1 x 150

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        # Input: 1 x 120
        self.unflatten = nn.Sequential(nn.Unflatten(-1,(12,10)))
        self.decoder = nn.Sequential(
            # 1 x 12 x 10
            nn.ConvTranspose2d(1, 32, kernel_size=3, output_padding=1, padding=(0,1), stride=2),
            nn.ReLU(),

            # 64 x 26 x 20
            nn.ConvTranspose2d(32, 32, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.ReLU(),
            
            # 64 x 52 x 40
            nn.ConvTranspose2d(32, 32, kernel_size=3, output_padding=(0,1), padding=(0,1), stride=2),
            nn.ReLU(),

            # 64 x 105 x 80
            nn.ConvTranspose2d(32, 2, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.Sigmoid(),
        )

        # Output: 2 x 210 x 160

    def forward(self, x):
        x = self.unflatten(x)
        x = x.reshape(x.shape[0], -1, 12, 10)
        return 255*self.decoder(x)


class Autoencoder(pl.LightningModule):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.save_hyperparameters()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.example_input_array = torch.zeros(2,2,210,150)

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
                                                         patience=5,
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

class GenerateCallback(pl.Callback):

    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs # Images to reconstruct during training
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0,1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1,1))
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)


def main():
    encoder = Encoder()
    decoder = Decoder()
    autoencoder = Autoencoder()

    input = torch.randn(2,210,150)
    print("\n\nInput:", tuple(input.size()), "\n\n")
    summary(encoder, input_size=input.size())
    encoded_state = encoder(input)
    print("\n\nEncoded State:", tuple(encoded_state.size()), "\n\n")
    summary(decoder, input_size=encoded_state.size())
    decoded_state = decoder(encoded_state)
    print("\n\nDecoded State:", tuple(decoded_state.size()), "\n\n")

    print("\n\nInput:", tuple(input.size()), "\n\n")
    summary(autoencoder, input_size=input.size())
    

if __name__ == '__main__':
    main()
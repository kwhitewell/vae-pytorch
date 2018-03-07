import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np

from vae import Encoder, Decoder

save_dir = "model"
data_dir = "data"

# calculate kullback leibler divergence
def kl_divergence(
  mu: Variable,
  var: Variable,
) -> Variable:
  return torch.sum(
    1 + torch.log(var.pow(2)) - mu.pow(2) - var.pow(2),
    dim=1
  ) / 2


# train the encoder and the decoder for one epoch
def train(
  encoder,
  decoder,
  opt_enc, 
  opt_dec,
  train_loader,
  epoch: int,
  batch_size: int,
  guid: int,
) -> None:

  epoch_kl = 0.
  epoch_mse = 0.

  for batch_i, (x, _) in enumerate(train_loader):
    # datapoints
    x = Variable(
      x.view(batch_size, -1)
    )

    # noise variables
    e = Variable(
      torch.from_numpy(
        np.random.normal(0, 1, (batch_size, encoder.n_z))
      ).float()
    )

    if guid >= 0:
      x = x.cuda(guid)
      e = e.cuda(guid)

    # encode datapoints and predict mu and var
    mu, var = encoder(x)
   
    # reparametrization trick
    z = mu + var * e 

    # deocde z into x
    x_bar = decoder(z)

    kl = torch.sum(kl_divergence(mu, var)) / (batch_size * encoder.n_in)
    mse = F.mse_loss(x_bar, x, size_average=True)
    #mse = F.binary_cross_entropy(x_bar, x, size_average=True)

    loss = -kl + mse

    # backward and optimize parameters
    opt_enc.zero_grad()
    opt_dec.zero_grad()
    loss.backward()
    opt_enc.step()
    opt_dec.step()

    epoch_kl += kl.data.tolist()[0]
    epoch_mse += mse.data.tolist()[0]

    print("epoch {:<3}, batch {:<4}, kl-div {:<10}, mse {:<9}".format(
      epoch,
      batch_i + 1,
      round(kl.data.tolist()[0], 7),
      round(mse.data.tolist()[0], 7),
    ))

  print("finished epoch {}, epoch-kl-div {}, epoch-mse {}\n".format(
    epoch,
    epoch_kl,
    epoch_mse,
  ))


def main():
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("-z", "--n_z", type=int, default=100, help="n_z")
  parser.add_argument("-f", "--n_f", type=int, default=500, help="n_f")
  parser.add_argument("-b", "--batch_size", type=int, default=100, help="batch_size")
  parser.add_argument("-e", "--epoch_size", type=int, default=100, help="epoch_size")
  parser.add_argument("-g", "--guid", type=int, default=-1, help="gpu id")
  args = parser.parse_args()

  n_in = n_out = 28 * 28
  n_f = args.n_f
  n_z = args.n_z

  encoder = Encoder(
    n_in, 
    n_f, 
    n_z,
  )

  decoder = Decoder(
    n_z, 
    n_f,
    n_out,
  )

  if args.guid >= 0:
    encoder = encoder.cuda(args.guid)
    decoder = decoder.cuda(args.guid)

  opt_enc = optim.Adam(encoder.parameters())
  opt_dec = optim.Adam(decoder.parameters())

  train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(data_dir, train=True, download=True,
                   transform=transforms.Compose([
                     transforms.ToTensor(),
                     #transforms.Normalize((0.1307,), (0.3081,)),
                   ]),
    ),
    batch_size=args.batch_size, shuffle=True,
  )

  encoder.train()
  decoder.train()

  for epoch in range(1, args.epoch_size + 1):
    train(
      encoder,
      decoder,
      opt_enc,
      opt_dec,
      train_loader,
      epoch,
      args.batch_size,
      args.guid,
    )

    torch.save(
      encoder,
      "{}/encoder_epoch_{}".format(save_dir, epoch),
    )

    torch.save(
      decoder,
      "{}/decoder_epoch_{}".format(save_dir, epoch),
    )


if __name__ == "__main__":
  main()

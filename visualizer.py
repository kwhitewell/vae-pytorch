import torch
from torch.autograd import Variable

import numpy as np
from scipy.misc import imsave

save_dir = "test"

def visualizer(
  decoder,
  n_samples: int,
) -> None:
  
  z = Variable(
    torch.from_numpy(
      np.random.normal(0, 1, (n_samples, decoder.n_z)),
    ).float()
  )

  x_bar = decoder(z)
  for i, image in enumerate(x_bar.data.numpy()):
    imsave(
      "{}/test_{}.png".format(save_dir, i + 1), 
      image.reshape(28, 28),
    )
    


def main():
  import os
  import argparse
  
  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--model_dir", type=str, default=None, help="model directory")
  parser.add_argument("-n", "--n_samples", type=int, default=100, help="number of samples")
  args = parser.parse_args()

  if os.path.exists(args.model_dir) is False:
    raise Exception("invalid model directory {}".format(args.model_dir))

  model = torch.load(
    args.model_dir,
    map_location=lambda x, loc: x,
  )

  model.eval()
    
  visualizer(
    model,
    args.n_samples,
  )


if __name__ == "__main__":
  main()

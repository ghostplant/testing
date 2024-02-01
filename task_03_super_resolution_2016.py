import torch
import torchvision

import os
import autort

device = os.environ.get('DEVICE', autort.device())

sr_weight = torch.load('super_resolution_weights/super-resolution-10.pt')
sr_weight = dict([(x, sr_weight[x].to(device)) for x in sr_weight])

conv0, w0 = sr_weight['Conv_9.weight'], sr_weight['Conv_9.bias']
conv1, w1 = sr_weight['Conv_11.weight'], sr_weight['Conv_11.bias']
conv2, w2 = sr_weight['Conv_13.weight'], sr_weight['Conv_13.bias']
conv3, w3 = sr_weight['Conv_15.weight'], sr_weight['Conv_15.bias']


def model(x):
  H, W = x.size(-2), x.size(-1)
  y = x.view(3, 1, H, W)
  y = torch.nn.functional.conv2d(y, conv0, w0, 1, 2, 1, 1)
  y = torch.nn.functional.relu(y)
  y = torch.nn.functional.conv2d(y, conv1, w1, 1, 1, 1, 1)
  y = torch.nn.functional.relu(y)
  y = torch.nn.functional.conv2d(y, conv2, w2, 1, 1, 1, 1)
  y = torch.nn.functional.relu(y)
  y = torch.nn.functional.conv2d(y, conv3, w3, 1, 1, 1, 1)
  y = y.view(-1, 3, 3, H, W)
  y = y.permute(0, 3, 1, 4, 2).contiguous().view(-1, H * 3, W * 3)
  y = (torch.clip(y.squeeze(1), 0, 1) * 256)
  return y

x = (torchvision.io.read_image('samurai_nn.png') / 255.0).to(device)
y = model(x)

torchvision.io.write_jpeg(y.cpu().to(torch.uint8), 'samurai_nn_sr2016.jpg', 100)
print('Image saved to ./samurai_nn_sr2016.jpg')

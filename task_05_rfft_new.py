import torch
import autort
import os


##########################################################################
os.environ['LOG'] = '1'

fn = autort.export(name='rfft_f32', source=r'''
@DEF_FUNC: input:float32[N, C, H, W] -> output:float32[2, N, C, H, WO]

void main() {
  // TODO:

  for (int wo = 0; wo < size_of_WO(); ++wo)
    for (int h = 0; h < size_of_H(); ++h)
      for (int n = 0; n < size_of_N(); ++n)
        for (int c = 0; c < size_of_C(); ++c)
          output(0, n, c, h, wo) = 123;

  for (int wo = 0; wo < size_of_WO(); ++wo)
    for (int h = 0; h < size_of_H(); ++h)
      for (int n = 0; n < size_of_N(); ++n)
        for (int c = 0; c < size_of_C(); ++c)
          output(1, n, c, h, wo) = 456;
}
''')
##########################################################################

device = autort.device()
torch.manual_seed(0)
x = torch.randn([1, 1, 6, 6]).to(device)

output_cudnn = torch.fft.rfftn(x, dim=(2, 3))
print('CUDNN output.real:', output_cudnn.real)
print('CUDNN output.imag:', output_cudnn.imag)
print()

output_custom = torch.zeros([2, 1, 1, 6, 4], dtype=torch.float32, device=device)
autort.ops.rfft_f32(x, output_custom)

print('CUSTOM output.real', output_custom[0, :])
print('CUSTOM output.imag', output_custom[1, :])
print()


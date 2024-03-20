import torch
import autort
import os


##########################################################################
with open('/tmp/rfft_f32.cxx', 'w') as fp:
  fp.write(r'''
@DEF_FUNC: input:float32[N, C, H, W], temp:float32[2, H] -> output:float32[2, N, C, H, WO]

void main() {
    
  float PI = 3.1415926535897932;
        
  for (int n = 0; n < size_of_N(); ++n) {
    for (int c = 0; c < size_of_C(); ++c) {
      for (int h = 0; h < size_of_H(); ++h) {
        for (int wo = 0; wo < size_of_WO(); ++wo) {
          output(0, n, c, h, wo) = 0;
          output(1, n, c, h, wo) = 0;
        }
      }
    }
  }
  
  for (int n = 0; n < size_of_N(); ++n) {
    for (int c = 0; c < size_of_C(); ++c) {
      for (int h = 0; h < size_of_H(); ++h) {
        for (int w = 0; w < size_of_W(); ++w) {
          for (int wo = 0; wo < size_of_WO(); ++wo) {
            output(0, n, c, h, wo) += input(n, c, h, w) * cos(2.0 * PI * w * wo / size_of_W());
            output(1, n, c, h, wo) -= input(n, c, h, w) * sin(2.0 * PI * w * wo / size_of_W());
          }
        }
      }
    }
  }
  
  for (int n = 0; n < size_of_N(); ++n) {
    for (int c = 0; c < size_of_C(); ++c) {
      for (int wo = 0; wo < size_of_WO(); ++wo) {
        for (int h = 0; h < size_of_H(); ++h) {
          temp(0, h) = 0;
          temp(1, h) = 0;
        }
        for (int h = 0; h < size_of_H(); ++h) {
          for (int k = 0; k < size_of_H(); ++k) {
            temp(0, h) += output(0, n, c, k, wo) * cos(2.0 * PI * h * k / size_of_H());
            temp(0, h) += output(1, n, c, k, wo) * sin(2.0 * PI * h * k / size_of_H());
            temp(1, h) += output(1, n, c, k, wo) * cos(2.0 * PI * h * k / size_of_H());
            temp(1, h) -= output(0, n, c, k, wo) * sin(2.0 * PI * h * k / size_of_H());
          }
        }
        for (int h = 0; h < size_of_H(); ++h) {
          output(0, n, c, h, wo) = temp(0, h);
          output(1, n, c, h, wo) = temp(1, h);
        }
      }
    }
  }
  
}
''')
assert 0 == os.system('autort -n /tmp/rfft_f32.cxx >/dev/null')
##########################################################################

device = autort.device()
torch.manual_seed(0)
x = torch.randn([4, 8, 16, 16]).to(device)
temp = torch.zeros([x.size(-2)], dtype=x.dtype, device=device)

output_cudnn = torch.fft.rfftn(x, dim=(2, 3))
print('CUDNN output.real:', output_cudnn.real)
print('CUDNN output.imag:', output_cudnn.imag)
print()

output_custom = torch.zeros([2, 4, 8, 16, 9], dtype=torch.float32, device=device)
autort.ops.rfft_f32(x, temp, output_custom)

print('CUSTOM output.real', output_custom[0, :])
print('CUSTOM output.imag', output_custom[1, :])

real_diff = torch.abs(output_cudnn.real - output_custom[0, :]).max().item()
imag_diff = torch.abs(output_cudnn.imag - output_custom[1, :]).max().item()

print('Max diff (real):', real_diff)
print('Max diff (imag):', imag_diff)

import torch
import autort
import os


##########################################################################
os.environ['LOG'] = '1'

fn_irfft = autort.export(name='irfftn_f32', source=r'''
@DEF_FUNC: input:float32[2, N, C, H, W], temp_w:float32[2, WO], temp_h:float32[2, H] -> output:float32[N, C, H, WO]

void main() {
    
  float PI = 3.1415926535897932;
  int m = size_of_H();
  
  // ifft on the second-to-last dimension
  for (int n = 0; n < size_of_N(); ++n) {
    for (int c = 0; c < size_of_C(); ++c) {
      for (int w = 0; w < size_of_W(); ++w) {
        
        // write to temp and initialize
        for (int h = 0; h < size_of_H(); ++h) {
          temp_h(0, h) = input(0, n, c, h, w);
          temp_h(1, h) = input(1, n, c, h, w);
        }
        
        int i, j, k;
        for (i = 1, j = m / 2; i < m - 1; ++i) {
          if (i < j) {
            float t = temp_h(0, i);
            temp_h(0, i) = temp_h(0, j);
            temp_h(0, j) = t;
            t = temp_h(1, i);
            temp_h(1, i) = temp_h(1, j);
            temp_h(1, j) = t;
          }
          k = m / 2;
          while (j >= k) {
            j = j - k;
            k = k / 2;
          }
          if (j < k) {
            j += k;
          }
        }
        
        for (int x = 2; x <= m; x = x * 2) {
          float temp_real = cos(2 * PI / x);
          float temp_imag = sin(2 * PI / x);
          for (int j = 0; j < m; j += x) {
            float w_real = 1;
            float w_imag = 0;
            for (int k = j; k < j + x / 2; ++k) {
              float u_real = temp_h(0, k);
              float u_imag = temp_h(1, k);
              float t_real = w_real * temp_h(0, k + x / 2) - w_imag * temp_h(1, k + x / 2);
              float t_imag = w_real * temp_h(1, k + x / 2) + w_imag * temp_h(0, k + x / 2);
              temp_h(0, k) = u_real + t_real;
              temp_h(1, k) = u_imag + t_imag;
              temp_h(0, k + x / 2) = u_real - t_real;
              temp_h(1, k + x / 2) = u_imag - t_imag;
              float w_tmp = w_real;
              w_real = w_real * temp_real - w_imag * temp_imag;
              w_imag = w_tmp * temp_imag + w_imag * temp_real;
            }
          }
        }
        
        // write the result back
        for (int h = 0; h < m; ++h) {
          input(0, n, c, h, w) = temp_h(0, h) / m;
          input(1, n, c, h, w) = temp_h(1, h) / m;
        }
      }
    }
  }
  
  // irfft on the last dimension
  
  m = size_of_WO();
  
  for (int n = 0; n < size_of_N(); ++n) {
    for (int c = 0; c < size_of_C(); ++c) {
      for (int h = 0; h < size_of_H(); ++h) {
        
        // write to temp and initialize
        for (int w = 0; w < size_of_W(); ++w) {
          temp_w(0, w) = input(0, n, c, h, w);
          temp_w(1, w) = input(1, n, c, h, w);
        }
        for (int w = size_of_W(); w < size_of_WO(); ++w) {
          temp_w(0, w) = temp_w(0, size_of_WO() - w);
          temp_w(1, w) = -temp_w(1, size_of_WO() - w);
        }
        
        int i, j, k;
        for (i = 1, j = m / 2; i < m - 1; ++i) {
          if (i < j) {
            float t = temp_w(0, i);
            temp_w(0, i) = temp_w(0, j);
            temp_w(0, j) = t;
            t = temp_w(1, i);
            temp_w(1, i) = temp_w(1, j);
            temp_w(1, j) = t;
          }
          k = m / 2;
          while (j >= k) {
            j = j - k;
            k = k / 2;
          }
          if (j < k) {
            j += k;
          }
        }
        
        for (int x = 2; x <= m; x = x * 2) {
          float temp_real = cos(2 * PI / x);
          float temp_imag = sin(2 * PI / x);
          for (int j = 0; j < m; j += x) {
            float w_real = 1;
            float w_imag = 0;
            for (int k = j; k < j + x / 2; ++k) {
              float u_real = temp_w(0, k);
              float u_imag = temp_w(1, k);
              float t_real = w_real * temp_w(0, k + x / 2) - w_imag * temp_w(1, k + x / 2);
              float t_imag = w_real * temp_w(1, k + x / 2) + w_imag * temp_w(0, k + x / 2);
              temp_w(0, k) = u_real + t_real;
              temp_w(1, k) = u_imag + t_imag;
              temp_w(0, k + x / 2) = u_real - t_real;
              temp_w(1, k + x / 2) = u_imag - t_imag;
              float w_tmp = w_real;
              w_real = w_real * temp_real - w_imag * temp_imag;
              w_imag = w_tmp * temp_imag + w_imag * temp_real;
            }
          }
        }
        
        // write the result back
        for (int w = 0; w < size_of_WO(); ++w) {
          output(n, c, h, w) = temp_w(0, w) / size_of_WO();
        }
      }
    }
  }
}
''')
##########################################################################

device = autort.device()
torch.manual_seed(0)
x = torch.randn([1, 1, 8, 8]).to(device)
y = torch.fft.rfftn(x, dim=(2, 3)) # [1, 1, 8, 5], complex32
output_cudnn = torch.fft.irfftn(y, dim=(2,3)) # [1, 1, 8, 8], float32

print(torch.max(torch.abs(output_cudnn - x)))

temp_w = torch.zeros([2, 8], dtype=torch.float32, device=device) # [2, 8]
temp_h = torch.zeros([2, 8], dtype=torch.float32, device=device) # [2, 8]

z = torch.zeros([2, 1, 1, 8, 5], dtype=torch.float32, device=device)
for i in range(2):
    z[i, 0, 0, :, :] = y.real if i == 0 else y.imag

print(z)

output_irfftn = torch.zeros([1, 1, 8, 8], dtype=torch.float32, device=device)
autort.ops.irfftn_f32(z, temp_w, temp_h, output_irfftn)

print(output_irfftn)
# print(z)
# print(temp_h)
# print(torch.fft.ifft(y, dim=-2))

# print(torch.max(torch.abs(temp_output - torch.fft.ifft(y, dim=-2))))

diff = torch.max(torch.abs(output_cudnn - output_irfftn))
print('Max diff: ', diff)

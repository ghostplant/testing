import torch
import autort
import os


##########################################################################
os.environ['LOG'] = '1'

fn_rfft = autort.export(name='rfftn_f32', source=r'''
@DEF_FUNC: input:float32[N, C, H, W], temp_w:float32[2, W], temp_h:float32[2, H], order_w:int32[W], order_h:int32[H] -> output:float32[2, N, C, H, WO]

void main() {
    
  float PI = 3.1415926535897932;
  int m = size_of_W();
  
  // rfft on the last dimension
  for (int n = 0; n < size_of_N(); ++n) {
    for (int c = 0; c < size_of_C(); ++c) {
      for (int h = 0; h < size_of_H(); ++h) {
        
        // write to temp and initialize
        for (int w = 0; w < size_of_W(); ++w) {
          temp_w(0, w) = input(n, c, h, w);
          temp_w(1, w) = 0;
        }
        
        for (int i = 0; i < size_of_W(); ++i) {
          // swap x[i] and x[order[i]]
          if (i < order_w(i)) {
            float t = temp_w(0, i);
            temp_w(0, i) = temp_w(0, order_w(i));
            temp_w(0, order_w(i)) = t;
            // no need to swap the imaginary part because it is always 0
          }
        }
        
        for (int x = 2; x <= m; x = x * 2) {
          float temp_real = cos(2 * PI / x);
          float temp_imag = sin(-2 * PI / x);
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
        for (int wo = 0; wo < size_of_WO(); ++wo) {
          output(0, n, c, h, wo) = temp_w(0, wo);
          output(1, n, c, h, wo) = temp_w(1, wo);
        }
      }
    }
  }
  
  // fft on the second-to-last dimension
  
  m = size_of_H();
  
  for (int n = 0; n < size_of_N(); ++n) {
    for (int c = 0; c < size_of_C(); ++c) {
      for (int w = 0; w < size_of_WO(); ++w) {
        
        // write to temp and initialize
        for (int h = 0; h < size_of_H(); ++h) {
          temp_h(0, h) = output(0, n, c, h, w);
          temp_h(1, h) = output(1, n, c, h, w);
        }
        
        for (int i = 0; i < size_of_H(); ++i) {
          // swap x[i] and x[order[i]]
          if (i < order_h(i)) {
            float t = temp_h(0, i);
            temp_h(0, i) = temp_h(0, order_h(i));
            temp_h(0, order_h(i)) = t;
            t = temp_h(1, i);
            temp_h(1, i) = temp_h(1, order_h(i));
            temp_h(1, order_h(i)) = t;
          }
        }
        
        for (int x = 2; x <= m; x = x * 2) {
          float temp_real = cos(2 * PI / x);
          float temp_imag = sin(-2 * PI / x);
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
        for (int h = 0; h < size_of_H(); ++h) {
          output(0, n, c, h, w) = temp_h(0, h);
          output(1, n, c, h, w) = temp_h(1, h);
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
output_cudnn = torch.fft.rfftn(x, dim=(2,3))

print(output_cudnn.real)
print(output_cudnn.imag)

temp_w = torch.zeros([2, x.size(-1)], dtype=torch.float32, device=device)
temp_h = torch.zeros([2, x.size(-2)], dtype=torch.float32, device=device)

W, H = x.size(-1), x.size(-2)
order_w = [0] * W
for i in range(W):
  order_w[i] = order_w[i >> 1] >> 1
  if i & 1:
    order_w[i] |= W >> 1
    
order_h = [0] * H
for i in range(H):
  order_h[i] = order_h[i >> 1] >> 1
  if i & 1:
    order_h[i] |= H >> 1
    
order_w = torch.tensor(order_w, dtype=torch.int32, device=device)
order_h = torch.tensor(order_h, dtype=torch.int32, device=device)

output_rfftn = torch.zeros([2, 1, 1, 8, 5], dtype=torch.float32, device=device)
autort.ops.rfftn_f32(x, temp_w, temp_h, order_w, order_h, output_rfftn)

print(output_rfftn[0])
print(output_rfftn[1])

real_diff = torch.max(torch.abs(output_cudnn.real - output_rfftn[0]))
imag_diff = torch.max(torch.abs(output_cudnn.imag - output_rfftn[1]))

print('Max diff (real):', real_diff)
print('Max diff (imag):', imag_diff)

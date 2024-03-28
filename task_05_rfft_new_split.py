import torch
import autort
import os


autort.export(name='fftn2_f32', source=r'''
@DEF_FUNC: input0:float32[NC, L, W, R], input1:float32[NC, L, W, R], temp_w:float32[2, W], order_w:int32[W] -> output:float32[2, NC, L, WO, R]

#define PI 3.1415926535897932

void main() {

  // rfft on the last dimension
  for (int nc = 0; nc < size_of_NC(); ++nc) {
    for (int l = 0; l < size_of_L(); ++l) {
      for (int r = 0; r < size_of_R(); ++r) {
        
        // write to temp and initialize
        for (int w = 0; w < size_of_W(); ++w) {
          temp_w(0, w) = input0(nc, l, w, r);
          temp_w(1, w) = input1(nc, l, w, r);
        }
        
        for (int i = 0; i < size_of_W(); ++i) {
          // swap x[i] and x[order[i]]
          if (i < order_w(i)) {
            float t = temp_w(0, i);
            temp_w(0, i) = temp_w(0, order_w(i));
            temp_w(0, order_w(i)) = t;
            t = temp_w(1, i);
            temp_w(1, i) = temp_w(1, order_w(i));
            temp_w(1, order_w(i)) = t;
          }
        }
        
        for (int x = 2; x <= size_of_W(); x = x * 2) {
          float temp_real = cos(2 * PI / x);
          float temp_imag = sin(-2 * PI / x);
          for (int j = 0; j < size_of_W(); j += x) {
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
          output(0, nc, l, wo, r) = temp_w(0, wo);
          output(1, nc, l, wo, r) = temp_w(1, wo);
        }
      }
    }
  }
}
''')

##########################################################################
os.environ['LOG'] = '1'

device = autort.device()
torch.manual_seed(0)
x = torch.randn([1, 1, 8, 8]).to(device)
output_cudnn = torch.fft.rfftn(x, dim=(2, 3))

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

# output_rfftn = torch.zeros([2, 1, 1, 8, 5], dtype=torch.float32, device=device)
# autort.ops.rfftn_f32(x, temp_w, temp_h, order_w, order_h, output_rfftn)

### Step-1: Full FFT on [N, C, H, `W`], requiring reshape from: N, C, H, W -> NC, H, W, 1
x_real = x.view(x.size(0) * x.size(1), x.size(2), x.size(3), 1)
x_imag = torch.zeros_like(x_real)
output_rfftn = torch.empty([2, 1, 1, 8, 5], dtype=torch.float32, device=device)
autort.ops.fftn2_f32(x_real, x_imag, temp_w, order_w, output_rfftn.view(2, output_rfftn.size(1) * output_rfftn.size(2), output_rfftn.size(3), output_rfftn.size(4), 1))

### Step-2: Full FFT on [N, C, `H`, W], requiring reshape from: 2, N, C, H, W -> NC, 1, H, W
x = output_rfftn
x_real = x[0, ...].view(x.size(1) * x.size(2), 1, x.size(3), x.size(4))
x_imag = x[1, ...].view(x.size(1) * x.size(2), 1, x.size(3), x.size(4))
output_rfftn = torch.empty_like(x)
autort.ops.fftn2_f32(x_real, x_imag, temp_h, order_h, output_rfftn.view(2, output_rfftn.size(1) * output_rfftn.size(2), 1, output_rfftn.size(3), output_rfftn.size(4)))

print(output_rfftn[0])
print(output_rfftn[1])

real_diff = torch.max(torch.abs(output_cudnn.real - output_rfftn[0]))
imag_diff = torch.max(torch.abs(output_cudnn.imag - output_rfftn[1]))

print('Max diff (real):', real_diff)
print('Max diff (imag):', imag_diff)

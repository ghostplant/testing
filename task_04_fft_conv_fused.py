from functools import partial
from typing import Iterable, Tuple, Union

import torch
import autort
import torch.nn.functional as f
from torch import Tensor, nn
from torch.fft import irfftn, rfftn
from math import ceil, floor


autort.export(ir='complex_matmul[N, F, M] +=! input0[N, C, M] * input1[F, C, M]',
  inputs=['input0=float32[N:64,C:32,M:4096]', 'input1=float32[F:32,C:32,M:4096]'],
  config='~N~:[2,8,4],~F~:[8,4,1],~M~:[1,8,1],~C~:[2,4],~@input1~:1,~@input0~:0')

autort.export(name='rfftn_f32', source=r'''
 @DEF_FUNC: input:float32[N, C, HS, WS], temp_w:float32[2, W], temp_h:float32[2, H] -> output:float32[2, N, C, H, WO]

 void main() {
    
  float PI = 3.1415926535897932;
  int m = size_of_W();
  
  // rfft on the last dimension
  for (int n = 0; n < size_of_N(); ++n) {
    for (int c = 0; c < size_of_C(); ++c) {
      for (int h = 0; h < size_of_H(); ++h) {
        
        // write to temp and initialize
        for (int w = 0; w < size_of_W(); ++w) {
          temp_w(0, w) = 0;
          if (h < size_of_HS() && w < size_of_WS())
            temp_w(0, w) = input(n, c, h, w);
          temp_w(1, w) = 0;
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

autort.export(name='irfftn_f32', source=r'''
 @DEF_FUNC: input0:float32[N, C, H, W], input1:float32[N, C, H, W], temp_w:float32[2, WO], temp_h:float32[2, H] -> output:float32[N, C, H, WO]

 void main() {

  float PI = 3.1415926535897932;
  int m = size_of_H();

  // ifft on the second-to-last dimension
  for (int n = 0; n < size_of_N(); ++n) {
    for (int c = 0; c < size_of_C(); ++c) {
      for (int w = 0; w < size_of_W(); ++w) {
        
        // write to temp and initialize
        for (int h = 0; h < size_of_H(); ++h) {
          temp_h(0, h) = input0(n, c, h, w);
          temp_h(1, h) = input1(n, c, h, w);
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
          input0(n, c, h, w) = temp_h(0, h) / m;
          input1(n, c, h, w) = temp_h(1, h) / m;
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
          temp_w(0, w) = input0(n, c, h, w);
          temp_w(1, w) = input1(n, c, h, w);
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


def complex_matmul(ar_, ai_, br_, bi_, groups: int = 1) -> Tensor:
    ar, ai = ar_.view(ar_.size(0), ar_.size(1), -1), ai_.view(ar_.size(0), ar_.size(1), -1)
    br, bi = br_.view(br_.size(0), br_.size(1), -1), bi_.view(br_.size(0), br_.size(1), -1)

    fn = autort.ops.complex_matmul
    # fn = lambda x, y: torch.einsum('ncm,fcm->nfm', x, y)
    real = fn(ar, br) + fn(ai, bi)
    imag = fn(ai, br) - fn(ar, bi)
    return real.view(ar_.shape), imag.view(ar_.shape)


def to_ntuple(val: Union[int, Iterable[int]], n: int) -> Tuple[int, ...]:
    """Casts to a tuple with length 'n'.  Useful for automatically computing the
    padding and stride for convolutions, where users may only provide an integer.

    Args:
        val: (Union[int, Iterable[int]]) Value to cast into a tuple.
        n: (int) Desired length of the tuple

    Returns:
        (Tuple[int, ...]) Tuple of length 'n'
    """
    if isinstance(val, Iterable):
        out = tuple(val)
        if len(out) == n:
            return out
        else:
            raise ValueError(f"Cannot cast tuple of length {len(out)} to length {n}.")
    else:
        return n * (val,)


def fft_conv(
    signal: Tensor,
    kernel: Tensor,
    bias: Tensor = None,
    padding: Union[int, Iterable[int], str] = 0,
    padding_mode: str = "constant",
    stride: Union[int, Iterable[int]] = 1,
    dilation: Union[int, Iterable[int]] = 1,
    groups: int = 1,
) -> Tensor:
    """Performs N-d convolution of Tensors using a fast fourier transform, which
    is very fast for large kernel sizes. Also, optionally adds a bias Tensor after
    the convolution (in order ot mimic the PyTorch direct convolution).

    Args:
        signal: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Tensor) Bias tensor to add to the output.
        padding: (Union[int, Iterable[int], str) If int, Number of zero samples to pad then
            input on the last dimension. If str, "same" supported to pad input for size preservation.
        padding_mode: (str) Padding mode to use from {constant, reflection, replication}.
                      reflection not available for 3d.
        stride: (Union[int, Iterable[int]) Stride size for computing output values.
        dilation: (Union[int, Iterable[int]) Dilation rate for the kernel.
        groups: (int) Number of groups for the convolution.

    Returns:
        (Tensor) Convolved tensor
    """

    # Cast padding, stride & dilation to tuples.
    n = signal.ndim - 2
    stride_ = to_ntuple(stride, n=n)
    dilation_ = to_ntuple(dilation, n=n)
    if isinstance(padding, str):
        if padding == "same":
            if stride != 1 or dilation != 1:
                raise ValueError("stride must be 1 for padding='same'.")
            padding_ = [(k - 1) / 2 for k in kernel.shape[2:]]
        else:
            raise ValueError(f"Padding mode {padding} not supported.")
    else:
        padding_ = to_ntuple(padding, n=n)

    # internal dilation offsets
    offset = torch.zeros(1, 1, *dilation_, device=signal.device, dtype=signal.dtype)
    offset[(slice(None), slice(None), *((0,) * n))] = 1.0

    # correct the kernel by cutting off unwanted dilation trailing zeros
    cutoff = tuple(slice(None, -d + 1 if d != 1 else None) for d in dilation_)

    # pad the kernel internally according to the dilation parameters
    kernel = torch.kron(kernel, offset)[(slice(None), slice(None)) + cutoff]

    # Pad the input signal & kernel tensors (round to support even sized convolutions)
    signal_padding = [r(p) for p in padding_[::-1] for r in (floor, ceil)]
    if any(signal_padding):
      signal = f.pad(signal, signal_padding, mode=padding_mode)

    # Because PyTorch computes a *one-sided* FFT, we need the final dimension to
    # have *even* length.  Just pad with one more zero if the final dimension is odd.
    signal_size = signal.size()  # original signal size without padding to even
    if signal.size(-1) % 2 != 0:
        signal = f.pad(signal, [0, 1])

    temp_w = torch.empty([2, signal.size(-1)], dtype=torch.float32, device=signal.device)
    temp_h = torch.empty([2, signal.size(-2)], dtype=torch.float32, device=signal.device)

    signal_fr_ = torch.empty([2,] + list(signal.shape[:-1]) + [signal.shape[-1] // 2 + 1], dtype=torch.float32, device=signal.device)
    autort.ops.rfftn_f32(signal, temp_w, temp_h, signal_fr_)

    kernel_fr_ = torch.empty([2, kernel.size(1)] + list(signal_fr_.shape[2:]), dtype=torch.float32, device=kernel.device)
    autort.ops.rfftn_f32(kernel, temp_w, temp_h, kernel_fr_)

    output_fr, output_fi  = complex_matmul(signal_fr_[0], signal_fr_[1], kernel_fr_[0], kernel_fr_[1], groups=groups)
    output = autort.ops.irfftn_f32(output_fr, output_fi, temp_w, temp_h)

    # Remove extra padded values
    crop_slices = [slice(None), slice(None)] + [
        slice(0, (signal_size[i] - kernel.size(i) + 1), stride_[i - 2])
        for i in range(2, signal.ndim)
    ]
    output = output[crop_slices].contiguous()

    # Optionally, add a bias term before returning.
    if bias is not None:
        bias_shape = tuple([1, -1] + (signal.ndim - 2) * [1])
        output += bias.view(bias_shape)

    return output


torch.manual_seed(0)

x = torch.randn([64, 32, 64, 64])
y = torch.randn([32, 32, 5, 5])

x, y = x.to(autort.device()), y.to(autort.device())

torch.perform(lambda: torch.nn.functional.conv2d(x, y))
torch.perform(lambda: fft_conv(x, y))
exit(0)

z = fft_conv(x, y).cpu()

z2 = torch.nn.functional.conv2d(x.cpu(), y.cpu())
print('Custom Result Sum:', z.sum().item())
print('CUDNN Result Sum:' , z2.sum().item())
print('Max Difference: ', (z - z2).max().item())
print('Output Shape', z.shape)

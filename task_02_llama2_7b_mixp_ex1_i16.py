#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os, sys, math, random, re
import torch
import autort


value_map = torch.tensor([6.1037e-5 * x - 1 for x in range(32768)], dtype=torch.float32)

def weight_preprocess(w):
  w_low = torch.bucketize(w, value_map[:-1]).to(torch.int16)

  # Debug closeness between w, w_low and w_recover ->
  '''
  print('True Origin Value =', w)
  print('Low Rank Value =', w_low)
  print('Recover from Low Rank =', torch.index_select(value_map, 0, w_low.view(-1).int()).view(w_low.shape))
  exit(0)
  '''

  return w_low

value_map_gpu = value_map.to(autort.device())


if False: # using standard TorchOps, but GPU memory will overflow

  def matmul_dequat(x, w, memory_out=None):
    x = x.view(-1)
    memory_out = memory_out if memory_out is not None else torch.empty([w.size(0)], dtype=x.dtype, device=x.device)
    w = torch.index_select(value_map_gpu, 0, w.view(-1).int()).view(w.shape)
    return torch.matmul(x, w.t(), out=memory_out.view(1, -1))

else:

  my_custom_fn = autort.export(ir="""
    w[M, K] = value_map_gpu[input1[M, K].unsigned_cast()]
    my_result[M] +=! input0[K] * w[M, K]
  """, inputs=["input0=float32[K]", "input1=int16[M, K]", "value_map_gpu=float32[L]"])

  def matmul_dequat(x, w, memory_out=None):
    x = x.view(-1)
    memory_out = memory_out if memory_out is not None else torch.empty([w.size(0)], dtype=x.dtype, device=x.device)
    return my_custom_fn(x, w, value_map_gpu, memory_out.view(-1), out=3)


os.environ['D3D12_ENABLE_FP16'] = '1'

vocab = torch.load(autort.download('./llama-2-7b-chat-hf/vocab_32K.pt', 'https://huggingface.co/datasets/ghostplant/data-collections/resolve/main/vocab_32K.pt?download=true'))

try:
  param_1 = torch.load(autort.download('./llama-2-7b-chat-hf/pytorch_model-00001-of-00002.bin'))
  param_2 = torch.load(autort.download('./llama-2-7b-chat-hf/pytorch_model-00002-of-00002.bin'))
except FileNotFoundError:
  raise Exception('Please visit https://huggingface.co/meta-llama/Llama-2-7b-chat-hf to download the required dataset.')

dictionary = {}
for i, word in enumerate(vocab):
  dictionary[word] = i

for k in param_2:
  param_1[k] = param_2[k]
param = param_1
del param_1, param_2

for n_layers in range(1024):
  try:
    q, k, v = param[f'model.layers.{n_layers}.self_attn.q_proj.weight'], param[f'model.layers.{n_layers}.self_attn.k_proj.weight'], param[f'model.layers.{n_layers}.self_attn.v_proj.weight']
    vqk = torch.cat([v, q, k])
    del q, k, v, param[f'model.layers.{n_layers}.self_attn.q_proj.weight'], param[f'model.layers.{n_layers}.self_attn.k_proj.weight'], param[f'model.layers.{n_layers}.self_attn.v_proj.weight']
    param[f'model.layers.{n_layers}.self_attn.vqk_proj.weight'] = vqk
    n_inv_freq = f'model.layers.{n_layers}.self_attn.rotary_emb.inv_freq'
    if n_inv_freq in param:
      del param[n_inv_freq]
  except KeyError:
    break

device = autort.device()
for k in param:
  print(f'Loading weight: {k}, val_min = {param[k].min()}, val_max = {param[k].max()}')
  if k == 'model.embed_tokens.weight':
    param[k] = param[k].float()
  elif 'layernorm' not in k:
    param[k] = weight_preprocess(param[k])
  param[k] = param[k].to(device)
print('')

token_embedding_table = param['model.embed_tokens.weight']
rms_end_w = param['model.norm.weight']
weight_classify = param['lm_head.weight']
data_type = token_embedding_table.dtype

rms_att_w = [param[f'model.layers.{i}.input_layernorm.weight'] for i in range(n_layers)]
rms_ffn_w = [param[f'model.layers.{i}.post_attention_layernorm.weight'] for i in range(n_layers)]
weight_o = [param[f'model.layers.{i}.self_attn.o_proj.weight'] for i in range(n_layers)]
weight_f1 = [param[f'model.layers.{i}.mlp.gate_proj.weight'] for i in range(n_layers)]
weight_f2 = [param[f'model.layers.{i}.mlp.down_proj.weight'] for i in range(n_layers)]
weight_f3 = [param[f'model.layers.{i}.mlp.up_proj.weight'] for i in range(n_layers)]
weight_vqk = [param[f'model.layers.{i}.self_attn.vqk_proj.weight'] for i in range(n_layers)]

n_heads = 32
head_size = token_embedding_table.size(-1) // n_heads
token_embedding_table = token_embedding_table.view([token_embedding_table.size(0), n_heads, head_size])

vocab_size, n_heads, head_size, = token_embedding_table.size(0), token_embedding_table.size(1), token_embedding_table.size(2)
seq_len, hidden, = 1024, weight_f1[0].size(0)
kv_heads, dim = n_heads, n_heads * head_size

assert n_heads // kv_heads == 1 and head_size % 2 == 0

key_cache = torch.zeros([n_layers, seq_len, dim], dtype=data_type, device=device).clone()
val_cache = torch.zeros([n_layers, seq_len, dim], dtype=data_type, device=device).clone()

ceof = 1 / torch.pow(1e4, torch.arange(0, dim, 2, dtype=torch.int64) % head_size / head_size).view(1, -1).to(device)
att_f = torch.tensor([1 / math.sqrt(head_size)], dtype=torch.float32, device=device)

inv_freq = (1.0 / (10000.0 ** (torch.arange(0, head_size, 2).float() / head_size)).to(data_type))
inv_freq = torch.cat([inv_freq, inv_freq]).view(head_size).to(device)

def rmsnorm(x, weight):
  x = x.float()
  vsum = (x * x).sum().view(1)
  return autort.ops.rmsnorm_f32(x.view(-1), vsum, weight.float(), extra=[1.0 / int(x.numel())])

def forward(token, pos):
  x = token_embedding_table.select(0, token).view(1, dim)

  for l in range(n_layers):
    xb = rmsnorm(x, rms_att_w[l])
    local_cache = val_cache.select(0, l).narrow(0, pos, 3)
    matmul_dequat(xb, weight_vqk[l], memory_out=local_cache.view(-1, 3 * xb.size(-1)))
    sq, sk = local_cache[1], local_cache[2]

    sq_out = torch.empty_like(sq).view(n_heads, head_size)
    sk_out = key_cache.select(0, l).narrow(0, pos, 1).view(n_heads, head_size)
    autort.ops.rotary_f32(sq.view(n_heads, 2, -1), inv_freq, sq_out, extra=[pos,])
    autort.ops.rotary_f32(sk.view(n_heads, 2, -1), inv_freq, sk_out, extra=[pos,])
    sq, sk = sq_out, sk_out

    b_sq = sq.view(n_heads, head_size)
    b_sk = key_cache.select(0, l).view(seq_len, n_heads, head_size).narrow(0, 0, pos + 1)
    b_sv = val_cache.select(0, l).view(seq_len, n_heads, head_size).narrow(0, 0, pos + 1)

    xb = autort.ops.attention_f32(b_sq, b_sk, b_sv, att_f)

    xb = matmul_dequat(xb, weight_o[l])
    x = x + xb
    xb = rmsnorm(x, rms_ffn_w[l])

    xb = torch.nn.functional.silu(matmul_dequat(xb, weight_f1[l])) * matmul_dequat(xb, weight_f3[l])
    xb = matmul_dequat(xb, weight_f2[l])
    x = x + xb

  x = rmsnorm(x, rms_end_w)
  logits = matmul_dequat(x, weight_classify)
  return logits.half()

def decode(prev, next):
  piece = vocab[next]
  if prev == 1 and piece.startswith(' '):
    piece = piece[1:]
  if re.match(r'^\<0x..\>$', piece):
    piece = chr(int(piece[1:-1], 16))
  return piece

if __name__ == '__main__':
  prompt = 'How large is Atlantic Ocean'
  prompt_tokens = [1] + [dictionary[f' {x}' if f' {x}' in dictionary else x] for x in prompt.split()]

  with torch.no_grad():
    pos, token = 0, prompt_tokens[0]

    while pos < seq_len:
      logits = forward(token, pos)

      if pos < len(prompt_tokens) - 1:
        next = int(prompt_tokens[pos + 1])
      else:
        next = int(torch.argmax(logits))
      if next <= 2:
        print()
        break

      sys.stdout.write(decode(token, next))
      sys.stdout.flush()
      pos, token = pos + 1, next


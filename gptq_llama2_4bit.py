import os, sys, math, random, re
import torch
import autort
from safetensors import safe_open

os.environ['D3D12_ENABLE_FP16'] = '1'
vocab = torch.load(autort.download('./llama-2-7b-chat-hf/vocab_32K.pt', 'https://huggingface.co/datasets/ghostplant/data-collections/resolve/main/vocab_32K.pt?download=true'))

dictionary = {}
for i, word in enumerate(vocab):
  dictionary[word] = i
    
param = {}
with safe_open('./Llama-2-7B-Chat-GPTQ/model.safetensors', framework='pt') as f:
  for k in f.keys():
    param[k] = f.get_tensor(k)
        
for n_layers in range(32):
  # We skipped `bias` and `g_idx` parameters here.
  try:
    qweight_q = param[f'model.layers.{n_layers}.self_attn.q_proj.qweight']
    qweight_v = param[f'model.layers.{n_layers}.self_attn.v_proj.qweight']
    qweight_k = param[f'model.layers.{n_layers}.self_attn.k_proj.qweight']
    qweight_vqk = torch.cat([qweight_v, qweight_q, qweight_k], dim=1)
    del qweight_q, qweight_v, qweight_k
    del param[f'model.layers.{n_layers}.self_attn.q_proj.qweight']
    del param[f'model.layers.{n_layers}.self_attn.v_proj.qweight']
    del param[f'model.layers.{n_layers}.self_attn.k_proj.qweight']
    param[f'model.layers.{n_layers}.self_attn.vqk_proj.qweight'] = qweight_vqk
    
    ####################################################
    ## qzeros are not requires, since they are all 8. ##
    ####################################################
    del param[f'model.layers.{n_layers}.self_attn.q_proj.qzeros']
    del param[f'model.layers.{n_layers}.self_attn.v_proj.qzeros']
    del param[f'model.layers.{n_layers}.self_attn.k_proj.qzeros']
    del param[f'model.layers.{n_layers}.self_attn.o_proj.qzeros']
    del param[f'model.layers.{n_layers}.mlp.gate_proj.qzeros']
    del param[f'model.layers.{n_layers}.mlp.down_proj.qzeros']
    del param[f'model.layers.{n_layers}.mlp.up_proj.qzeros']
    
    scales_q = param[f'model.layers.{n_layers}.self_attn.q_proj.scales']
    scales_v = param[f'model.layers.{n_layers}.self_attn.v_proj.scales']
    scales_k = param[f'model.layers.{n_layers}.self_attn.k_proj.scales']
    scales_vqk = torch.cat([scales_v, scales_q, scales_k], dim=1)
    del scales_q, scales_v, scales_k
    del param[f'model.layers.{n_layers}.self_attn.q_proj.scales']
    del param[f'model.layers.{n_layers}.self_attn.v_proj.scales']
    del param[f'model.layers.{n_layers}.self_attn.k_proj.scales']
    param[f'model.layers.{n_layers}.self_attn.vqk_proj.scales'] = scales_vqk
    
    # g_idx parameters are useless
    del param[f'model.layers.{n_layers}.self_attn.q_proj.g_idx']
    del param[f'model.layers.{n_layers}.self_attn.v_proj.g_idx']
    del param[f'model.layers.{n_layers}.self_attn.k_proj.g_idx']
    del param[f'model.layers.{n_layers}.self_attn.o_proj.g_idx']
    del param[f'model.layers.{n_layers}.mlp.gate_proj.g_idx']
    del param[f'model.layers.{n_layers}.mlp.down_proj.g_idx']
    del param[f'model.layers.{n_layers}.mlp.up_proj.g_idx']
    
    # bias parameters are useless
    del param[f'model.layers.{n_layers}.self_attn.q_proj.bias']
    del param[f'model.layers.{n_layers}.self_attn.v_proj.bias']
    del param[f'model.layers.{n_layers}.self_attn.k_proj.bias']
    del param[f'model.layers.{n_layers}.self_attn.o_proj.bias']
    del param[f'model.layers.{n_layers}.mlp.gate_proj.bias']
    del param[f'model.layers.{n_layers}.mlp.down_proj.bias']
    del param[f'model.layers.{n_layers}.mlp.up_proj.bias']
    
    n_inv_freq = f'model.layers.{n_layers}.self_attn.rotary_emb.inv_freq'
    if n_inv_freq in param:
      del param[n_inv_freq]
      
  except KeyError:
    break


n_layers = 32

device = autort.device()
for k in param:
  # print(f'Loading weight: {k}')
  if k == 'model.embed_tokens.weight' or k == 'lm_head.weight' or 'norm' in k:
    param[k] = param[k].float()
  param[k] = param[k].to(device)


token_embedding_table = param['model.embed_tokens.weight']
rms_end_w = param['model.norm.weight']
weight_classify = param['lm_head.weight']
data_type = token_embedding_table.dtype

rms_att_w = [param[f'model.layers.{i}.input_layernorm.weight'] for i in range(n_layers)]
rms_ffn_w = [param[f'model.layers.{i}.post_attention_layernorm.weight'] for i in range(n_layers)]

qweight_o = [param[f'model.layers.{i}.self_attn.o_proj.qweight'] for i in range(n_layers)]
scales_o = [param[f'model.layers.{i}.self_attn.o_proj.scales'] for i in range(n_layers)]

qweight_f1 = [param[f'model.layers.{i}.mlp.gate_proj.qweight'] for i in range(n_layers)]
scales_f1 = [param[f'model.layers.{i}.mlp.gate_proj.scales'] for i in range(n_layers)]

qweight_f2 = [param[f'model.layers.{i}.mlp.down_proj.qweight'] for i in range(n_layers)]
scales_f2 = [param[f'model.layers.{i}.mlp.down_proj.scales'] for i in range(n_layers)]

qweight_f3 = [param[f'model.layers.{i}.mlp.up_proj.qweight'] for i in range(n_layers)]
scales_f3 = [param[f'model.layers.{i}.mlp.up_proj.scales'] for i in range(n_layers)]

qweight_vqk = [param[f'model.layers.{i}.self_attn.vqk_proj.qweight'] for i in range(n_layers)]
scales_vqk = [param[f'model.layers.{i}.self_attn.vqk_proj.scales'] for i in range(n_layers)]

n_heads = 32
head_size = token_embedding_table.size(-1) // n_heads
token_embedding_table = token_embedding_table.view([token_embedding_table.size(0), n_heads, head_size])

vocab_size, n_heads, head_size, = token_embedding_table.size(0), token_embedding_table.size(1), token_embedding_table.size(2)
seq_len, hidden, = 1024, qweight_f1[0].size(0)
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

#------ Dequat Config ------#
bits = 4
group_size = 128
wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0).to(device)
#---------------------------#


my_custom_fn = autort.export(ir="""
  input1[S, N] = (qweight[S // 8, N] >> (S % 8 * 4)).call(strs.bitwise_and, 15) * input0[S].like(1)
  w[S, N] = scales[S // 128, N] * (input1[S, N] - 8)
  my_result[N] +=! input0[S] * w[S, N]
""", inputs=["input0=float32[S:4096]", "qweight=int32[L:512, N:12288]", "scales=float16[K:32, N:12288]"], config='~N~:[256,8,32],~S~:[64,8]')


def matmul_dequat(x, qweight, scales, memory_out=None):
  x = x.view(-1)
  memory_out = memory_out if memory_out is not None else torch.empty([qweight.size(-1)], dtype=x.dtype, device=x.device)
  return my_custom_fn(x, qweight, scales, out=memory_out.view(-1))


# my_custom_fn = autort.export(ir="""
#   w[S, N] = scales[S / (S.val() / scales.size(0)), 0, N] * (input1[S, N] - 8)
#   my_result[N] +=! input0[S] * w[S, N]
# """, inputs=["input0=float32[S:4096]", "input1=int8[S:4096, N:12288]", "scales=float32[K:32, 1, N:12288]"], config='~N~:[256,8,32],~S~:[64,8]')

# def matmul_dequat(x, qweight, scales, memory_out=None):
    
#   scales = scales.view(-1, 1, scales.shape[-1]).to(torch.float32)
#   weight = torch.bitwise_right_shift(
#     torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1),
#     wf.unsqueeze(-1)
#   ).to(torch.int8)
#   weight = torch.bitwise_and(weight, (2**bits) - 1)
  
#   print(weight)
#   exit(-1)
  
#   # print(weight.shape, weight.dtype)
#   # torch.Size([512, 8, 12288]) torch.int8
  
#   x = x.view(-1)
#   memory_out = memory_out if memory_out is not None else torch.empty([weight.size(-1)], dtype=x.dtype, device=x.device)
  
#   if True:
#     weight = weight.view(-1, weight.size(2))
#     return my_custom_fn(x, weight, scales, out=memory_out.view(-1))
#   else:
#     weight = weight.view(-1, group_size, weight.shape[2])
#     weight = scales * (weight - 8)
#     weight = weight.view(-1, weight.shape[2])
#     return torch.matmul(x, weight, out=memory_out.view(1, -1))


def forward(token, pos):
  x = token_embedding_table.select(0, token).view(1, dim)

  for l in range(n_layers):
    xb = rmsnorm(x, rms_att_w[l])
    local_cache = val_cache.select(0, l).narrow(0, pos, 3)
    matmul_dequat(xb, qweight_vqk[l], scales_vqk[l], memory_out=local_cache.view(-1, 3 * xb.size(-1)))
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

    xb = matmul_dequat(xb, qweight_o[l], scales_o[l])
    x = x + xb
    xb = rmsnorm(x, rms_ffn_w[l])

    xb = torch.nn.functional.silu(matmul_dequat(xb, qweight_f1[l], scales_f1[l])) * matmul_dequat(xb, qweight_f3[l], scales_f3[l])
    xb = matmul_dequat(xb, qweight_f2[l], scales_f2[l])
    x = x + xb

  x = rmsnorm(x, rms_end_w)
  logits = torch.matmul(x, weight_classify.t())
  return logits

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

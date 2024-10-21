import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.hparams import hparams

class MultiHeadAttention(nn.Module):
	def __init__(self, embed_size, heads):
		super(MultiHeadAttention, self).__init__()
		self.embed_size = embed_size
		self.heads = heads
		self.head_dim = embed_size // heads
		assert (
			self.head_dim * heads == embed_size
		), "Embedding size needs to be divisible by heads"
		self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
		self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
		self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
		self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
	def forward(self, values, keys, query, mask):
		N = query.shape[0]
		value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
		# Split the embedding into self.heads different pieces
		values = values.reshape(N, value_len, self.heads, self.head_dim)
		keys = keys.reshape(N, key_len, self.heads, self.head_dim)
		queries = query.reshape(N, query_len, self.heads, self.head_dim)
		values = self.values(values)
		keys = self.keys(keys)
		queries = self.queries(queries)
		# Einsum does matrix multiplication for query*keys for each training example
		# with every other training example, don't be confused by einsum
		# it's just a way to do matrix multiplication
		attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
		if mask is not None:
			attention = attention.masked_fill(mask == 0, float("-1e20"))
		# Apply softmax activation to the attention scores
		attention = F.softmax(attention / (self.embed_size ** (1 / 2)), dim=3)
		out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
			N, query_len, self.heads * self.head_dim
		)
		out = self.fc_out(out)
		return out

class TransformerBlock(nn.Module):
	def __init__(self, embed_size, heads, dropout=0.1, forward_expansion=2):
		super(TransformerBlock, self).__init__()
		self.attention = MultiHeadAttention(embed_size, heads)
		self.norm1 = nn.LayerNorm(embed_size)
		self.norm2 = nn.LayerNorm(embed_size)
		self.feed_forward = nn.Sequential(
			nn.Linear(embed_size, forward_expansion * embed_size),
			nn.ReLU(),
			nn.Linear(forward_expansion * embed_size, embed_size)
		)
		self.dropout = nn.Dropout(dropout)
	def forward(self, value, key, query):
		value = value.transpose(1, 2)
		key = key.transpose(1, 2)
		query = query.transpose(1, 2)
		attention = self.attention(value, key, query, None)
		x = self.dropout(self.norm1(attention + query))
		forward = self.feed_forward(x)
		out = self.dropout(self.norm2(forward + x))
		return out.transpose(1, 2)

class ReporelPositionEncoding(nn.Module):
	def __init__(self, odim=512, max_len=25000, conloop=1):
		super().__init__()
		table = torch.arange(max_len) / 32
		wif1 = []
		vcp = 1
		for i in range(0, odim):
			if i % 2 == 0:
				wif1.append(torch.sin(table * vcp))
			else:
				wif1.append(torch.cos(table * vcp))
				vcp = vcp * 1.05946309436
		self.wif1 = torch.stack(wif1 * conloop,dim=0).unsqueeze(0)
	def forward(self, x):
		return x + (self.wif1.to(x.device))[:, :, :x.size(2)]

class Transpose(nn.Module):
	def __init__(self, dims):
		super().__init__()
		assert len(dims) == 2, 'dims must be a tuple of two dimensions'
		self.dims = dims

	def forward(self, x):
		return x.transpose(*self.dims)

class ApcaNetContaper(nn.Module):
	def __init__(self, dim=512, dim_cond=256, dropout=0.1):
		super().__init__()
		assert dim % 4 == 0, "ApcaNet needs residual_channel to be divisible by 4"
		self.dim = dim
		self.diffusion_projection = nn.Conv1d(dim, dim, 1)
		self.conditioner_projection = nn.Conv1d(dim_cond, dim, 1)
		self.vergeknown = nn.Sequential(
			nn.Conv1d(dim * 24, dim * 6, kernel_size=1, groups=dim * 2),
			nn.Mish(),
			nn.Conv1d(dim * 6, dim * 6, kernel_size=81, padding=40, groups=dim * 3),
			Transpose((1,2)),
			nn.LayerNorm(dim * 6),
			Transpose((1,2)),
			ReporelPositionEncoding(odim=dim * 3 // 2, conloop=4),
			nn.Conv1d(dim * 6, dim * 6, kernel_size=1),
		)
		self.transform = TransformerBlock(embed_size=dim * 2, heads=8, dropout=dropout)
		self.proj = nn.Sequential(
			nn.Conv1d(dim * 2, dim * 6, kernel_size=1),
			nn.Mish(),
			nn.Conv1d(dim * 6, dim * 24, kernel_size=5, padding=2, groups=dim // 2),
			Transpose((1,2)),
			nn.LayerNorm(dim * 24),
			Transpose((1,2)),
		)
	def forward(self, x, cond, diffstep):
		residual = x
		x = x + self.diffusion_projection(diffstep)
		x = x + self.conditioner_projection(cond)
		if x.size(2) % 24 != 0:
			x = F.pad(x, (24 - (x.size(2) % 24) ,0), "replicate")
		x = x.reshape(-1, self.dim * 24, x.size(2) // 24)
		x = self.vergeknown(x)
		q, k, v = torch.split(x, [self.dim * 2, self.dim * 2, self.dim * 2], dim=1)
		x = self.transform(q, k, v)
		x = self.proj(x)
		x = x.reshape(-1, self.dim, x.size(2) * 24)
		x = x[:, :, :residual.size(2)]
		return x + residual

class SinusoidalPosEmb(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.dim = dim

	def forward(self, x):
		device = x.device
		half_dim = self.dim // 2
		emb = math.log(10000) / (half_dim - 1)
		emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
		emb = x[:, None] * emb[None, :]
		emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
		return emb

class ApcaNet(nn.Module):
	def __init__(self, in_dims, n_feats, *, n_layers=6, n_chans=512, n_dilates=2, dropout=0.1):
		super().__init__()
		self.input_projection = nn.Conv1d(in_dims * n_feats, n_chans, 1)
		self.diffusion_embedding = nn.Sequential(
			SinusoidalPosEmb(n_chans),
			nn.Linear(n_chans, n_chans * 4),
			nn.GELU(),
			nn.Linear(n_chans * 4, n_chans)
		)
		self.residual_layers = nn.ModuleList(
			[
				ApcaNetContaper(
					dim_cond=hparams['hidden_size'], 
					dim=n_chans,
					dropout=dropout
				)
				for i in range(n_layers)
			]
		)
		self.norm = nn.LayerNorm(n_chans)
		self.output_projection = nn.Conv1d(n_chans, in_dims * n_feats, kernel_size=1)
		nn.init.zeros_(self.output_projection.weight)
	
	def forward(self, spec, diffusion_step, cond):
		"""
		:param spec: [B, F, M, T]
		:param diffusion_step: [B, 1]
		:param cond: [B, H, T]
		:return:
		"""
		
		# To keep compatibility with DiffSVC, [B, 1, M, T]
		x = spec
		use_4_dim = False
		if x.dim() == 4:
			x = x[:, 0]
			use_4_dim = True

		assert x.dim() == 3, f"mel must be 3 dim tensor, but got {x.dim()}"

		x = self.input_projection(x)  # x [B, residual_channel, T]
		x = F.gelu(x)
		
		diffusion_step = self.diffusion_embedding(diffusion_step).unsqueeze(-1)
		
		for layer in self.residual_layers:
			x = layer(x, cond, diffusion_step)

		# post-norm
		x = self.norm(x.transpose(1, 2)).transpose(1, 2)
		
		# MLP and GLU
		x = self.output_projection(x)  # [B, 128, T]
		
		return x[:, None] if use_4_dim else x

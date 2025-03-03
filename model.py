#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2024/2025
#############################################################################
###
### Машинен превод чрез генеративен езиков модел
###
#############################################################################

import torch
import torch.nn as nn

from parameters import device, vocab_size
from positional_encoding import PositionalEncoding
from utils import padTokenIdx, endTokenIdx


class SingleHeadAttention(torch.nn.Module):
	def __init__(self, d_model, d_k, d_v):
		super(SingleHeadAttention, self).__init__()

		self.d_k = d_k
		self.WQ = torch.nn.Linear(d_model, d_k, bias=False)
		self.WK = torch.nn.Linear(d_model, d_k, bias=False)
		self.WV = torch.nn.Linear(d_model, d_v, bias=False)

	def forward(self, Q, K, V, mask=None):
		# Q, K, V: shape=(batch_size, seq_len, d_model)
		Q = self.WQ(Q)
		K = self.WK(K)
		V = self.WV(V)
		# Q, K, V: shape=(batch_size, seq_len, d_k/d_v)


		x = torch.matmul(Q, K.transpose(-1,-2)) / (self.d_k ** 0.5)
		# x: shape=(batch_size, seq_len, seq_len)

		if mask is not None:
			x = x.masked_fill(mask == 0, float('-inf'))
			# x: shape=(batch_size, seq_len, seq_len)

		x = nn.functional.softmax(x, dim=-1)
		# x: shape=(batch_size, seq_len, seq_len)

		H = torch.matmul(x, V)
		# H: shape=(batch_size, seq_len, d_v)
		return H

class DecoderBlock(torch.nn.Module):
	def __init__(self, d_model, d_k, d_v, num_heads, dropout, d_ff):
		super(DecoderBlock, self).__init__()

		self.multi_head_attention = torch.nn.MultiheadAttention(
			d_model, num_heads,
			dropout=dropout, bias=True, #kdim=d_k, vdim=d_v,
			batch_first=True, device=device, dtype=torch.float32
		)

		self.layer_norm1 = torch.nn.LayerNorm(d_model)

		self.feed_forward = torch.nn.Sequential(
			torch.nn.Linear(d_model, d_ff),
			torch.nn.ReLU(),
			torch.nn.Dropout(0.9 * dropout),
			torch.nn.Linear(d_ff, d_model),
			torch.nn.Dropout(0.1 * dropout),
		)
		self.layer_norm2 = torch.nn.LayerNorm(d_model)

	def forward(self, Q, K, V, mask=None, key_padding_mask=None, is_single_query=False):
		# Q, K, V: shape=(batch_size, seq_len, d_model)
		if not is_single_query:
			H = self.multi_head_attention(Q, K, V,
										  key_padding_mask=key_padding_mask,
										  attn_mask=mask, is_causal=True,
										  need_weights=False)[0]
		else:
			H = self.multi_head_attention(Q, K, V, need_weights=False)[0]

		# H: shape=(batch_size, seq_len, d_model)
		Q = H + Q
		# Q: shape=(batch_size, seq_len, d_model)
		H = self.layer_norm1(Q)
		# Q: shape=(batch_size, seq_len, d_model)
		Q = self.feed_forward(H)
		# Q: shape=(batch_size, seq_len, d_model)
		Q = Q + H
		# Q: shape=(batch_size, seq_len, d_model)
		Q = self.layer_norm2(Q)
		# Q: shape=(batch_size, seq_len, d_model)
		return Q

class LanguageModel(torch.nn.Module):
	def __init__(self,
	    max_seq_len,
		num_blocks,
		d_model,
		num_heads,
		d_k,
		d_v,
		dropout,
		d_ff,
		label_smoothing,
	):
		super(LanguageModel, self).__init__()

		self.d_model = d_model
		self.positional_encoding = PositionalEncoding(d_model, 0.5 * dropout, max_seq_len)

		self.embedding = torch.nn.Embedding(
			vocab_size,
			d_model,
			padding_idx=padTokenIdx,
			scale_grad_by_freq=True,
			device=device,
		)

		self. decoder_blocks = torch.nn.ModuleList([
			DecoderBlock(d_model, d_k, d_v, num_heads, dropout, d_ff)
			for _ in range(num_blocks)
		])

		self.linear = torch.nn.Linear(d_model, vocab_size, device=device)

		self.label_smoothing = label_smoothing

	def preparePaddedBatch(self, source):
		device = next(self.parameters()).device
		m = max(len(s) for s in source)
		sents_padded = [s + (m - len(s)) * [padTokenIdx] for s in source]
		return torch.tensor(sents_padded, dtype=torch.long, device=device)	# shape=(batch_size, seq_len)

	def save(self,fileName):
		torch.save(self.state_dict(), fileName)

	def load(self,fileName):
		self.load_state_dict(torch.load(fileName))

	def forward(self, source, targets=None):
		x = source
		if isinstance(x, list):
			x = self.preparePaddedBatch(x)
		# source: shape=(batch_size, seq_len)
		seq_len = x.shape[1]
		key_padding_mask = (x == padTokenIdx).to(device)


		x = self.embedding(x)
		x = self.positional_encoding(x)
		# x: shape=(batch_size, seq_len, d_model)

		mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)  # Upper triangular part (masking future tokens)
		mask = mask.bool().to(device)
		for block in self.decoder_blocks:
			x = block(x, x, x, mask=mask, key_padding_mask=key_padding_mask)
			# x: shape=(batch_size, seq_len, d_model)

		logits = self.linear(x)
		# logits: shape=(batch_size, seq_len, vocab_size)

		if targets is None:
			loss = None
		else:
			batch_size, context_length = x.shape[:2]
			logits = logits.view(batch_size * context_length, vocab_size)
			targets = targets.reshape(batch_size * context_length)
			loss = nn.functional.cross_entropy(
				logits, targets,
				label_smoothing=self.label_smoothing,
				ignore_index=padTokenIdx,
				reduction='mean',
			)
		return logits, loss

	def gen_forward(self, source, limit=1500, temperature=1.0):
		for token in source:
			yield token

		src = torch.zeros(1, limit, dtype=torch.long, device=device)
		src[:1, :len(source)] = torch.tensor(source, dtype=torch.long, device=device)

		pos = -1
		while len(source) < limit:
			emb = torch.zeros(1, limit, self.d_model, dtype=torch.float, device=device)
			if pos == -1:
				emb[:1, :len(source)] = self.embedding(src[:1, :len(source)])
			else:
				emb[:1, pos: pos + 1] = self.embedding(src[:1, pos: pos + 1])

			pos_enc = torch.zeros_like(emb, dtype=torch.float, device=device)
			if pos == -1:
				pos_enc[:1, :len(source), :] = self.positional_encoding(emb[:1, :len(source), :])
			else:
				pos_enc[:1, pos: pos + 1, :] = self.positional_encoding(emb[:1, pos: pos + 1, :], pos)

			mask = torch.triu(torch.ones(limit, limit), diagonal=1)  # Upper triangular part (masking future tokens)
			mask = mask.bool().to(device)
			block_queries = [
				torch.zeros_like(pos_enc, dtype=torch.float, device=device)
				for _ in range(len(self.decoder_blocks))
			]
			for i, block in enumerate(self.decoder_blocks):
				if pos == -1:
					key_value = emb[:1, :len(source), :] if i == 0 else block_queries[i - 1][:1, :len(source), :]
					block_queries[i][:1, :len(source), :] = block(key_value, key_value, key_value,
																 mask=mask[:len(source), :len(source)])
				else:
					query = emb[:1, pos: pos + 1, :] if i == 0 else block_queries[i - 1][:1, pos: pos + 1, :]
					key_value = emb[:1, :pos, :] if i == 0 else block_queries[i - 1][:1, :pos, :]
					block_queries[i][:1, pos: pos + 1, :] = block(query, key_value, key_value, is_single_query=True)

			if pos == -1:
				logits = self.linear(block_queries[-1][:1, len(source) - 1: len(source), :])
			else:
				logits = self.linear(block_queries[-1][:1, pos: pos + 1, :])

			logits = logits.view(vocab_size)
			logits = logits / temperature
			probs = nn.functional.softmax(logits, dim=-1)

			next_token = torch.multinomial(probs, num_samples=1)
			next_token = next_token.detach().cpu().numpy()[0]


			pos = len(source)
			source.append(next_token)

			yield next_token

			if next_token == endTokenIdx:
				break
	def generate_next_token(self, prefix, temperature=1.0):
		logits, _ = self.forward([prefix])
		logits = logits[0, -1] / temperature
		probs = nn.functional.softmax(logits, dim=-1)
		next_token = torch.multinomial(probs, num_samples=1)
		next_token = next_token.detach().cpu().numpy()[0]
		return next_token
		
	def generate(self, prefix, limit=1500, temperature=1.0):
		for token in prefix:
			yield token

		result = prefix
		while len(result) < limit:
			next_token = self.generate_next_token(result, temperature=temperature)
			result.append(next_token)
			yield next_token
			if next_token == endTokenIdx:
				break


if __name__ == '__main__':
	from parameters import *
	nmt = LanguageModel(
		max_seq_len,
		num_blocks,
		d_model,
		num_heads,
		d_k,
		d_v,
		dropout,
		d_ff,
		label_smoothing,
	).to(device)
	nmt.load(modelFileName)
	nmt.eval()

	from tokenizer import BPETokenizer
	from utils import *

	tokenizer = BPETokenizer.from_files()


	import time
	nmt.eval()

	def quadratic(tokens, log=False):
		global nmt, tokenizer
		begin = time.time()
		for token in nmt.gen_forward(tokens):
			if log:
				print(tokenizer.untokenize([token])[0], end='')
		if log:
			print()
		end = time.time()
		if log:
			print('Time:', end - begin)
		return end - begin

	def cubic(tokens, log=False):
		global nmt, tokenizer
		begin = time.time()
		for token in nmt.generate(tokens):
			if log:
				print(tokenizer.untokenize([token])[0], end='')
		if log:
			print()
		end = time.time()
		if log:
			print('Time:', end - begin)
		return end - begin

	import string
	import random

	def generate_fake_word(length=6):
		return ''.join(random.choices(string.ascii_lowercase, k=length))

	with torch.no_grad():
		for l in (10, 100, 300, 500):
			text = generate_fake_word(l)
			tokens = [startTokenIdx] + tokenizer.tokenize(list(text)) + [transTokenIdx]
			print("N=", l)
			print('Quadratic:', quadratic(tokens))
			print('Cubic:', cubic(tokens))
			print()

	# From the experiment, qubic is faster than quadratic,
	# as quadratic one has to pre-allocate the memory for the limit long sequence,


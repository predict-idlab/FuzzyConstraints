import torch
import torch.nn as nn
from models.Model import Model

class DistMult(Model):

	def __init__(self, multimodal, loss=None, batch_size=128, margin = None, epsilon = None):
		super(DistMult, self).__init__(multimodal, loss, batch_size)

		self.dim = multimodal.embedding_size
		self.margin = margin
		self.epsilon = epsilon
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
		self.l3_regul_rate = 0.001

		if margin == None or epsilon == None:
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
		else:
			self.embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
			)
			nn.init.uniform_(
				tensor = self.ent_embeddings.weight.data, 
				a = -self.embedding_range.item(), 
				b = self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)

	def _calc(self, h, t, r, mode):
		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
		if mode == 'head_batch':
			score = h * (r * t)
		else:
			score = (h * r) * t
		score = torch.sum(score, -1).flatten()
		return score

	def _forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']
		h = self.enhance(batch_h)
		t = self.enhance(batch_t)
		r = self.rel_embeddings(batch_r)
		score = self._calc(h ,t, r, mode)
		return score

	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.enhance(batch_h)
		t = self.enhance(batch_t)
		r = self.rel_embeddings(batch_r)
		regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
		return regul

	def l3_regularization(self):
		return (self.ent_embeddings.weight.norm(p = 3)**3 + self.rel_embeddings.weight.norm(p = 3)**3)

	def predict(self, data):
		score = -self._forward(data)
		return score.cpu().data.numpy()

	def enhance(self, e):
		if self.multimodal.sensory:
			# separate the entities corresponding to literals from non-valued entities
			e_num = [i for i, ex in enumerate(e) if ex.item() in self.multimodal.numliteral2id]
			e_txt = [i for i, ex in enumerate(e) if ex.item() in self.multimodal.textliteral2id]
			e_rel = [i for i in range(len(e)) if i not in e_num and i not in e_txt]

			# get the original batch indices corresponding to each category
			num_ind = [e[i].item() for i in e_num]
			txt_ind = [e[i].item() for i in e_txt]
			rel_ind = [e[i].item() for i in e_rel]

			# get the embeddings corresponding to non-literal entities
			rel_emb = self.ent_embeddings(torch.LongTensor(rel_ind))

			# enhance the non-literal entities
			if self.multimodal.attributive:
				if self.multimodal.numerical_literals is not None and self.multimodal.textual_literals is not None:
					num_lit = self.multimodal.numerical_literals[rel_ind]
					txt_lit = self.multimodal.textual_literals[rel_ind]
					rel_emb = self.multimodal.attr_gate(rel_emb, num_lit, txt_lit)
				elif self.multimodal.numerical_literals is not None and self.multimodal.textual_literals is None:
					num_lit = self.multimodal.numerical_literals[rel_ind]
					rel_emb = self.multimodal.attr_gate(rel_emb, num_lit)
				elif self.multimodal.numerical_literals is None and self.multimodal.textual_literals is not None:
					txt_lit = self.multimodal.textual_literals[rel_ind]
					rel_emb = self.multimodal.attr_gate(rel_emb, txt_lit)

			# get the structural embeddings corresponding to numerical literals
			emb = self.ent_embeddings(torch.LongTensor(num_ind))

			# enhance the numerical entities with their literal embedding
			if len(num_ind) > 0:
				num_emb = self.multimodal.sen_num_gate(emb, self.multimodal.numerical_embeddings(
					torch.LongTensor([self.multimodal.numliteral2id[i] for i in num_ind])))
			else:
				num_emb = emb

			# get the structural embeddings corresponding to numerical literals
			emb = self.ent_embeddings(torch.LongTensor(txt_ind))

			# enhance the textual entities with their literal embedding
			if len(txt_ind) > 0:
				txt_emb = self.multimodal.sen_txt_gate(emb, self.multimodal.textual_embeddings(
					torch.LongTensor([self.multimodal.textliteral2id[i] for i in txt_ind])))
			else:
				txt_emb = emb

			# create a batch of literal-enhanced embeddings
			e_emb = torch.zeros((e.shape[0], self.dim))

			for i, ind in enumerate(e_rel):
				e_emb[ind] = rel_emb[i]
			for i, ind in enumerate(e_num):
				e_emb[ind] = num_emb[i]
			for i, ind in enumerate(e_txt):
				e_emb[ind] = txt_emb[i]
		else:
			e_emb = self.ent_embeddings(torch.LongTensor(e))
			if self.multimodal.attributive:
				if self.multimodal.numerical_literals is not None and self.multimodal.textual_literals is not None:
					num_lit = self.multimodal.numerical_literals[e]
					txt_lit = self.multimodal.textual_literals[e]
					e_emb = self.multimodal.attr_gate(e_emb, num_lit, txt_lit)
				elif self.multimodal.numerical_literals is not None and self.multimodal.textual_literals is None:
					num_lit = self.multimodal.numerical_literals[e]
					e_emb = self.multimodal.attr_gate(e_emb, num_lit)
				elif self.multimodal.numerical_literals is None and self.multimodal.textual_literals is not None:
					txt_lit = self.multimodal.textual_literals[e]
					e_emb = self.multimodal.attr_gate(e_emb, txt_lit)

		return e_emb

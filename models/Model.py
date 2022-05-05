import torch
import torch.nn as nn
from BaseModule import BaseModule


class Model(BaseModule):

    def __init__(self, multimodal, loss=None, batch_size=128):
        super(Model, self).__init__()
        self.ent_tot = len(multimodal.entity2id)
        self.rel_tot = len(multimodal.relation2id)
        self.multimodal = multimodal
        self.loss = loss
        self.regul_rate = 0.0001
        self.l3_regul_rate = 0.0
        self.batch_size = batch_size

    def _forward(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def _get_positive_score(self, score):
        positive_score = score[:self.batch_size]
        positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)
        return positive_score

    def _get_negative_score(self, score):
        negative_score = score[self.batch_size:]
        negative_score = negative_score.view(-1, self.batch_size).permute(1, 0)
        return negative_score

    def forward(self, data):
        score = self._forward(data)
        p_score = self._get_positive_score(score)
        n_score = self._get_negative_score(score)
        loss_res = self.loss(p_score, n_score)
        if self.regul_rate != 0:
            loss_res += self.regul_rate * self.regularization(data)
        if self.l3_regul_rate != 0:
            loss_res += self.l3_regul_rate * self.l3_regularization()
        return loss_res
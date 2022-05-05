import sys
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import rankdata
from tqdm import tqdm

from Multimodal import Multimodal
from builddata import Utils
from loss.MarginLoss import MarginLoss
from loss.SoftplusLoss import SoftplusLoss
from models.DistMult import DistMult
from models.TransE import TransE
from pickling import *
from sampling.generic_trainer import GenericTrainer
from sampling.hybrid.batching import BatchLoader


class Trainer(GenericTrainer):

    def __init__(self, dataset_name, fold, types_per_entity=None, entities_per_type=None,
                 ontology=None, train_store=None, attributive=False, sensory=False, strict=False, neg_ratio=1.0):

        super().__init__()

        self.dataset_name = dataset_name
        self.fold = fold

        self.types_per_entity = types_per_entity
        self.entities_per_type = entities_per_type
        self.ontology = ontology
        self.train_store = train_store

        self.attributive = attributive
        self.sensory = sensory
        self.strict = strict
        self.neg_ratio = neg_ratio

        self.ent_embeddings = dict()
        self.rel_embeddings = dict()

        self.num_fc = None

    def train_and_eval(self, mode="TransE", train_data=None, test_data=None, valid_data=None):
        print("Getting", mode, "embeddings...")
        print("Preparing data...")
        self.mode = mode
        self.train_triples = train_data
        self.valid_triples = valid_data
        self.test_triples = test_data
        self.train, self.test, self.valid, self.word2index, self.index2word, \
        self.head_tail_selector, self.entity2id, self.id2entity, \
        self.relation2id, self.id2relation = Utils.build_data(train_data, test_data, valid_data, self.dataset_name)

        # ensure that the embeddings are always the same so that we don't have to recalculate the literal clusters
        if not os.path.isfile(os.path.join("pickles", "pickle_" + self.dataset_name, "fold-" + str(self.fold) + "_multimodal.pkl")):
            multimodal = Multimodal(self.train, self.valid, self.test,
                                self.word2index, self.index2word,
                                self.entity2id, self.id2entity,
                                self.id2relation, self.relation2id)
            dump_to_pickle(multimodal, self.dataset_name, "fold-" + str(self.fold) + "_multimodal.pkl")
        else:
            multimodal = load_from_pickle(self.dataset_name, "fold-" + str(self.fold) + "_multimodal.pkl")
            # synchronise these variables with what is stored inside multimodal
            # important for training routine...
            self.train, self.test, self.valid, \
            self.word2index, self.index2word, \
            self.entity2id, self.id2entity, \
            self.relation2id, self.id2relation = multimodal.train, multimodal.test, multimodal.valid, \
                                                 multimodal.word2index, multimodal.index2word, \
                                                 multimodal.entity2id, multimodal.id2entity, \
                                                 multimodal.relation2id, multimodal.id2relation
        multimodal.sensory = self.sensory
        multimodal.attributive = self.attributive

        batch = BatchLoader(multimodal, self.dataset_name, self.head_tail_selector,
                            types_per_entity=self.types_per_entity, entities_per_type=self.entities_per_type,
                            ontology=self.ontology, train_store=self.train_store, fold=self.fold,
                            batch_size=self.batch_size, neg_ratio=self.neg_ratio, strict=self.strict)

        if mode == "TransE":
            model = TransE(multimodal, MarginLoss(margin = 1.0), self.batch_size)
        elif mode == "DistMult":
            model = DistMult(multimodal, SoftplusLoss(), self.batch_size)

        self._start(batch, model, prec=True)

    def _start(self, batch, model, prec=False):
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        num_batches_per_epoch = int((len(self.train) - 1) / self.batch_size) + 1

        if prec:
            if not os.path.exists(os.path.join("pickles", "pickle_" + self.dataset_name, str(self.neg_ratio) + "_batches_hybrid.pkl")):
                batches = self.precompute(batch,
                                          self.num_epochs, num_batches_per_epoch, single=False)
                dump_to_pickle(batches, self.dataset_name, str(self.neg_ratio) + "_batches_hybrid.pkl")
            else:
                batches = load_from_pickle(self.dataset_name, str(self.neg_ratio) + "_batches_hybrid.pkl")
            batch.batches = iter(batches)

        for epoch in tqdm(range(self.num_epochs)):
            batch.reset()
            res = 0.0
            for batch_num in range(num_batches_per_epoch):
                x_batch, y_batch = batch()

                x_batch = self._convert_batch(x_batch)

                loss = model.forward(x_batch)

                opt.zero_grad()
                loss.backward()
                opt.step()
                res += loss.item()
            print("loss:", loss)

        # evaluate with test set...
        self._eval(model, batch)

    def _convert_batch(self, batch):
        new_h = torch.LongTensor([self.entity2id[self.index2word[x]] for x in batch[:, 0]])
        new_t = torch.LongTensor([self.entity2id[self.index2word[x]] for x in batch[:, 2]])
        new_r = torch.LongTensor([self.relation2id[self.index2word[x]]for x in batch[:, 1]])
        new_y = [1.0] * self.batch_size
        new_y.extend([-1.0] * self.batch_size)
        new_batch = dict()
        new_batch['batch_h'] = new_h
        new_batch['batch_t'] = new_t
        new_batch['batch_r'] = new_r
        new_batch['batch_y'] = torch.FloatTensor(new_y)
        new_batch['mode'] = "normal"
        return new_batch

    def _convert_htr(self, h, t, r):
        new_h = torch.LongTensor(h)
        new_t = torch.LongTensor(t)
        new_r = torch.LongTensor(r)
        new_batch = dict()
        new_batch['batch_h'] = new_h
        new_batch['batch_t'] = new_t
        new_batch['batch_r'] = new_r
        new_batch['mode'] = "normal"
        return new_batch

    def _eval(self, model, batch, is_valid=False):
        print("Start evaluation...")
        model.eval()
        num_splits = 8
        if is_valid:
            x_test = np.array(list(self.valid.keys())).astype(np.int32)
            y_test = np.array(list(self.valid.values())).astype(np.float32)
        else:
            x_test = np.array(list(self.test.keys())).astype(np.int32)
            y_test = np.array(list(self.test.values())).astype(np.float32)
        len_test = len(x_test)
        batch_test = int(len_test / (num_splits - 1))
        entity_array = np.array(list(self.entity2id.values()))

        print(len_test, "evaluation triples,", batch_test, "evaluation batches")

        def predict(ph, pt, pr):
            with torch.no_grad():
                # only make predictions for triples that make schematic sense
                new_ph, new_pt, new_pr = list(), list(), list()
                new_inds, old_inds = list(), list()
                for i, el in enumerate(ph):
                    tr = self.id2entity[ph[i]], self.id2relation[pr[i]], self.id2entity[pt[i]]
                    if batch.validate_triple(tr) < 2:
                        new_ph.append(ph[i])
                        new_pt.append(pt[i])
                        new_pr.append(pr[i])
                        new_inds.append(i)
                    else:
                        old_inds.append(i)
                new_ph = np.array(new_ph)
                new_pt = np.array(new_pt)
                new_pr = np.array(new_pr)
                new_batch = self._convert_htr(new_ph, new_pt, new_pr)
                pred = model.predict(new_batch)
                final_pred = np.zeros(ph.shape[0])
                for i, p in enumerate(pred):
                    final_pred[new_inds[i]] = p
                # add lowest prediction score to all remaining nonsensical triples
                for i in old_inds:
                    final_pred[i] = -sys.maxsize
                return pred

        def test_prediction(x_batch, y_batch, head_or_tail='head'):
            entity_weights = load_from_pickle(self.dataset_name, "entity_frequencies.pkl")
            relation_weights = load_from_pickle(self.dataset_name, "relation_frequencies.pkl")
            micro_hits1, micro_hits5, micro_hits10 = defaultdict(float), defaultdict(float), defaultdict(float)
            macro_hits1, macro_hits5, macro_hits10 = defaultdict(float), defaultdict(float), defaultdict(float)
            micro_mrr, macro_mrr = defaultdict(float), defaultdict(float)
            micro_mr, macro_mr = defaultdict(float), defaultdict(float)
            r_frequencies = defaultdict(int)
            e_frequencies = defaultdict(int)
            print("testing prediction...")
            for i in range(len(x_batch)):
                # we repeat the triple at index i in x_batch as many times as there are entities
                new_x_batch = np.tile(x_batch[i], (len(self.entity2id), 1))
                # should be all ones, as all triples in x_batch are real
                new_y_batch = np.tile(y_batch[i], (len(self.entity2id), 1))
                if head_or_tail == 'head':
                    new_x_batch[:, 0] = entity_array
                    new_x_batch[:, 2] = [self.entity2id[self.index2word[x]] for x in new_x_batch[:, 2]]
                else:  # 'tail'
                    new_x_batch[:, 2] = entity_array
                    new_x_batch[:, 0] = [self.entity2id[self.index2word[x]]for x in new_x_batch[:, 0]]
                # convert word ids for relations to relation specific ids; not necessary for entities,
                # as these have already been filled with only entities in the few lines of code above
                new_x_batch[:, 1] = [self.relation2id[self.index2word[x]]for x in new_x_batch[:, 1]]

                lstIdx = []
                for tmpIdxTriple in range(len(new_x_batch)):
                    # create a temporary corrupted triple
                    tmpTriple = (new_x_batch[tmpIdxTriple][0],
                                 new_x_batch[tmpIdxTriple][1],
                                 new_x_batch[tmpIdxTriple][2])
                    if (tmpTriple in self.train) or (tmpTriple in self.valid) or (tmpTriple in self.test):
                        lstIdx.append(tmpIdxTriple)
                new_x_batch = np.delete(new_x_batch, lstIdx, axis=0)
                new_y_batch = np.delete(new_y_batch, lstIdx, axis=0)

                # thus, insert the valid test triple again, to the beginning of the array
                # also needs to be converted to the correct indices
                valid_test_triple = [self.entity2id[self.index2word[x_batch[i, 0]]],
                                     self.relation2id[self.index2word[x_batch[i, 1]]],
                                     self.entity2id[self.index2word[x_batch[i, 2]]]]

                new_x_batch = np.insert(new_x_batch, 0, valid_test_triple, axis=0)
                # thus, the index of the valid test triple is equal to 0
                new_y_batch = np.insert(new_y_batch, 0, y_batch[i], axis=0)

                print("total batch size", new_x_batch.shape)

                results = []
                listIndexes = range(0, len(new_x_batch), (int(self.neg_ratio) + 1) * self.batch_size)
                for tmpIndex in range(len(listIndexes) - 1):
                    h_batch = new_x_batch[listIndexes[tmpIndex]:listIndexes[tmpIndex + 1], 0]
                    t_batch = new_x_batch[listIndexes[tmpIndex]:listIndexes[tmpIndex + 1], 2]
                    r_batch = new_x_batch[listIndexes[tmpIndex]:listIndexes[tmpIndex + 1], 1]
                    results = np.append(results, predict(h_batch, t_batch, r_batch))
                h_batch = new_x_batch[listIndexes[-1]:, 0]
                t_batch = new_x_batch[listIndexes[-1]:, 2]
                r_batch = new_x_batch[listIndexes[-1]:, 1]
                results = np.append(results, predict(h_batch, t_batch, r_batch))

                results = np.reshape(results, -1)

                print("predictions size:", results.shape)

                results_with_id = rankdata(results, method='ordinal')
                _filter = results_with_id[0]

                ent = self.index2word[x_batch[i, 0]]
                other = self.index2word[x_batch[i, 2]]
                if head_or_tail == "tail":
                    ent = self.index2word[x_batch[i, 2]]
                    other = self.index2word[x_batch[i, 0]]
                rel = self.index2word[x_batch[i, 1]]

                r_frequencies[rel] += 1
                e_frequencies[ent] += 1

                micro_mr[rel] += 1.0 * _filter
                macro_mr[rel] += 1.0 * _filter
                micro_mrr[rel] += (1.0 / _filter)
                macro_mrr[rel] += (1.0 / _filter)
                if _filter <= 1:
                    micro_hits1[rel] += 1.0
                    macro_hits1[rel] += 1.0
                if _filter <= 5:
                    micro_hits5[rel] += 1.0
                    macro_hits5[rel] += 1.0
                if _filter <= 10:
                    micro_hits10[rel] += 1.0
                    macro_hits10[rel] += 1.0

            # change micro-averages (relations are weighted according to test set frequency)
            # to macro-averages (relations are all weighted equally)

            batch_micro_hits1, batch_micro_hits5, batch_micro_hits10 = 0.0, 0.0, 0.0
            batch_macro_hits1, batch_macro_hits5, batch_macro_hits10 = 0.0, 0.0, 0.0
            batch_micro_mrr, batch_macro_mrr = 0.0, 0.0
            batch_micro_mr, batch_macro_mr = 0.0, 0.0
            for rel in macro_mr:
                macro_mr[rel] /= r_frequencies[rel]
                macro_mrr[rel] /= r_frequencies[rel]
                macro_hits1[rel] /= r_frequencies[rel]
                macro_hits5[rel] /= r_frequencies[rel]
                macro_hits10[rel] /= r_frequencies[rel]

                batch_micro_mr += micro_mr[rel]
                batch_micro_mrr += micro_mrr[rel]
                batch_micro_hits1 += micro_hits1[rel]
                batch_micro_hits5 += micro_hits5[rel]
                batch_micro_hits10 += micro_hits10[rel]

                batch_macro_mr += macro_mr[rel]
                batch_macro_mrr += macro_mrr[rel]
                batch_macro_hits1 += macro_hits1[rel]
                batch_macro_hits5 += macro_hits5[rel]
                batch_macro_hits10 += macro_hits10[rel]

            # this is used for micro-averages (total number of relationships in test triples)
            micro_num = max(len(x_batch), 1)
            # for macro-averages, use this instead (total number of distinct relationships in test triples)
            num = max(len(macro_mr.keys()), 1)
            return np.array([batch_micro_mr / micro_num,
                             batch_micro_mrr / micro_num,
                             batch_micro_hits1 / micro_num,
                             batch_micro_hits5 / micro_num,
                             batch_micro_hits10 / micro_num,

                             batch_macro_mr / num,
                             batch_macro_mrr / num,
                             batch_macro_hits1 / num,
                             batch_macro_hits5 / num,
                             batch_macro_hits10 / num
                             ])

        total_head_results = []
        total_tail_results = []
        for testIdx in tqdm(range(0, num_splits - 1)):
            head_results = test_prediction(
                x_test[batch_test * testIdx: batch_test * (testIdx + 1)],
                y_test[batch_test * testIdx: batch_test * (testIdx + 1)],
                head_or_tail='head')
            tail_results = test_prediction(
                x_test[batch_test * testIdx: batch_test * (testIdx + 1)],
                y_test[batch_test * testIdx: batch_test * (testIdx + 1)],
                head_or_tail='tail')
            total_head_results.append(head_results)
            total_tail_results.append(tail_results)
        head_results = test_prediction(x_test[batch_test * (num_splits - 1): len_test],
                                       y_test[batch_test * (num_splits - 1): len_test],
                                       head_or_tail='head')
        tail_results = test_prediction(x_test[batch_test * (num_splits - 1): len_test],
                                       y_test[batch_test * (num_splits - 1): len_test],
                                       head_or_tail='tail')
        total_head_results.append(head_results)
        total_tail_results.append(tail_results)

        agg_head_results = [0.0] * 10
        agg_tail_results = [0.0] * 10
        for h, t in zip(total_head_results, total_tail_results):
            agg_head_results = [x + y for x, y in zip(agg_head_results, h)]
            agg_tail_results = [x + y for x, y in zip(agg_tail_results, t)]
        total_head_results = [x / num_splits for x in agg_head_results]
        total_tail_results = [x / num_splits for x in agg_tail_results]
        total_results = [(x + y) / 2 for x, y in zip(total_head_results, total_tail_results)]

        print("head results:", total_head_results)
        print("tail results:", total_tail_results)
        print("results", total_results)

        strict = "strict"
        enhanced = "enhanced"
        if not self.strict:
            strict = "fuzzy"
        if not self.attributive:
            enhanced = "regular"
        wri = open("results/hybrid/" + self.mode + "_" + self.dataset_name + "_f" + str(self.fold) + "_"
                   + enhanced + "_" + strict + "_r" + str(self.neg_ratio), 'w')

        wri.write("micro_mr micro_mrr micro_hits1 micro_hits5 micro_hits10 "
                  "macro_mr macro_mrr macro_hits1 macro_hits5 macro_hits10\n")
        for _val in total_head_results:
            wri.write(str(_val) + ' ')
        wri.write('\n')
        for _val in total_tail_results:
            wri.write(str(_val) + ' ')
        wri.write('\n')
        for _val in total_results:
            wri.write(str(_val) + ' ')
        wri.write('\n')

        wri.close()
import numpy as np

from sampling.generic_batching import GenericBatchLoader


class BatchLoader(GenericBatchLoader):
    def __init__(self, multimodal, dataset_name, headTailSelector,
                 train_store=None, fold=0,
                 batch_size=128, neg_ratio=1.0):
        super().__init__(batch_size, neg_ratio)
        self.train_triples = multimodal.train
        self.test_triples = multimodal.test
        self.indexes = np.array(list(self.train_triples.keys())).astype(np.int32)
        self.values = np.array(list(self.train_triples.values())).astype(np.float32)
        self.fold = fold
        self.words_indexes = multimodal.word2index
        self.indexes_words = multimodal.index2word  # heads, relations, tails are also considered as words
        self.n_words = len(self.indexes_words)
        self.headTailSelector = headTailSelector
        self.relation2id = multimodal.relation2id
        self.id2relation = multimodal.id2relation
        self.entity2id = multimodal.entity2id
        self.id2entity = multimodal.id2entity

        self.num2id = multimodal.numliteral2id
        self.num_emb = multimodal.or_numerical_embeddings.numpy()

        self.text2id = multimodal.textliteral2id
        self.text_emb = multimodal.or_textual_embeddings.numpy()

        self.train_store = train_store
        self.ontology = None

        self.indexes_rels = {} # indices for relations between entities
        self.indexes_ents = {} # indices for entities
        for _word in self.words_indexes:
            index = self.words_indexes[_word]
            if _word in self.relation2id:
                self.indexes_rels[index] = _word
            elif _word in self.entity2id:
                self.indexes_ents[index] = _word

        # empty arrays to contain individual batches (both 0 and 1 values)
        self.new_triples_indexes = np.empty((self.batch_size * (self.neg_ratio + 1), 3)).astype(np.int32)
        self.new_triples_values = np.empty((self.batch_size * (self.neg_ratio + 1), 1)).astype(np.float32)

        # indices that remain to be explored in the current epoch
        self.remaining = range(0, len(self.values))

        self.illegals = set()

    def reset(self):
        self.remaining = range(0, len(self.values))

    def __call__(self):
        if type(self.batches) is not list:
            return super(BatchLoader, self).__call__()
        # there are as many idxs as can be fit inside a single batch (so, e.g. 128);
        # these are randomly selected; the corresponding triple indices are selected for these value indices
        # also, the corresponding values are selected for these value indices
        idxs = np.random.choice(self.remaining, min(len(self.remaining), self.batch_size), replace=False)
        # the last batch has to be filled up to the brim
        if len(self.remaining) < self.batch_size:
            idxs = np.hstack([idxs, np.random.randint(0, len(self.values), self.batch_size - len(self.remaining))])
        # makes sure that all indices are eventually sampled during training
        self.remaining = [idx for idx in self.remaining if idx not in idxs]
        self.new_triples_indexes[:self.batch_size, :] = self.indexes[idxs, :]
        self.new_triples_values[:self.batch_size] = self.values[idxs, :]

        last_idx = self.batch_size

        # if we require negative samples...
        if self.neg_ratio > 0:
            # Pre-sample everything, faster
            rdm_words = np.random.randint(0, self.n_words, last_idx * self.neg_ratio)
            # Pre-copy everything
            # np.tile repeats the first argument as many times as indicated by the second argument;
            # so neg_ratio along axis 0 and once along axis 1; in other words, these instructions
            # fill the second half (last_idx:2*last_idx) of the new_triples_indexes and
            # new_triples_values once (neg_ratio = 1) with the values of the first half (:last_idx),
            # which were already filled in previously (cfr. beginning of __call__())
            self.new_triples_indexes[last_idx:(last_idx * (self.neg_ratio + 1)), :] = np.tile(
                self.new_triples_indexes[:last_idx, :], (self.neg_ratio, 1))
            self.new_triples_values[last_idx:(last_idx * (self.neg_ratio + 1))] = np.tile(
                self.new_triples_values[:last_idx], (self.neg_ratio, 1))

            # iterate over the batch with i and over the number of negatives
            # with j (will always be 0 in our case, as neg_ratio is 1)
            valid = True
            for i in range(last_idx):
                for j in range(self.neg_ratio):
                    cur_idx = i * self.neg_ratio + j
                    # clearly, iterate over the second half of the batch
                    tmp_rel = self.indexes_words[self.new_triples_indexes[last_idx + cur_idx, 1]]
                    tmp_index_rel = self.relation2id[tmp_rel]
                    # get a head-tail probability in order to be able
                    # to generate a false triple
                    pr = self.headTailSelector[tmp_index_rel]

                    # Sample a random subject or object
                    if (np.random.randint(np.iinfo(np.int32).max) % 1000) > pr:
                        while ((self.ontology is not None and not valid) or
                            rdm_words[cur_idx] in self.indexes_rels or (
                            rdm_words[cur_idx],
                            self.new_triples_indexes[last_idx + cur_idx, 1],
                            self.new_triples_indexes[last_idx + cur_idx, 2]) in self.train_triples):
                            rdm_words[cur_idx] = np.random.randint(0, self.n_words)
                        # set the subject index: 2nd dimension = 0
                        self.new_triples_indexes[last_idx + cur_idx, 0] = rdm_words[cur_idx]
                    else:
                        while ((self.ontology is not None and not valid) or
                            rdm_words[cur_idx] in self.indexes_rels or (
                            self.new_triples_indexes[last_idx + cur_idx, 0],
                            self.new_triples_indexes[last_idx + cur_idx, 1],
                            rdm_words[cur_idx]) in self.train_triples):
                            rdm_words[cur_idx] = np.random.randint(0, self.n_words)
                        # set the object index: 2nd dimension = 2
                        self.new_triples_indexes[last_idx + cur_idx, 2] = rdm_words[cur_idx]
                    # the new triple does not exist and thus has value -1 instead of +1
                    if (self.new_triples_indexes[last_idx + cur_idx, 0],
                        self.new_triples_indexes[last_idx + cur_idx, 1],
                        self.new_triples_indexes[last_idx + cur_idx, 2]) in self.test_triples:
                        self.illegals.add((self.new_triples_indexes[last_idx + cur_idx, 0],
                                           self.new_triples_indexes[last_idx + cur_idx, 1],
                                           self.new_triples_indexes[last_idx + cur_idx, 2]))
                        print("nr of test triples encountered so far:", len(self.illegals))
                    self.new_triples_values[last_idx + cur_idx] = [-1]

            last_idx += cur_idx + 1

        # return the batch and the corresponding existential values
        return self.new_triples_indexes[:last_idx, :], self.new_triples_values[:last_idx]


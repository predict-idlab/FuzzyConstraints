from multiprocessing import cpu_count

import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from rdflib.term import Literal, URIRef
from sklearn import utils
from torch.autograd import Variable
from tqdm import tqdm

from gating import *


class Multimodal:

    def __init__(self, train, valid, test, word2index, index2word, entity2id, id2entity, id2relation, relation2id):
        self.embedding_size = 100
        self.train, self.valid, self.test = train, valid, test
        self.word2index = word2index
        self.index2word = index2word
        self.entity2id = entity2id
        self.id2entity = id2entity
        self.id2relation = id2relation
        self.relation2id = relation2id

        self.num_fc = None

        numerical_embeddings, numerical_literals, self.numliteral2id = self.get_numerical()
        textual_embeddings, textual_literals, self.textliteral2id = self.get_textual()

        print(numerical_literals.shape)
        print(textual_literals.shape)

        # Num. Literal
        # num_ent x n_num_lit
        self.numerical_literals = None
        self.n_num_lit = numerical_literals.shape[1]
        print("# numerical lit", self.n_num_lit)
        if self.n_num_lit > 0: self.numerical_literals = Variable(torch.from_numpy(numerical_literals))

        # Txt. Literal
        # num_ent x n_txt_lit
        self.textual_literals = None
        self.n_txt_lit = textual_literals.shape[1]
        print("# textual lit:", self.n_txt_lit)
        if self.n_txt_lit > 0: self.textual_literals = Variable(torch.from_numpy(textual_literals))

        # final gates for attributive view
        if self.n_num_lit > 0 and self.n_txt_lit > 0:
            print("using multi gate: both numerical and textual literals found!")
            self.attr_gate = MultiGate(self.embedding_size, self.n_num_lit, self.n_txt_lit)
        elif self.n_num_lit > 0 and self.n_txt_lit == 0:
            print("using single gate: only numerical literals found!")
            self.attr_gate = SingleGate(self.embedding_size, self.n_num_lit)
        elif self.n_num_lit == 0 and self.n_txt_lit > 0:
            print("using single gate: only textual literals found!")
            self.attr_gate = SingleGate(self.embedding_size, self.n_txt_lit)

        # preserve originals
        self.or_numerical_embeddings = torch.Tensor(numerical_embeddings)
        self.or_textual_embeddings = torch.Tensor(textual_embeddings)

        # sensory domain embeddings
        self.numerical_embeddings = torch.nn.Embedding.from_pretrained(torch.Tensor(numerical_embeddings),
                                                                       freeze=False, padding_idx=0)
        self.textual_embeddings = torch.nn.Embedding.from_pretrained(torch.Tensor(textual_embeddings),
                                                                     freeze=False, padding_idx=0)

        # gates for sensory view
        self.sen_num_gate = Gate(self.embedding_size + self.embedding_size, self.embedding_size)
        self.sen_txt_gate = Gate(self.embedding_size + self.embedding_size, self.embedding_size)

    def get_numerical(self):
        train_num_embeddings, train_num_literals = self._get_numerical_embeddings(self.train, literals=True)
        valid_num_embeddings = self._get_numerical_embeddings(self.valid)
        test_num_embeddings = self._get_numerical_embeddings(self.test)
        numerical_embeddings = {**train_num_embeddings, **test_num_embeddings, **valid_num_embeddings}

        numliteral2id = dict()
        num_embeddings = list()
        for i, literal in enumerate(list(numerical_embeddings.keys())):
            numliteral2id[literal] = i
            num_embeddings.append(numerical_embeddings[literal])

        if len(num_embeddings) == 0: num_embeddings = [[]]

        return num_embeddings, train_num_literals, numliteral2id

    def get_textual(self):
        train_txt_embeddings, tags, docs, train_txt_literals = self._get_textual_embeddings(self.train, literals=True)
        valid_txt_embeddings, _, _ = self._get_textual_embeddings(self.valid, tags, docs)
        test_txt_embeddings, _, _ = self._get_textual_embeddings(self.test, tags, docs)
        textual_embeddings = {**train_txt_embeddings, **test_txt_embeddings, **valid_txt_embeddings}

        txt_embeddings = list()
        txtliteral2id = dict()
        for i, literal in enumerate(list(textual_embeddings.keys())):
            txtliteral2id[literal] = i
            txt_embeddings.append(textual_embeddings[literal])

        if len(txt_embeddings) == 0: txt_embeddings = [[]]

        return txt_embeddings, train_txt_literals, txtliteral2id

    def _get_numerical_embeddings(self, triples, literals=False):
        print("getting numerical literals...")
        numerical_embeddings = dict()
        if not self.num_fc:
            self.num_fc = torch.nn.Linear(1, self.embedding_size)

        relations = np.zeros((len(self.id2entity.keys()), len(self.id2relation.keys())), dtype=object)
        for triple in triples:
            obj = self.index2word[triple[2]]
            if type(obj) is Literal:
                if obj.datatype == URIRef('http://www.w3.org/2001/XMLSchema#decimal') \
                        or obj.datatype == URIRef('http://www.w3.org/2001/XMLSchema#integer') \
                        or obj.datatype == URIRef('http://www.w3.org/2001/XMLSchema#double') \
                        or obj.datatype == URIRef('http://www.w3.org/2001/XMLSchema#float') \
                        or obj.datatype == URIRef('http://www.w3.org/2001/XMLSchema#short') \
                        or obj.datatype == URIRef('http://www.w3.org/2001/XMLSchema#int') \
                        or obj.datatype == URIRef('http://www.w3.org/2001/XMLSchema#long') \
                        or obj.datatype == URIRef('http://www.w3.org/2001/XMLSchema#string'):
                    # print(obj)
                    number = obj.toPython()
                    if type(number) is not str:
                        with torch.no_grad():
                            s, r, o = self.entity2id[self.index2word[triple[0]]], \
                                      self.relation2id[self.index2word[triple[1]]], \
                                      self.entity2id[self.index2word[triple[2]]]
                            numerical_embedding = self.num_fc(torch.Tensor([number]))
                            numerical_embeddings[o] = numerical_embedding.tolist()
                            relations[s, r] = o

        if len(numerical_embeddings) > 0:
            # axis 0 indicates that we are normalising per feature: this is what we want!
            max_lit, min_lit = np.max(list(numerical_embeddings.values()), axis=0), np.min(list(numerical_embeddings.values()), axis=0)
            numerical_embeddings = {k: (v - min_lit) / (max_lit - min_lit + 1e-8) for k, v in numerical_embeddings.items()}

        if literals:
            useful_dimensions = relations.shape[1]
            redundant_dimensions = []
            for i, col in enumerate(relations.T):
                if np.isin(col, [0]).all():
                    redundant_dimensions.append(i)

            relations = np.delete(relations, redundant_dimensions, axis=1)
            useful_dimensions -= len(redundant_dimensions)
            print(useful_dimensions, "useful dimensions found!")

            numerical_literals = np.zeros((len(self.entity2id.keys()),
                                           useful_dimensions, self.embedding_size))
            print(numerical_literals.shape)
            for i, _ in enumerate(numerical_literals):
                for j, __ in enumerate(numerical_literals[i]):
                    if relations[i, j] in numerical_embeddings:
                        numerical_literals[i, j] = numerical_embeddings[relations[i, j]]
            return numerical_embeddings, numerical_literals

        return numerical_embeddings

    def _get_textual_embeddings(self, triples, tags=list(), documents=list(), literals=False):
        print("getting textual literals...")
        if len(tags) > 0:
            tags = list(tags)
        if len(documents) > 0:
            documents = list(documents)
        new_tags = list()

        relations = np.zeros((len(self.id2entity.keys()), len(self.id2relation.keys())), dtype=object)
        for triple in triples:
            obj = self.index2word[triple[2]]
            if type(obj) is Literal:
                # also check for empty datatype as this is syntactic sugar for a simple string
                if obj.datatype == URIRef('http://www.w3.org/2001/XMLSchema#string') \
                        or obj.datatype is None:
                    text = str(obj.toPython())
                    s, r, o = self.entity2id[self.index2word[triple[0]]], \
                              self.relation2id[self.index2word[triple[1]]], \
                              self.entity2id[self.index2word[triple[2]]]
                    tags.append(o)
                    new_tags.append(tags[-1])
                    documents.append(TaggedDocument(text, [tags[-1]]))
                    relations[s, r] = o

        textual_embeddings = dict()
        if len(documents) > 0:
            # train the doc2vec model to get text embeddings for each literal entity
            model = Doc2Vec(vector_size=self.embedding_size, window=5, workers=cpu_count())
            model.build_vocab([doc for doc in tqdm(documents)])
            for _ in tqdm(range(20)):
                model.train(utils.shuffle([doc for doc in documents]),
                            total_examples=model.corpus_count, epochs=1)
                model.alpha -= 0.002
                model.min_alpha = model.alpha

            for doc in new_tags:
                textual_embeddings[doc] = model.docvecs[doc]

        if literals:
            useful_dimensions = relations.shape[1]
            redundant_dimensions = []
            for i, col in enumerate(relations.T):
                if np.isin(col, [0]).all():
                    redundant_dimensions.append(i)

            relations = np.delete(relations, redundant_dimensions, axis=1)
            useful_dimensions -= len(redundant_dimensions)
            print(useful_dimensions, "useful dimensions found!")

            textual_literals = np.zeros((len(self.id2entity.keys()),
                                         useful_dimensions, self.embedding_size))
            for i, _ in enumerate(textual_literals):
                for j, __ in enumerate(textual_literals[i]):
                    if relations[i, j] in textual_embeddings:
                        textual_literals[i, j] = textual_embeddings[relations[i, j]]
            return textual_embeddings, tags, documents, textual_literals
        return textual_embeddings, tags, documents
from collections import defaultdict

import numpy as np
from rdflib.term import URIRef

from sampling.generic_batching import GenericBatchLoader


class BatchLoader(GenericBatchLoader):
    def __init__(self, multimodal, dataset_name, headTailSelector,
                 types_per_entity=None, entities_per_type=None,
                 ontology=None, train_store=None, fold=0,
                 batch_size=128, neg_ratio=1.0,
                 strict=True):
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

        self.strict = strict

        self.ontology = ontology
        self.train_store = train_store

        self.types_per_entity = types_per_entity
        self.entities_per_type = entities_per_type

        self.set_constraints(convert=True)

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
            for i in range(last_idx):
                for j in range(self.neg_ratio):
                    cur_idx = i * self.neg_ratio + j
                    # clearly, iterate over the second half of the batch
                    tmp_rel = self.indexes_words[self.new_triples_indexes[last_idx + cur_idx, 1]]
                    tmp_index_rel = self.relation2id[tmp_rel]
                    # get a head-tail probability in order to be able
                    # to generate a false triple
                    pr = self.headTailSelector[tmp_index_rel]

                    num_cycles = 0
                    # Sample a random subject or object
                    if (np.random.randint(np.iinfo(np.int32).max) % 1000) > pr:
                        # valid = True
                        valid = self._validate_perturbation(rdm_words, cur_idx, last_idx, head=True)
                        # we keep looking until we find a valid perturbation...
                        while ((self.ontology is not None and not valid) or
                            rdm_words[cur_idx] in self.indexes_rels or (
                            rdm_words[cur_idx],
                            self.new_triples_indexes[last_idx + cur_idx, 1],
                            self.new_triples_indexes[last_idx + cur_idx, 2]) in self.train_triples):
                            rdm_words[cur_idx] = np.random.randint(0, self.n_words)
                            if num_cycles < 100:
                                valid = self._validate_perturbation(rdm_words, cur_idx, last_idx, head=True)
                                num_cycles += 1
                            else:
                                valid = True
                        # set the subject index: 2nd dimension = 0
                        self.new_triples_indexes[last_idx + cur_idx, 0] = rdm_words[cur_idx]
                    else:
                        # valid = True
                        valid = self._validate_perturbation(rdm_words, cur_idx, last_idx, head=False)
                        # we keep looking until we find a valid perturbation...
                        while ((self.ontology is not None and not valid) or
                            rdm_words[cur_idx] in self.indexes_rels or (
                            self.new_triples_indexes[last_idx + cur_idx, 0],
                            self.new_triples_indexes[last_idx + cur_idx, 1],
                            rdm_words[cur_idx]) in self.train_triples):
                            rdm_words[cur_idx] = np.random.randint(0, self.n_words)
                            if num_cycles < 100:
                                valid = self._validate_perturbation(rdm_words, cur_idx, last_idx, head=False)
                                num_cycles += 1
                            else:
                                valid = True
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

    def _validate_perturbation(self, rdm_words, cur_idx, last_idx, head=True):
        if head:
            triple = (self.indexes_words[rdm_words[cur_idx]],
                      self.indexes_words[self.new_triples_indexes[last_idx + cur_idx, 1]],
                      self.indexes_words[self.new_triples_indexes[last_idx + cur_idx, 2]])
        else:
            triple = (self.indexes_words[self.new_triples_indexes[last_idx + cur_idx, 0]],
                      self.indexes_words[self.new_triples_indexes[last_idx + cur_idx, 1]],
                      self.indexes_words[rdm_words[cur_idx]])
        return self.validate_triple(triple) < 2

    def validate_triple(self, triple):
        """
        Validate a triple against schema constraints for a given predicate.
        :param triple:
        :return:
        """
        stypes = self.types_per_entity[triple[0]]
        otypes = self.types_per_entity[triple[2]]

        # check simple rdfs restrictions
        if triple[1] in self.prop2def:
            dom, ran, status = self.prop2def[triple[1]]
            if status:
                # interpretation of rdfs domain is AND
                if not (len(stypes) > 0 and all([t in stypes for t in dom])):
                    return 2
                # interpretation of rdfs range is AND
                if not (len(otypes) > 0 and all([t in otypes for t in ran])):
                    return 2
                return 0
            else:
                # interpretation of owl constraints is nested
                if len(stypes) > 0 and any([t in stypes for t in dom]):
                    if len(otypes) > 0 and all([t in otypes for t in ran]):
                        return 0
                    return 2
                return 0
        return 1

    def set_constraints(self, convert=False):
        print("setting constraints...")
        # get owl restrictions (more specific than domain and range, allows to define a range contingent on a domain)
        restrictions = set(self.ontology.subjects(predicate=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
                                                   object=URIRef("http://www.w3.org/2002/07/owl#Restriction")))

        print(len(restrictions), "owl restrictions found")
        # get corresponding properties
        self.prop2rest = defaultdict(list)
        for rest in restrictions:
            properties_for_rest = self.ontology.objects(subject=rest, predicate=URIRef("http://www.w3.org/2002/07/owl#onProperty"))
            for prop in properties_for_rest:
                print("found restriction", rest, "for property", prop)
                self.prop2rest[prop].append(rest)

        # get domain and range
        self.rest2def = dict()
        for rest in restrictions:
            # the domain consists of those classes that are declared as subclasses of the restriction
            rest_domain = set(self.ontology.subjects(predicate=URIRef("http://www.w3.org/2000/01/rdf-schema#subClassOf"), object=rest))
            # the range is determined by the allValuesFrom/someValuesFrom properties
            # however, someValuesFrom does not allow us to look on a per-triple basis
            # since it acts as an existential rather than universal operator
            rest_range = set(self.ontology.objects(subject=rest, predicate=URIRef("http://www.w3.org/2002/07/owl#allValuesFrom")))

            print("found domain", rest_domain, "and range", rest_range, "for", rest)

            dom, ran = set(), set()
            if len(rest_domain) > 0: dom = rest_domain
            if len(rest_range) > 0: ran = rest_range

            # boolean used to indicate true rdfs constraint or owl constraint
            self.rest2def[rest] = (dom, ran, False)

        self.prop2def = dict()
        domain_properties = set(self.ontology.subjects(predicate=URIRef("http://www.w3.org/2000/01/rdf-schema#domain"), object=None))
        range_properties = set(self.ontology.subjects(predicate=URIRef("http://www.w3.org/2000/01/rdf-schema#range"), object=None))
        all_properties = domain_properties
        all_properties = all_properties.union(range_properties)
        print("all properties with rdfs constraints:", all_properties)
        for property in all_properties:
            prop_domain = set(self.ontology.objects(subject=property, predicate=URIRef("http://www.w3.org/2000/01/rdf-schema#domain")))
            prop_range = set(self.ontology.objects(subject=property, predicate=URIRef("http://www.w3.org/2000/01/rdf-schema#range")))

            dom, ran = set(), set()
            if len(prop_domain) > 0: dom = prop_domain
            if len(prop_range) > 0: ran = prop_range

            # boolean used to indicate true rdfs constraint or owl constraint
            self.prop2def[property] = (dom, ran, True)

        print(len(self.prop2def), "rdfs constraints found")

        if convert:
            # convert OWL constraints to RDFS constraints
            for prop in self.prop2rest:
                for rest in self.prop2rest[prop]:
                    self.prop2def[prop] = self.rest2def[rest]
                    print(prop, self.prop2def[prop])
                    # manually fix axiomatic expansion for allValuesFrom
                    # normally, entities participating in a given relation are attributed the types given by
                    # the domain and range axioms; this is not the case with owl restrictions
                    # so, in addition to the work we did in test.py, we need to do this also
                    for obj in set(self.train_store.objects(subject=None, predicate=prop)):
                        self.types_per_entity[obj].union(set(self.prop2def[prop][1]))
                        for t in self.prop2def[prop][1]:
                            self.entities_per_type[t].add(obj)
            print(len(self.prop2def), "rdfs constraints found after conversion")


from collections import defaultdict
from random import random

import numpy as np
from rdflib.term import URIRef, Literal
from scipy.spatial.distance import cosine
from tqdm import tqdm

from pickling import *
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

        self.create_fuzzy_sets(dataset_name)
        self.set_clusters()

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
                        valid = self._validate_perturbation(rdm_words, cur_idx, last_idx,
                                                            self.new_triples_indexes[last_idx + cur_idx, 0],
                                                            head=True)
                        # we keep looking until we find a valid perturbation...
                        while ((self.ontology is not None and not valid) or
                            rdm_words[cur_idx] in self.indexes_rels or (
                            rdm_words[cur_idx],
                            self.new_triples_indexes[last_idx + cur_idx, 1],
                            self.new_triples_indexes[last_idx + cur_idx, 2]) in self.train_triples):
                            rdm_words[cur_idx] = np.random.randint(0, self.n_words)
                            if num_cycles < 100:
                                valid = self._validate_perturbation(rdm_words, cur_idx, last_idx,
                                                                    self.new_triples_indexes[last_idx + cur_idx, 0],
                                                                    head=True)
                                num_cycles += 1
                            else:
                                valid = True
                        # set the subject index: 2nd dimension = 0
                        self.new_triples_indexes[last_idx + cur_idx, 0] = rdm_words[cur_idx]
                    else:
                        # valid = True
                        valid = self._validate_perturbation(rdm_words, cur_idx, last_idx,
                                                            self.new_triples_indexes[last_idx + cur_idx, 2],
                                                            head=False)
                        # we keep looking until we find a valid perturbation...
                        while ((self.ontology is not None and not valid) or
                            rdm_words[cur_idx] in self.indexes_rels or (
                            self.new_triples_indexes[last_idx + cur_idx, 0],
                            self.new_triples_indexes[last_idx + cur_idx, 1],
                            rdm_words[cur_idx]) in self.train_triples):
                            rdm_words[cur_idx] = np.random.randint(0, self.n_words)
                            if num_cycles < 100:
                                valid = self._validate_perturbation(rdm_words, cur_idx, last_idx,
                                                                    self.new_triples_indexes[last_idx + cur_idx, 2],
                                                                    head=False)
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

    def _validate_perturbation(self, rdm_words, cur_idx, last_idx, prev_word, dist=True, head=True):
        pr = random()

        if head:
            triple = (self.indexes_words[rdm_words[cur_idx]],
                      self.indexes_words[self.new_triples_indexes[last_idx + cur_idx, 1]],
                      self.indexes_words[self.new_triples_indexes[last_idx + cur_idx, 2]])
            if self.validate_triple(triple) < 2:
                if (triple[1], triple[0]) in self.domains:
                    return self.domains[(triple[1], triple[0])] <= pr
        else:
            triple = (self.indexes_words[self.new_triples_indexes[last_idx + cur_idx, 0]],
                      self.indexes_words[self.new_triples_indexes[last_idx + cur_idx, 1]],
                      self.indexes_words[rdm_words[cur_idx]])
            if self.validate_triple(triple) < 2:
                if (triple[1], triple[2]) in self.ranges:
                    return self.ranges[(triple[1], triple[2])] <= pr

        return False

    def validate_entity(self, predicate, entity, domain=True):
        """
        Validate an entity against schema constraints for a given predicate.
        :param predicate:
        :param entity:
        :param domain:
        :return:
        """
        types = self.types_per_entity[entity]

        # check simple rdfs restrictions
        if predicate in self.prop2def:
            if domain:
                rest, _, status = self.prop2def[predicate]
                if status:
                    # interpretation of rdfs domain is AND
                    if len(types) > 0 and all([t in types for t in rest]):
                        return 0
                else:
                    # interpretation of owl domain is OR
                    if len(types) > 0 and any([t in types for t in rest]):
                        return 0
            else:
                _, rest, status = self.prop2def[predicate]
                # interpretation of range is AND
                if len(types) > 0 and all([t in types for t in rest]):
                    return 0
            return 2
        return 1

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

    def create_fuzzy_sets(self, name):
        if os.path.isfile(os.path.join("pickles", "pickle_" + name, "fold-" + str(self.fold) + "_domains.pkl")) and \
                    os.path.isfile(os.path.join("pickles", "pickle_" + name, "fold-" + str(self.fold) + "_ranges.pkl")):
            self.domains = load_from_pickle(name, "fold-" + str(self.fold) + "_domains.pkl")
            self.ranges = load_from_pickle(name, "fold-" + str(self.fold) + "_ranges.pkl")
            self.prop2def = load_from_pickle(name, "fold-" + str(self.fold) + "_prop2def.pkl")
            self.num_clusters = load_from_pickle(name, "fold-" + str(self.fold) + "_num_clusters.pkl")
            self.text_clusters = load_from_pickle(name, "fold-" + str(self.fold) + "_text_clusters.pkl")
            return
        else:
            self.set_constraints(convert=True)
            self.set_literal_clusters()
            dump_to_pickle(self.prop2def, name, "fold-" + str(self.fold) + "_prop2def.pkl")
            dump_to_pickle(self.num_clusters, name, "fold-" + str(self.fold) + "_num_clusters.pkl")
            dump_to_pickle(self.text_clusters, name, "fold-" + str(self.fold) + "_text_clusters.pkl")
            # these also need to be updated, if convert = True
            dump_to_pickle(self.types_per_entity, name, "fold-" + str(self.fold) + "_types_per_entity.pkl")
            dump_to_pickle(self.entities_per_type, name, "fold-" + str(self.fold) + "_entities_per_type.pkl")

        print("creating fuzzy sets...")
        all_properties = set(self.train_store.predicates(subject=None, object=None))

        top_level_classes = {
                                URIRef("http://www.w3.org/2000/01/rdf-schema#Resource"),
                                URIRef("http://www.w3.org/2002/07/owl#Thing")
                            }

        print("calculate cardinalities...")

        self.domain_cardinality = defaultdict(list)
        self.range_cardinality = defaultdict(list)
        for p in tqdm(all_properties):
            O_p = set(self.train_store.objects(subject=None, predicate=p))
            SN, SN_reverse, SN_max = defaultdict(list), defaultdict(list), 0
            for o in O_p:
                SN[(o, p)] = len(set(self.train_store.subjects(predicate=p, object=o)))
                SN_reverse[SN[(o, p)]].append(o)
                if SN[(o, p)] > SN_max:
                    SN_max = SN[(o, p)]
            for i in range(SN_max):
                self.domain_cardinality[p].append(len(SN_reverse[i + 1]) / len(O_p))

            S_p = set(self.train_store.subjects(predicate=p, object=None))
            ON, ON_reverse, ON_max = defaultdict(list), defaultdict(list), 0
            for s in S_p:
                ON[(s, p)] = len(set(self.train_store.objects(subject=s, predicate=p)))
                ON_reverse[ON[(s, p)]].append(s)
                if ON[(s, p)] > ON_max:
                    ON_max = ON[(s, p)]
            for i in range(ON_max):
                self.range_cardinality[p].append(len(ON_reverse[i + 1]) / len(S_p))

        self.domain_sp, self.domain_ge = dict(), dict()
        self.range_sp, self.range_ge = dict(), dict()

        def calc_sd(c, entities, classes):
            """
            Calculate the supporting degrees.
            :param classes:
            :param c:
            :param entities:
            :return:
            """
            relevant_entities = {e for e in entities if c in classes[e]}
            return len(relevant_entities) / len(entities)

        print("calculate hierarchical property domain/range...")

        def prune_hierarchies(hierarchies):
            """
            Prune the hierarchy by removing classical solipsism and nihilism.
            :param hierarchies:
            :return:
            """
            new_classes = set()
            for tr in hierarchies:
                if tr[0] != tr[2] and tr[0] != URIRef("http://www.w3.org/2002/07/owl#Nothing"):
                    new_classes.add(tr)
            return new_classes


        all_CS_sp, all_CS_ge = dict(), dict()
        all_CO_sp, all_CO_ge = dict(), dict()
        for p in tqdm(all_properties):
            SE_p = set(self.train_store.subjects(predicate=p, object=None))
            OE_p = set(self.train_store.objects(subject=None, predicate=p))

            print("get subject hierarchies...")
            CS_sp, CS_ge = set(), set()
            CS_sp_per_subject, CS_ge_per_subject = defaultdict(set), defaultdict(set)
            for s in tqdm(SE_p):
                types = self.types_per_entity[s]
                for t in types:
                    hierarchies = prune_hierarchies(set(
                        self.ontology.triples((None, URIRef("http://www.w3.org/2000/01/rdf-schema#subClassOf"), t))))
                    if t not in top_level_classes:
                        if len(hierarchies) > 0:
                            CS_ge.add(t)
                            CS_ge_per_subject[s].add(t)
                        else:
                            CS_sp.add(t)
                            CS_sp_per_subject[s].add(t)

            print("get object hierarchies...")
            CO_sp, CO_ge = set(), set()
            CO_sp_per_object, CO_ge_per_object = defaultdict(set), defaultdict(set)
            for o in tqdm(OE_p):
                types = self.types_per_entity[o]
                for t in types:
                    hierarchies = prune_hierarchies(set(
                        self.ontology.triples((None, URIRef("http://www.w3.org/2000/01/rdf-schema#subClassOf"), t))))
                    if t not in top_level_classes:
                        if len(hierarchies) > 0:
                            CO_ge.add(t)
                            CO_ge_per_object[o].add(t)
                        else:
                            CO_sp.add(t)
                            CO_sp_per_object[o].add(t)

            all_CS_sp[p] = CS_sp
            all_CS_ge[p] = CS_ge
            all_CO_sp[p] = CO_sp
            all_CO_ge[p] = CO_ge

            print("calc sd for specific subject types...")
            for c in tqdm(CS_sp):
                self.domain_sp[(p, c)] = calc_sd(c, SE_p, CS_sp_per_subject)
                print(p, c, self.domain_sp[(p, c)])
            print("calc sd for generic subject types...")
            for c in tqdm(CS_ge):
                self.domain_ge[(p, c)] = calc_sd(c, SE_p, CS_ge_per_subject)
                print(p, c, self.domain_ge[(p, c)])
            print("calc sd for specific object types...")
            for c in tqdm(CO_sp):
                self.range_sp[(p, c)] = calc_sd(c, OE_p, CO_sp_per_object)
                print(p, c, self.range_sp[(p, c)])
            print("calc sd for generic object types...")
            for c in tqdm(CO_ge):
                self.range_ge[(p, c)] = calc_sd(c, OE_p, CO_ge_per_object)
                print(p, c, self.range_ge[(p, c)])

        def cardinality_score(predicate, n, summation, maximum, inv_maximum, domain=True):
            car_score = 0.0
            if domain:
                r = (n - maximum) * inv_maximum
                if not (SN_max == 0 or r >= 0.8):
                    # case: inverse functional property
                    if n == 1:
                        car_score = self.domain_cardinality[predicate][0]
                    else:
                        car_score = summation
                        if r > 0:
                            car_score *= (1 - r) * car_score
            else:
                r = (n - maximum) * inv_maximum
                if not (ON_max == 0 or r >= 0.8):
                    # case: functional property
                    if n == 1:
                        car_score = self.range_cardinality[predicate][0]
                    else:
                        car_score = summation
                        if r > 0:
                            car_score *= (1 - r) * car_score

            return car_score

        def membership(predicate, entity, domain=True):
            """
            Calculate the fuzzy membership of a given entity.
            :param predicate:
            :param entity:
            :param domain:
            :return:
            """
            con_score_sp, con_score_ge = 1.0, 1.0
            if domain:
                stypes = set(self.types_per_entity[entity])
                stypes = stypes.intersection(all_CS_sp[predicate])
                for t in stypes:
                    con_score_sp *= 1 - self.domain_sp[(predicate, t)]
                con_score_sp = 1 - con_score_sp

                stypes = set(self.types_per_entity[entity])
                stypes = stypes.intersection(all_CS_ge[predicate])
                for t in stypes:
                    con_score_ge *= 1 - self.domain_ge[(predicate, t)]
                con_score_ge = 1 - con_score_ge

                if type(entity) is Literal:
                    lit_score = 0.0
                    if entity in self.num2id:
                        cluster = self.num_clusters[predicate]
                        lit_score = max(lit_score, (1.5 - (cosine(cluster[0], self.num_emb[self.num2id[entity]]) / cluster[1]) / 1.5))
                    elif entity in self.text2id:
                        cluster = self.text2id[predicate]
                        lit_score = max(lit_score, (1.5 - (cosine(cluster[0], self.text_emb[self.text2id[entity]]) / cluster[1]) / 1.5))
                    return 0.25 * (0.2 * con_score_ge + 0.8 * con_score_sp) + 0.75 * lit_score

            else:
                otypes = set(self.types_per_entity[entity])
                otypes = otypes.intersection(all_CO_sp[predicate])
                for t in otypes:
                    con_score_sp *= 1 - self.range_sp[(predicate, t)]
                con_score_sp = 1 - con_score_sp

                otypes = set(self.types_per_entity[entity])
                otypes = otypes.intersection(all_CO_ge[predicate])
                for t in otypes:
                    con_score_ge *= 1 - self.range_ge[(predicate, t)]
                con_score_ge = 1 - con_score_ge

                if type(entity) is Literal:
                    lit_score = 0.0
                    if entity in self.num2id:
                        cluster = self.num_clusters[predicate]
                        lit_score = max(lit_score, (1.5 - (cosine(cluster[0], self.num_emb[self.num2id[entity]]) / cluster[1]) / 1.5))
                    elif entity in self.text2id:
                        cluster = self.text2id[predicate]
                        lit_score = max(lit_score, (1.5 - (cosine(cluster[0], self.text_emb[self.text2id[entity]]) / cluster[1]) / 1.5))
                    return 0.25 * (0.2 * con_score_ge + 0.8 * con_score_sp) + 0.75 * lit_score

            return 0.2 * con_score_ge + 0.8 * con_score_sp

        print("calculate domain and range sets...")

        domain_sets, range_sets = dict(), dict()
        for p in tqdm(all_properties):
            for e in tqdm(list(self.entity2id.keys())):
                domain_sets[(p, e)] = set(self.train_store.subjects(predicate=p, object=e))
                range_sets[(p, e)] = set(self.train_store.objects(subject=e, predicate=p))

        print("calculate n values...")

        domain_ns = defaultdict(float)
        range_ns = defaultdict(float)
        for p in tqdm(all_properties):
            for e in tqdm(list(self.entity2id.keys())):
                domain_ns[(p, e)] = sum({len(domain_sets[(p, o)] | {e}) for o in self.entity2id.keys()})
                range_ns[(p, e)] = sum({len(range_sets[(p, o)] | {e}) for o in self.entity2id.keys()})
                domain_ns[(p, e)] /= len(self.entity2id.keys())
                range_ns[(p, e)] /= len(self.entity2id.keys())

        print("calculate cardinality scores...")

        domain_car_scores = dict()
        range_car_scores = dict()
        # first calculate cardinalities
        for p in tqdm(all_properties):
            SN_max, ON_max = len(self.domain_cardinality[p]), len(self.range_cardinality[p])
            inv_SN_max, inv_ON_max = (1 / SN_max), (1 / ON_max)
            domain_summation = sum([self.domain_cardinality[p][i] for i in range(1, SN_max)])
            range_summation = sum([self.range_cardinality[p][i] for i in range(1, ON_max)])
            for e in tqdm(list(self.entity2id.keys())):
                domain_car_scores[(p, e)] = cardinality_score(p, domain_ns[(p, e)],
                                                                domain_summation, SN_max, inv_SN_max, domain=True)
                range_car_scores[(p, e)] = cardinality_score(p, range_ns[(p, e)],
                                                                range_summation, ON_max, inv_ON_max, domain=False)

        print("calculate membership values...")

        # construct membership values for fuzzy sets
        self.domains, self.ranges = dict(), dict()
        for p in tqdm(all_properties):
            for e in list(self.entity2id.keys()):
                # check fuzzy conformance
                self.domains[(p, e)] = 0.5 * domain_car_scores[(p, e)] + 0.5 * membership(p, e, domain=True)
                self.ranges[(p, e)] = 0.5 * range_car_scores[(p, e)] + 0.5 * membership(p, e, domain=False)

        # ensure that membership score is always calculated for all constrained properties,
        # even if they are not contained within the training set
        domain_properties = set(
            self.ontology.subjects(predicate=URIRef("http://www.w3.org/2000/01/rdf-schema#domain"), object=None))
        range_properties = set(
            self.ontology.subjects(predicate=URIRef("http://www.w3.org/2000/01/rdf-schema#range"), object=None))
        other_properties = domain_properties
        other_properties.union(range_properties).difference(all_properties)
        for p in tqdm(other_properties):
            for e in list(self.entity2id.keys()):
                # this should always be true, but we check nonetheless
                if (p, e) not in self.domains:
                    # only check exact conformance
                    self.domains[(p, e)] = 1.0 if self.validate_entity(p, e, domain=True) else 0.0
                # this should always be true, but we check nonetheless
                if (p, e) not in self.ranges:
                    # only check exact conformance
                    self.ranges[(p, e)] = 1.0 if self.validate_entity(p, e, domain=False) else 0.0

        domain_max = defaultdict(float)
        range_max = defaultdict(float)

        for (p, e) in self.domains:
            domain_max[p] = max(domain_max[p], self.domains[(p, e)])
            range_max[p] = max(range_max[p], self.ranges[(p, e)])

        # normalise domains and ranges against max (only when not using lambda-cut)
        for (p, e) in self.domains:
            if domain_max[p] > 0:
                self.domains[(p, e)] *= (1 / domain_max[p])
            if range_max[p] > 0:
                self.ranges[(p, e)] *= (1 / range_max[p])

        dump_to_pickle(self.domains, name, "fold-" + str(self.fold) + "_domains.pkl")
        dump_to_pickle(self.ranges, name, "fold-" + str(self.fold) + "_ranges.pkl")

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

    def set_literal_clusters(self):
        self.num_clusters = dict()
        self.text_clusters = dict()
        ps = set(self.train_store.predicates(subject=None, object=None))
        for p in ps:
            os = set(self.train_store.objects(subject=None, predicate=p))
            os = {o for o in os if type(o) is Literal}

            num_emb = [self.num_emb[self.num2id[self.entity2id[o]]] for o in os if self.entity2id[o] in self.num2id]
            text_emb = [self.text_emb[self.text2id[self.entity2id[o]]] for o in os if self.entity2id[o] in self.text2id]

            num_emb = np.array(num_emb)
            text_emb = np.array(text_emb)

            if len(num_emb) > 0:
                num_centroid = np.mean(num_emb, axis=0)
                num_radius = max({cosine(num_centroid, emb) for emb in num_emb})
                self.num_clusters[p] = num_centroid, num_radius

            if len(text_emb) > 0:
                text_centroid = np.mean(text_emb, axis=0)
                text_radius = max({cosine(text_centroid, emb) for emb in text_emb})
                self.text_clusters[p] = text_centroid, text_radius

        print("numerical literal clusters:", self.num_clusters)
        print("textual literal clusters:", self.text_clusters)

    def set_clusters(self):
        clusters = dict()
        for entity in self.entity2id:
            clusters[entity] = (set(self.train_store.predicates(subject=entity, object=None)),
                                set(self.train_store.predicates(subject=None, object=entity)))

        self.clusters = clusters
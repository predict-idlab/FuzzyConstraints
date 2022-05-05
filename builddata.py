import os
from pickling import *
from boltons.setutils import IndexedSet
from collections import defaultdict


class Utils:

    @staticmethod
    def get_id_mappings(data=None, name=None):
        vertices = IndexedSet()
        predicates = IndexedSet()
        entity_frequencies = defaultdict(float)
        relation_frequences = defaultdict(float)
        for triple in data:
            vertices.add(triple[0])
            predicates.add(triple[1])
            vertices.add(triple[2])
            entity_frequencies[triple[0]] += 1
            relation_frequences[triple[1]] += 1
            entity_frequencies[triple[2]] += 1

        if name is not None and \
                (not os.path.exists(os.path.join("pickles", "pickle_" + name, "entity_frequencies.pkl")) or
                 not os.path.exists(os.path.join("pickles", "pickle_" + name, "relation_frequencies.pkl"))):
            dump_to_pickle(entity_frequencies, name, "entity_frequencies.pkl")
            dump_to_pickle(relation_frequences, name, "relation_frequencies.pkl")
        print("Getting id mappings for", len(vertices), "vertices and", len(predicates), "predicates, from", len(data),
              "triples.")

        entity2id = {}
        id2entity = {}
        for i, vertex in enumerate(vertices):
            entity2id[vertex] = i
            id2entity[i] = vertex

        relation2id = {}
        id2relation = {}
        for i, predicate in enumerate(predicates):
            relation2id[predicate] = i
            id2relation[i] = predicate

        return entity2id, id2entity, relation2id, id2relation

    @staticmethod
    def build_data(train_data=None, test_data=None, valid_data=None, name=None):
        num_train = len(train_data)
        num_valid = 0
        triples = list(train_data)
        if valid_data is not None:
            num_valid = len(valid_data)
            triples.extend(valid_data)
        if test_data is not None:
            triples.extend(test_data)

        print("Building data for", len(triples), "triples.")

        entity2id, id2entity, relation2id, id2relation = Utils.get_id_mappings(triples, name)

        word2index = {}
        left_entity = {}
        right_entity = {}

        next_ent = 0
        entities = set()
        train = dict()
        valid = dict()
        test = dict()
        vertices = IndexedSet()
        predicates = IndexedSet()
        for i, triple in enumerate(triples):
            head, tail, rel, val = triple[0], triple[2], triple[1], [1]
            # if the subject was already encountered
            # then use the already recorded index
            if head in entities:
                sub_ind = word2index[head]
            else:
                # otherwise, set a new index
                sub_ind = next_ent
                next_ent += 1
                word2index[head] = sub_ind
                # and add the subject to the known entities
                entities.add(head)

            # same for relations
            if rel in entities:
                rel_ind = word2index[rel]
            else:
                rel_ind = next_ent
                next_ent += 1
                word2index[rel] = rel_ind
                entities.add(rel)

            # same for objects
            if tail in entities:
                obj_ind = word2index[tail]
            else:
                obj_ind = next_ent
                next_ent += 1
                word2index[tail] = obj_ind
                entities.add(tail)

            # for testing
            vertices.add(head)
            predicates.add(rel)
            vertices.add(tail)

            if i < num_train:
                train[(sub_ind, rel_ind, obj_ind)] = val
            elif i < num_train + num_valid:
                valid[(sub_ind, rel_ind, obj_ind)] = val
            else:
                test[(sub_ind, rel_ind, obj_ind)] = val

            # count the number of occurrences for each (head, rel)
            # for each relation id, create a dict for head-rel occurrences
            if relation2id[rel] not in left_entity:
                left_entity[relation2id[rel]] = {}
            # for each head, initiate a count for how many times it co-appears with the rel
            if entity2id[head] not in left_entity[relation2id[rel]]:
                left_entity[relation2id[rel]][entity2id[head]] = 0
            left_entity[relation2id[rel]][entity2id[head]] += 1
            # count the number of occurrences for each (rel, tail)
            # for each relation id, create a dict for rel-tail occurrences
            if relation2id[rel] not in right_entity:
                right_entity[relation2id[rel]] = {}
            # for each tail, initiate a count for how many times it co-appears with the rel
            if entity2id[tail] not in right_entity[relation2id[rel]]:
                right_entity[relation2id[rel]][entity2id[tail]] = 0
            right_entity[relation2id[rel]][entity2id[tail]] += 1

        # also construct the inverse dict
        index2word = {}
        for tmp_key in word2index:
            index2word[word2index[tmp_key]] = tmp_key

        left_avg = {}
        for i in range(len(relation2id)):
            left_avg[i] = sum(left_entity[i].values()) * 1.0 / len(left_entity[i])

        right_avg = {}
        for i in range(len(relation2id)):
            right_avg[i] = sum(right_entity[i].values()) * 1.0 / len(right_entity[i])

        head_tail_selector = {}
        for i in range(len(relation2id)):
            head_tail_selector[i] = 1000 * right_avg[i] / (right_avg[i] + left_avg[i])

        return train, test, valid, word2index, index2word, head_tail_selector, entity2id, id2entity, relation2id, id2relation
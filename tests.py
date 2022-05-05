import sys
from collections import defaultdict

import rdflib
from boltons.setutils import IndexedSet
from owlrl import DeductiveClosure, CombinedClosure
from rdflib import URIRef, Literal
from pickling import *


def test_benchmark(mode="TransE", trainer="reference", dataset_name="AIFB", fold=0,
                   attributive=False, sensory=False, strict=False, neg_ratio=1.0):

    print("\n\n------------------------------------------------------------------")
    print("dataset:", os.path.join("training", dataset_name))

    if not os.path.isfile(os.path.join("pickles", "pickle_" + dataset_name, "fold-" + str(fold) + "_train_store.pkl")):
        train_store = rdflib.Graph()
        valid_store = rdflib.Graph()
        test_store = rdflib.Graph()
        type_store = rdflib.Graph()

        train_store.parse(os.path.join("training", dataset_name, "fold-" + str(fold), "train.nt"), format="nt")
        valid_store.parse(os.path.join("training", dataset_name, "fold-" + str(fold), "valid.nt"), format="nt")
        test_store.parse(os.path.join("training", dataset_name, "fold-" + str(fold), "test.nt"), format="nt")
        type_store.parse(os.path.join("training", dataset_name, "types.nt"), format="nt")

        ontology = rdflib.Graph()
        rdf_ontology = rdflib.Graph()
        rdfs_ontology = rdflib.Graph()
        owl_ontology = rdflib.Graph()

        if os.path.exists(os.path.join("training", dataset_name, "ontology.ttl")):
            ontology.parse(os.path.join("training", dataset_name, "ontology.ttl"), format="ttl")
        rdf_ontology.parse(os.path.join("ontologies", "22-rdf-syntax-ns.ttl"), format="ttl")
        rdfs_ontology.parse(os.path.join("ontologies", "rdf-schema.ttl"), format="ttl")
        owl_ontology.parse(os.path.join("ontologies", "owl.ttl"), format="ttl")

        ontology += rdf_ontology + rdfs_ontology + owl_ontology

        reasoner = DeductiveClosure(CombinedClosure.RDFS_OWLRL_Semantics)
        reasoner.expand(ontology)

        # add ontology to expand types properly
        type_store += ontology
        reasoner.expand(type_store)
        for tr in list(type_store.triples((None, None, None))):
            if tr[1] != URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"):
                type_store.remove(tr)

        print("nr of train triples:", len(list(train_store.triples((None, None, None)))))
        print("nr of valid triples:", len(list(valid_store.triples((None, None, None)))))
        print("nr of test triples:", len(list(test_store.triples((None, None, None)))))
        print("nr of ontology triples:", len(list(ontology.triples((None, None, None)))))

        expansion = rdflib.Graph()
        expansion += train_store
        expansion += ontology

        # expand training store, using ontology to get even more type information
        # this is type information that we get from entities participating in relations
        reasoner = DeductiveClosure(CombinedClosure.RDFS_OWLRL_Semantics)
        reasoner.expand(expansion)
        expansion -= train_store
        expansion -= ontology

        # add new type declarations, after expansion of training store
        # these type declarations
        for triple in expansion.triples((None, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), None)):
            type_store.add(triple)

        train_triples = list(train_store.triples((None, None, None)))
        valid_triples = list(valid_store.triples((None, None, None)))
        test_triples = list(test_store.triples((None, None, None)))
        type_triples = list(set(type_store.triples((None, None, None))))

        print("nr of type triples:", len(type_triples))

        types_per_entity = defaultdict(set)
        entities_per_type = defaultdict(set)
        for triple in type_triples:
            # we are allowed to suggest entities not strictly
            # contained within the training set; no need to filter...
            types_per_entity[triple[0]].add(triple[2])
            entities_per_type[triple[2]].add(triple[0])

        def get_data(src_triples):
            vertices = IndexedSet()
            predicates = IndexedSet()
            for triple in src_triples:
                vertices.add(triple[0])
                predicates.add(triple[1])
                vertices.add(triple[2])
                # in case we do not have type information for literals...
                if isinstance(triple[0], Literal):
                    types_per_entity[triple[0]] = types_per_entity[triple[0]].union({triple[0].datatype,
                                                        rdflib.term.URIRef('http://www.w3.org/2000/01/rdf-schema#Resource'),
                                                        rdflib.term.URIRef('http://www.w3.org/2002/07/owl#Thing'),
                                                        rdflib.term.URIRef('http://www.w3.org/2000/01/rdf-schema#Literal')})
                    entities_per_type[triple[0].datatype].add(triple[0])
                    entities_per_type[rdflib.term.URIRef('http://www.w3.org/2000/01/rdf-schema#Resource')].add(triple[0])
                    entities_per_type[rdflib.term.URIRef('http://www.w3.org/2002/07/owl#Thing')].add(triple[0])
                    entities_per_type[rdflib.term.URIRef('http://www.w3.org/2000/01/rdf-schema#Literal')].add(triple[0])

                if isinstance(triple[2], Literal):
                    types_per_entity[triple[2]] = types_per_entity[triple[2]].union({triple[2].datatype,
                                                        rdflib.term.URIRef('http://www.w3.org/2000/01/rdf-schema#Resource'),
                                                        rdflib.term.URIRef('http://www.w3.org/2002/07/owl#Thing'),
                                                        rdflib.term.URIRef('http://www.w3.org/2000/01/rdf-schema#Literal')})
                    entities_per_type[triple[2].datatype].add(triple[2])
                    entities_per_type[rdflib.term.URIRef('http://www.w3.org/2000/01/rdf-schema#Resource')].add(triple[2])
                    entities_per_type[rdflib.term.URIRef('http://www.w3.org/2002/07/owl#Thing')].add(triple[2])
                    entities_per_type[rdflib.term.URIRef('http://www.w3.org/2000/01/rdf-schema#Literal')].add(triple[2])
            return vertices, predicates

        print("before...")
        for en in entities_per_type[Literal]:
            print(en, en.datatype, types_per_entity[en])

        train_vertices, train_predicates = get_data(train_triples)
        valid_vertices, valid_predicates = get_data(valid_triples)
        test_vertices, test_predicates = get_data(test_triples)

        print("after...")
        for en in entities_per_type[Literal]:
            print(en, en.datatype, types_per_entity[en])

        print("nr of entities for which type information was found:", len(types_per_entity.keys()))
        print("nr of types for which entities were found:", len(entities_per_type.keys()))

        print("train set size:", len(train_store))

        dump_to_pickle(train_store, dataset_name, "fold-" + str(fold) + "_train_store.pkl")
        dump_to_pickle(ontology, dataset_name, "fold-" + str(fold) + "_ontology.pkl")
        dump_to_pickle(types_per_entity, dataset_name, "fold-" + str(fold) + "_types_per_entity.pkl")
        dump_to_pickle(entities_per_type, dataset_name, "fold-" + str(fold) + "_entities_per_type.pkl")
        dump_to_pickle(train_triples, dataset_name, "fold-" + str(fold) + "_train_triples.pkl")
        dump_to_pickle(valid_triples, dataset_name, "fold-" + str(fold) + "_valid_triples.pkl")
        dump_to_pickle(test_triples, dataset_name, "fold-" + str(fold) + "_test_triples.pkl")
    else:
        train_store = load_from_pickle(dataset_name, "fold-" + str(fold) + "_train_store.pkl")
        ontology = load_from_pickle(dataset_name, "fold-" + str(fold) + "_ontology.pkl")
        types_per_entity = load_from_pickle(dataset_name, "fold-" + str(fold) + "_types_per_entity.pkl")
        entities_per_type = load_from_pickle(dataset_name, "fold-" + str(fold) + "_entities_per_type.pkl")
        train_triples = load_from_pickle(dataset_name, "fold-" + str(fold) + "_train_triples.pkl")
        valid_triples = load_from_pickle(dataset_name, "fold-" + str(fold) + "_valid_triples.pkl")
        test_triples = load_from_pickle(dataset_name, "fold-" + str(fold) + "_test_triples.pkl")

    def get_statistics():
        all_triples = []
        all_triples.extend(train_triples)
        all_triples.extend(valid_triples)
        all_triples.extend(test_triples)
        print("total triples:", len(all_triples))
        in_map, out_map = defaultdict(int), defaultdict(int)
        stat_graph = rdflib.Graph()
        for tr in all_triples:
            stat_graph.add(tr)

        for p in set(stat_graph.predicates(subject=None, object=None)):
            for s in set(stat_graph.subjects(predicate=p, object=None)):
                in_map[p] += 1
            for o in set(stat_graph.objects(predicate=p, subject=None)):
                out_map[p] += 1

        inavg, inmin, inmax = sum(in_map.values()) / len(in_map.values()), min(in_map.values()), max(in_map.values())
        outavg, outmin, outmax = sum(out_map.values()) / len(out_map.values()), min(in_map.values()), max(in_map.values())
        print("in rels:", inavg, inmin, inmax)
        print("out rels:", outavg, outmin, outmax)

        literal_graph = rdflib.Graph()
        num_graph = rdflib.Graph()
        if os.stat(os.path.join("training", dataset_name, "literals", "numerical_literals.nt")).st_size != 0:
            num_graph.parse(os.path.join("training", dataset_name, "literals", "numerical_literals.nt"), format="nt")
        txt_graph = rdflib.Graph()
        if os.stat(os.path.join("training", dataset_name, "literals", "text_literals.nt")).st_size != 0:
            txt_graph.parse(os.path.join("training", dataset_name, "literals", "text_literals.nt"), format="nt")
        print("num:", len(num_graph))
        print("text:", len(txt_graph))
        literal_graph += num_graph + txt_graph
        in_map, out_map = defaultdict(int), defaultdict(int)
        for p in set(literal_graph.predicates(subject=None, object=None)):
            for s in set(literal_graph.subjects(predicate=p, object=None)):
                in_map[p] += 1
            for o in set(literal_graph.objects(predicate=p, subject=None)):
                out_map[p] += 1

        inavg, inmin, inmax = sum(in_map.values()) / len(in_map.values()), min(in_map.values()), max(in_map.values())
        outavg, outmin, outmax = sum(out_map.values()) / len(out_map.values()), min(out_map.values()), max(
            out_map.values())
        print("in rel-vals:", inavg, inmin, inmax)
        print("out rel-vals:", outavg, outmin, outmax)

        tavg, tmin, tmax = 0.0, sys.maxsize, 0.0
        max_ent = None
        for e in types_per_entity:
            tmin = min(len(types_per_entity[e]), tmin)
            if tmax < len(types_per_entity[e]):
                max_ent = e
            tmax = max(len(types_per_entity[e]), tmax)
            tavg += len(types_per_entity[e])
        tavg /= len(types_per_entity.keys())
        print("types:", tavg, tmin, tmax)
        print(max_ent, types_per_entity[max_ent])

    get_statistics()

    if trainer == "reference":
        from sampling.reference.trainer import Trainer
    elif trainer == "strict":
        from sampling.strict.trainer import Trainer
    elif trainer == "lcwa":
        from sampling.lcwa.trainer import Trainer
    elif trainer == "clustering":
        from sampling.clustering.trainer import Trainer
    elif trainer == "standard":
        from sampling.standard.trainer import Trainer
    else:
        from sampling.hybrid.trainer import Trainer

    trainer = Trainer(dataset_name=dataset_name, fold=fold,
                      types_per_entity=types_per_entity,
                      entities_per_type=entities_per_type,
                      ontology=ontology, train_store=train_store,
                      attributive=attributive, sensory=sensory,
                      strict=strict, neg_ratio=neg_ratio)
    trainer.train_and_eval(mode, train_triples, test_triples, valid_triples)


if __name__ == '__main__':
    assert len(sys.argv) >= 9
    mode = sys.argv[1]
    trainer = sys.argv[2]
    dataset = sys.argv[3]
    fold = int(sys.argv[4])
    attributive = bool(int(sys.argv[5]))
    sensory = bool(int(sys.argv[6]))
    strict = bool(int(sys.argv[7]))
    neg_ratio = float(sys.argv[8])
    sys.argv = [sys.argv[0]]

    print("params:", mode, trainer, dataset, fold, attributive, sensory, strict, neg_ratio)
    test_benchmark(mode=mode, trainer=trainer, dataset_name=dataset, fold=fold,
                   attributive=attributive, sensory=sensory, strict=strict, neg_ratio=neg_ratio)

Fuzzy Constraints for Negative Sampling
==============================

Link prediction experiments using knowledge graph embeddings powered by fuzzy constraints. Parts of the implementation are based on https://github.com/thunlp/OpenKE.


Running
-------

1. Clone this project: `git clone...`.
2. Preferably create a virtual environment (`conda create --name fuzzyconstraints`) and activate it (`conda activate fuzzyconstraints`).
3. `cd` to the project's root folder and install all required packages: `pip install -r requirements.txt`.
4. Run `python tests.py` with default parameters `TransE reference AIFB 0 0 0 0 1` to run a basic test.

Project Organization
------------

    |-- BaseModule.py
    |-- LICENSE             <-- ...
    |-- Multimodal.py       <-- Functionality to create multi-modal (i.e., literal-based) embeddings.
    |-- README.md           
    |-- __init__.py
    |-- builddata.py        <-- Utilities to prepapre data for training and testing.  
    |-- gating.py           <-- Gating used to create multi-modal embeddings.
    |-- loss                <-- Loss functions used to train embeddings.
    |   |-- Loss.py
    |   |-- MarginLoss.py
    |   |-- SigmoidLoss.py
    |   |-- SoftplusLoss.py
    |-- models              <-- Embedding models.
    |   |-- DistMult.py
    |   |-- Model.py
    |   |-- TransE.py
    |   `-- __init__.py
    |-- ontologies          <-- Top-Level modelling ontologies (RDF, RDFS, OWL)
    |   |-- 22-rdf-syntax-ns.ttl
    |   |-- owl.ttl
    |   `-- rdf-schema.ttl
    |-- pickles             <-- Destination for pickle files.
    |   |-- pickle_AIFB
    |   `-- pickle_MUTAG
    |-- pickling.py         <-- Pickling functionality.
    |-- requirements.txt       
    |-- results             <-- Destination for evaluation results.
    |   |-- clustering
    |   |-- hybrid
    |   |-- lcwa
    |   |-- reference
    |   |-- standard
    |   `-- strict
    |-- sampling            <-- Various sampling strategies.
    |   |-- clustering
    |   |   |-- batching.py
    |   |   `-- trainer.py
    |   |-- generic_batching.py
    |   |-- generic_trainer.py
    |   |-- hybrid          <-- Hybrid fuzzy sampling.
    |   |   |-- batching.py
    |   |   `-- trainer.py
    |   |-- lcwa
    |   |   |-- batching.py
    |   |   `-- trainer.py
    |   |-- reference
    |   |   |-- batching.py
    |   |   `-- trainer.py
    |   |-- standard        <-- Standard fuzzy sampling.
    |   |   |-- batching.py
    |   |   `-- trainer.py
    |   `-- strict
    |       |-- batching.py
    |       `-- trainer.py
    |-- tests.py            <-- Top-Level functionality: execute test for a given set of parameters.
    `-- training            <-- Data repository for training and testing.
        |-- AIFB
        |   |-- fold-0
        |   |   |-- test.nt
        |   |   |-- train.nt
        |   |   `-- valid.nt
        |   |-- literals
        |   |   |-- numerical_literals.nt
        |   |   `-- text_literals.nt
        |   |-- ontology.ttl
        |   `-- types.nt
        `-- MUTAG
            |-- fold-0
            |   |-- test.nt
            |   |-- train.nt
            |   `-- valid.nt
            |-- literals
            |   |-- numerical_literals.nt
            |   `-- text_literals.nt
            |-- ontology.ttl
            `-- types.nt

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

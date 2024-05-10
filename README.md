
**Experiments for the Paper**
> E. Werner, N. Kumar, M. Lieber, S. Torge, S. Gumhold, W.E. Nagel. *Towards Computational Performance Engineering for Unsupervised Concept Drift Detection - Complexities, Benchmarking, Performance Analysis*. Proceedings of the 13th International Conference on Data Science, Technology and Applications (DATA).

**Structure:**

- data - data for the experiments
- experiments
    - baselines
    - benchmark
    - performanceAnalysis
- uDD
    - iks - additional code for iks drift detection
    - studd - drift detection, model definition, and baselines
- requirements.txt - python requirements
- README.md - this file

Experiment setups for baselines and benchmarks consist of Python files for running the experiment. All experiments additionally consist of a `submit.sbatch` file for running on the ZIH HPC machine. Note, for performance analysis we utilize the same python files as for benchmark.

`uDD` directory consists of an adapted version of studd[1] to provide only necessary functionalities. We use studd's infrastructure for our workflows and the studd drift detection for the benchmark. `iks` directory provides the iks drift detection functionalities as provided by[2].

**Installation:**

*install all required packages*
```pip install -r requirements.txt```

*install scorep (for performance analysis)*
```pip install -t scorep scorep```

*install scorep-dummy (for benchmark but keeping the same files)*
```pip install -t scorep-dummy scorep-dummy```

Note: either add the path to scorep-dummy (benchmark) or scorep (performance analysis) to PYTHONPATH depending on the experiment to run.

**Sources:**

[1] https://github.com/vcerqueira/studd
> Cerqueira, Vitor, et al. "STUDD: A student–teacher method for unsupervised concept drift detection." Machine Learning 112.11 (2023): 4351-4378.

[2] https://github.com/denismr/incremental-ks/tree/master
> Dos Reis, Denis Moreira, et al. "Fast unsupervised online drift detection using incremental kolmogorov-smirnov test." Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining. 2016.

**ACK:**
The authors gratefully acknowledge the computing time made available to them on the high-performance computer at the NHR Center of TU Dresden. This center is jointly supported by the Federal Ministry of Education and Research and the state governments participating in the NHR (www.nhr-verein.de/unsere-partner).

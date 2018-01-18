## dynsbm-gpu
Multithread implementation of statistical clustering algorithm proposed by Matias and Miele in paper
Statistical clustering of temporal networks through a dynamic stochastic block model.

Algorithm written as a part of my master thesis. It can run even 71 times faster on GPU than on single CPU thread.


## Installation
### Requirements
- python >= 3.5
- TensorFlow GPU >= 1.4 (nightly version is recommended due to
[tf.diag support on GPU](https://github.com/tensorflow/tensorflow/pull/13666))
- CUDA related libraries (follow TensorFlow installation guide)


## Usage
### Run methods
To run single test `main.py` or `main_run_set.py` file can be used:
- `python3 main.py`,
- `python3 main_run_set.py data_set use_gpu init_method iteration` (use_gpu = true/false, init_method = k-means/random)

There is also `run_all_data_sets.sh` file which run algorithm on all
files in `DataSet` folder. On top of this script there are few variables
which are used to filter out some of the data sets.
- `./run_all_data_sets.sh > results.csv`.

### Results
Algorithm results are printed into standard output. `main.py` contains descriptive (but limited) output.
`main_run_set.py` on the other hand print results in CSV-like fashion (data is pipe separated '|').

## Data sets
In this repository there is 42 data sets. Each file is named `T_N_Q_K_distribution.csv`,
where **T** is number of time steps, **N** number of vertices, **Q** number of clusters,
**K** maximum weight and **distribution** is name of distribution used to generate
weights in graph. Algorithm itself do not require this naming convention to work,
but `run_all_data_sets.sh` use it to filter out some of the data sets.


## References
- [Matias, C., & Miele, V. (2017). Statistical clustering of temporal networks through a dynamic stochastic block model. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 79(4), 1119-1141.
ISO 690](https://arxiv.org/abs/1506.07464)

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
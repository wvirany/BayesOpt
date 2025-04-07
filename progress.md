* Do we want to continuously update the mean function for `ConstantMeanTanimotoGP` as we observe new data in BO?


# April

### To Do:

- [x] Pre-compute all fingerprints for the pool at initialization
- [ ] Run all regression experiments -> For which targets are the performance differences largest?



# March

### To Do:

#### Misc:
- [x] Profiling code
- [x] Cache Cholesky factorization of kernel matrix, implement efficient update function
- [x] Cache K_test_train
- [x] Correctly and efficiently compute incumbent
- [ ] Re-run regression experiments with `ConstantMeanTanimotoGP` [params: (`PARP1`, `F2`, `ESR2`), (compressed, uncompressed), (r2, r4), (binary, count)] --> record results and params
- [x] Optimize SLURM resource usage (smaller jobs, SLURM arrays w/ different seeds dictating different params / data inits)
- [x] Record SMILES strings at each BO iter
- [x] Test `update_choleky()` method in `test_kern_gp.py`
- [x] Implement `ConstantMeanTanimotoGP`
- [x] `FixedTanimotoGP` class? Reconcile `ConstantMeanTanimotoGP` and K_test_train caching
- [ ] Generalize `add_observation()` to $n$ observations
- [ ] Regression experiments
- [x] Test and fix caching functionality

#### BO experiments
- [x] Run all BO experiments for 1000 initial molecules and budget w/ GP regression params on bottom 80%
  - [x] `PARP1`
  - [x] `F2`
  - [x] `ESR2`
- [x] After making code faster and fixing SLURM resource usage, run more trials w/ larger initalizations / budgets
- [x] Re-run BO experiments with `FixedTanimotoGP` class
- [x] BO experiments w/ `n_init` 100 and `budget` 1000
- [x] BO experiments w/ `radius` 4
- [x] Scale up pool size

# Feb

### To Do:

#### Experiments:

- [x] EI in noisy case (posterior mean implementation)
- [x] BO plots: show median, min, max (or 25th / 75th percentile) instead of mean, std
- [x] Dockstring:
  - [x] Regression benchmark to see initial performance (PARP1, F2)
  - [x] Dockstring BO experiment (initialize on 100/1000 molecules, run BO w/ budget of 100/1000)
    - [x] Evaluate radiuses 2 and 4 on targets PARP1 and F2
    - [x] Compare each to corresponding compressed fingerprints
    - [x] Use regression GP parameters for corresponding BO experiment
    - [x] Try running on bottom 80% of dataset

#### Other: 

- ~~[ ] Configurable fingerprint and BO parameters from command line~~
- [x] Script for making Dockstring BO plots

### Updates:

* GP param tests helped iron out some issues

* Reran fingerprint comparison on new dataset with exact target values

  $\rightarrow$ more expected results

* BO experiments
  * Ran EI, varied tradeoff parameter in $\{0.01, 0.1, 1.0\}$


### To Do:

- [x] Evaluate on polaris competition intermediate benchmark

- ~~[ ] Run GP param diagnosis with `fcfp` fingerprint to compare~~

- [x] BO Experiments:

  - [x] Run BO experiments $\geq$ 3 times with different random initializations, plot mean and +- std error bars
  - [x] Make histogram of y values, plot vertical lines on histogram and horizontal lines on logS vs. iteration number
  - [x] Vary UCB `beta` $\in$ `{0.01, 0.1, 1.0}`
  - [x] Implement EI
  - ~~[ ] Run BO on harder tasks (Dockstring, PMO, etc.)~~


* ~~[ ] Make script to submit to Polaris benchmarks / competitions (e.g., specify benchmark / endpoint from command line, find best result)~~

- ~~[ ] Make parameters configurable from command line (fptype, sizes, ~~radius~~, ~~tol~~, etc.)~~


# January

### Updates:

* Plotted GP params, performed GP diagnostics, fixed small issues

* Ran more BO experiments $\rightarrow$ need to rerun these after having fixed GP issues

* Ran preliminary BO experiment

* Tested GP with adjusted mean
  * Computed mean of training data
  * Subtracted mean from training data, yielding "centered" data
  * Trained on centered data
  * Predicted mean, var of test data, then added mean of training data back to predicted mean

$\rightarrow$ Slightly improved results

* Code changes:
  * Modifications to `tanimoto_gp.py`: included class option for nonzero-mean GP
  * Created `utils.py`


### To Do:

- [x] Clone repos instead of copy

- [x] Implement tolerance for GP param optimization $\rightarrow$ re-run fingerprint comparison

- [x] Keep track of GP params for experiments (e.g., a table)
  * Make sure to undo softmax

- [x] Modify BO experiment:
  * Start w/ less points, different split, $n$ worst molecules, etc.
  * Fit parameters on larger subset of data, then run BO on complement, e.g.,
    * Take 1000/2000 molecules, maximize MLL
    * Using parameters, initialize GP and run BO experiment on remaining data,
      starting with a small subset of molecules (e.g., 100-200 of remaining 1000)

- ~~[ ] Run BO on harder tasks (Dockstring, PMO, etc.)~~
  * Hopefully will observe increasing difference in performance between model
    exact fingerprint vs. limited fingerprint w.r.t. iterations


## December

### Updates:

* Tested GP on 4 different fingerprint types:
  * Extended connectivity
  * Functional Connectivity
  * Topological
  * Atompair
* Evaluated on Polaris solubility dataset


### Next steps:

- [x] Currently using ZeroMeanGP $\rightarrow$ Instead, calculate mean of training set and add to mean of GP

- [x] Visualize test log-likelihood
  
- [x] Preliminary BO experiment:
  * Use offline dataset (e.g., Polaris logP training dataset)
  * Procedure:
     1. Pick some fraction of dataset (~20%)
     2. Train GP on this fraction
     3. Define an acquisition function (e.g., UCB, EI, etc.)
     4. At each iteration, make predictions for _all_ unlabeled points
     5. Choose molecule with highest acquisition value $\rightarrow$ Evaluate logP, add to training data
     6. Repeat
  * Compare to random baseline (i.e., compare to selecting molecule based on uniform distribution)
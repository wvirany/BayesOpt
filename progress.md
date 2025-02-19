
### Thoughts:

* Should we be using consistent GP params for fingerprint comparison? --> Maybe. First, let's see what they are
* UMAP / t-SNE projections of chemical space: have unobserved in gray, observed in another color
* Try BO exp for many iters? (e.g., 100+)
* ~~Try EI acquisition function~~
* Compare preliminary BO experiments to compressed fingerprints
* ~~Try BO experiments with larger tolerance~~
* Should we try predicting different endpoints besides logS?
* Is `gp.set_training_data()` the right method to update GP posterior?


## Feb. 9


## TO DO:

- [x] EI in noisy case (posterior mean implementation)
- [x] BO plots: show median, min, max (or 25th / 75th percentile) instead of mean, std
- [ ] Dockstring:
  - [x] Regression benchmark to see initial performance (PARP1, F2)
  - [ ] Dockstring BO experiment (initialize on 1000 molecules, run BO w/ budget of 1000)

## February

### Updates:

* GP param tests helped iron out some issues

* Reran fingerprint comparison on new dataset with exact target values

  $\rightarrow$ more expected results

* BO experiments
  * Ran EI, varied tradeoff parameter in $\{0.01, 0.1, 1.0\}$


### To Do:

- [x] Evaluate on polaris competition intermediate benchmark

- ~~[ ] Run GP param diagnosis with `fcfp` fingerprint to compare~~

- [ ] BO Experiments:

  - [x] Run BO experiments $\geq$ 3 times with different random initializations, plot mean and +- std error bars
  - [x] Make histogram of y values, plot vertical lines on histogram and horizontal lines on logS vs. iteration number
  - [x] Vary UCB `beta` $\in$ `{0.01, 0.1, 1.0}`
  - [x] Implement EI
  - [ ] Run BO on harder tasks (Dockstring, PMO, etc.)


* ~~[ ] Make script to submit to Polaris benchmarks / competitions (e.g., specify benchmark / endpoint from command line, find best result)~~

- [ ] Make parameters configurable from command line (fptype, sizes, ~~radius~~, ~~tol~~, etc.)


## January

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
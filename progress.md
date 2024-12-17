# Dec. 17

### Updates:

* Tested GP on 4 different fingerprint types:
  * Extended connectivity
  * Functional Connectivity
  * Topological
  * Atompair
* Evaluated on Polaris solubility dataset:
 <p align="center">
 <img src="figures/fingerprint_comparison.png" alt="fingerprint_comparison.png" width="70%"/>
 </p>

### Next steps:

* Currently using ZeroMeanGP $\rightarrow$ Instead, calculate mean of training set and add to mean of GP
* Visualize test log likelihood

* Try BO experiments:
  * Use offline dataset (e.g., Polaris logP training dataset)
  * Procedure:
     1. Pick some fraction of dataset (~20%)
     2. Train GP on this fraction
     3. Define an acquisition function (e.g., UCB, EI, etc.)
     4. At each iteration, make predictions for _all_ unlabeled points
     5. Choose molecule with highest acquisition value $\rightarrow$ Evaluate logP, add to training data
     6. Repeat
  * compare to random baseline (i.e., compare to selecting molecule based on uniform distribution)

* Stretch goal: try some experiments w/ less trivial dataset (e.g., DockString, PMO, etc.)

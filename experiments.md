# Experiments

## Bayes Opt comparison


### BO experiment #1:
 * Used UCB acquisition function, compared to uniform sampling
 * Plotted value of best sample in observations set at each iteration:
<p align="center">
<img src="figures/bayes_opt/bayes_opt_1.png" alt="bayes_opt_1.png" width="40%"/>
</p>

 * Made the task more difficult: started with bottom 10% of library, compared UCB to uniform:

<p align="center">
<img src="figures/bayes_opt/bo_exp1.png" alt="bayes_opt_1.png" width="40%"/>
</p>

To run:

```py
python3 bo_exp1.py --split_method 'n_worst' --split 0.1 --num_iters 30
```

## Fingerprint comparison

Here we compared the performance of TanimotoGP with different fingerprints, specifically looking at the the performance difference using exact fingerprints vs. fingerprints of a limited size.

Moreover, we varied the `radius` parameter (only applicable for `ecfp` and `fcfp` fingerprint types), with the hypothesis that a larger radius would lead to more hash collisions, and thus there would be an increased gap in performance as we increased the size of the fingerprints.


### Experiment #1, initial fingerprint comparison (parameters: `radius=2`):

* Complete fingerprint comparison, showing MSE, Pearson, and TLL:
<p align="center">
<img src="figures/fp_comparison/fingerprint_comparison_complete.png" alt="fingerprint_comparison_complete.png" width="100%"/>
</p>

As expected, we see an improvement in performance as the fingerprint size increases.


### Experiment #2 (parameters: `radius=4`):

<p align="center">
<img src="figures/fp_comparison/fingerprint_comparison_radius4.png" alt="fingerprint_comparison_radius4.png" width="100%"/>
</p>

We can see that the increased number of hash collisions decreases performance for limited-size fingerprints.


### Experiment #3 (parameters: `radius=2`, `tol=1e-3)`

* Implemented gradient norm tolerance criteria for optimization loop, got improved results with `tol=1e-3`:

<p align="center">
<img src="figures/fp_comparison/1e-3/fingerprint_comparison.png" alt="fingerprint_comparison_complete.png" width="100%"/>
</p>


Printed corresponding GP parameters:
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ecfp-512</th>
      <th>ecfp-1024</th>
      <th>ecfp-2048</th>
      <th>ecfp-sparse</th>
      <th>fcfp-512</th>
      <th>fcfp-1024</th>
      <th>fcfp-2048</th>
      <th>fcfp-sparse</th>
      <th>topological-512</th>
      <th>topological-1024</th>
      <th>topological-2048</th>
      <th>topological-sparse</th>
      <th>atompair-512</th>
      <th>atompair-1024</th>
      <th>atompair-2048</th>
      <th>atompair-sparse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Amplitude</th>
      <td>0.289229</td>
      <td>0.309973</td>
      <td>0.358875</td>
      <td>0.397465</td>
      <td>0.321568</td>
      <td>0.349986</td>
      <td>0.368100</td>
      <td>0.418882</td>
      <td>0.416156</td>
      <td>0.439664</td>
      <td>0.433452</td>
      <td>0.433704</td>
      <td>0.261918</td>
      <td>0.262317</td>
      <td>0.276256</td>
      <td>0.282044</td>
    </tr>
    <tr>
      <th>Noise</th>
      <td>0.156426</td>
      <td>0.137499</td>
      <td>0.101067</td>
      <td>0.074477</td>
      <td>0.174739</td>
      <td>0.156134</td>
      <td>0.143275</td>
      <td>0.116604</td>
      <td>0.121755</td>
      <td>0.094689</td>
      <td>0.088448</td>
      <td>0.081369</td>
      <td>0.190482</td>
      <td>0.187274</td>
      <td>0.177331</td>
      <td>0.170285</td>
    </tr>
  </tbody>
</table>
</div>


To run this experiment:

```py
python3 evaluate_fingerprints.py --generate_data --make_plots --savefig --filename '1e-3/fingerprint_comparison'
```
with parameters:
* `tol = 1e-3`
* `fps = ['ecfp', 'fcfp', 'topological', 'atompair']`
* `sizes = [512, 1024, 2048]`
* `radius = 2`


### Experiment #4 (parameters: `radius=4`, `tol=1e-3`):

Interestingly, we don't see the same trend as we saw in experiment #2. I wonder if this is due to overfitting, since the experiment is the same, we only adjusted `tol`.

<p align="center">
<img src="figures/fp_comparison/1e-3/fp_comparison_r4_1e-3.png" alt="fp_comparison_r4_1e-3.png" width="100%"/>
</p>

To run this experiment:

```py
python3 evaluate_fingerprints.py --generate_data --make_plots --savefig --filename '1e-3/fp_comparison_r4_1e-3'
```
with parameters:
* `tol = 1e-3`
* `fps = ['ecfp', 'fcfp']`
* `sizes = [512, 1024, 2048]`
* `radius = 4`
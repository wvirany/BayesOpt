# Benchmarks for caching

## Original benchmark (no caching):

Initial train / test:

```
100 / 9900
Total time: 130.56s
Average time per iteration: 4.35s
```

```
1000 / 9000
Total time: 1045.98s
Average time per iteration: 34.87s
```

## After caching Cholesky factorization:

```
100 / 9900
Total time: 127.07s
Average time per iteration: 4.24s
```

```
1000 / 9000
Total time: 858.84s
Average time per iteration: 28.63s
```


## After caching `K_test_train`:

```
100 / 9900
Total time: 32.86s
Average time per iteration: 1.10s
```

```
1000 / 9000
Total time: 62.90s
Average time per iteration: 2.10s
```


# Benchmarks for pool size

All of the following were ran with the parameters
* `n_init`: 100
* `target`: `PARP1`
* `budget`: 30
* `radius`: 2
* `sparse`: True

## Pool size: $50,000$

```
`n_init`: 100
Total time: 62.15s
Average time per iteration: 2.07s
``````

## Pool size: $100,000$

```
Total time: 102.96s
Average time per iteration: 3.43s
```

## Pool size: $105,000$

```
Total time: 920.25s
Average time per iteration: 30.68s
```

This should be ~linear.

After pre-computing all fingerprints:

```
Total time: 104.48s
Average time per iteration: 3.48s
```

Huge!

## Pool size: $150,000$

```
Total time: 143.08s
Average time per iteration: 4.77s
```

## Pool size: $250,000$

```
Total time: 196.79s
Average time per iteration: 6.56s
```
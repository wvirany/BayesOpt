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
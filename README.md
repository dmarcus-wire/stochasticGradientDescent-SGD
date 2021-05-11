# Stochastic Gradient Descent

[!image](./images/plots.png)

- work horse
- computes gradient + updates weights matrix on small batches of training data
- ...instead of the entire dataset (like vanilla gradient descent)
- ...which also runs slow and performs poorly on large datasets  
- leading to faster convergence without negative effects on loss/accuracy
- introduced ~60 years ago [link](https://www.google.com/books/edition/Adaptive_adaline_Neuron_Using_Chemical_m/Yc4EAAAAIAAJ?hl=en)

# Mini-batch SGD

- instead of computing gradient over entire dataset
- we yield a batch, evaluate and update W matrix
- purist will use 1 batch (or 1 datapoint from training set)
- mini-batches > 1 (32, 64, 128, 256)
- >1 because more stable convergence
- powers of 2 allow internal linear algebra libraries to be more efficient in power of 2
```
while True:
    batch = next_training_batch(data, 256)
    Wgradient = evaluate_gradient(loss, batch, W)
    W += -alpha * Wgradient
```



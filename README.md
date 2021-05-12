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
- 1 because more stable convergence
- powers of 2 allow internal linear algebra libraries to be more efficient in power of 2
```
while True:
    batch = next_training_batch(data, 256)
    Wgradient = evaluate_gradient(loss, batch, W)
    W += -alpha * Wgradient
```

- too small, takes too long to upate
- too large, fast, but to few updates

tune
- alpha
- epochs  
- batchSize

# Gradient Descent Analogy
- imagine mountains (this is our cost function)
- No GPS
- we have weight matrix (knows the direction of a step)
- weight matrix is like an accelerator
- it knows after we took a step if it was correct or not
- the weight matrix updates itself

## start
- initialize with random set of weights
- we make predicitons on a data point from training set
- we compute the Loss
- we tweak

## Loss/Optimization to drive loss as low as possible
- binary class: binary cross entropy
- multi-class: categorical cross entropy
- regression: mean squared/absolute error

## Variations of Gradient Descent

### Vanilla (think burning rubber at the start of a race)
>we only update the weights ONCE per iteration. The network sees the entire dataset everytime a weight update is performed.  
  - if you have *N* = 10,000 images
  - goal is to train NN to classify all 10,000 into *T* = 10 categories
  - run ALL images through network
  - compute Loss and gradient
  - update parameters of the network
  - if the training examples are large, descent is going to take a long time to converge
  - also, the larger the dataset, the more nuanced the gradients become (most time will be spent computing predictions)

    
### SGD
>instead of performing one weight update per epoch, performs multiple weight updates
- performs multiple weight updates
- N weight updates per epoch where N = datapoints
- 10.000 images, 10,000 weight updates per epoch
- Until convergence
    - randomly select single point from dataset
    - make prediction on it
    - compute loss and gradient
    - update parameters of the network
- this can also be very computationally wastefule

### Mini-batch SGD
>solves the time proble with vanilla gradient descent and compute waste of SGD
- Introduces the concept of batch-size, given dataset size N, there are N/S updates
- Randomly shuffle input data
- Until convergence
    - select next batch size of data size S
    - make predictions on subset
    - calculate the loss and mean gradient of the mini-batch
    - update parameters of the network
- purist form could be S=1 (datapoint)
- has trouble navigating areas of loss with significantly steeper in one dim than others

### **SGD w/momentum**
>solves the problem SGD bouncing around a local optimum
- problem with momentum is once you develop a head of steam, you can become out of control
- always use momentum

### **SGD w/Nesterov Acceleration**>
>"look ahead"
- sometimes it works and sometimes it doesn't
- treat as a hyperparameter "scientific method" (true or false boolean)
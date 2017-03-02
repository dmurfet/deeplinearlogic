# Linear Logic and Recurrent Neural Networks

This is the repository of the paper "Linear Logic and Recurrent Neural Networks" and the associated TensorFlow implementation. At the moment the repo is private, and this Readme will be used to flag some current TODOs in the code.

As of Friday 3rd of March, the NTM runs and on the Copy task (which asks the controller to simply repeat a given input sequence) gets the following (not very good) result:

```
N = 20, controller_state_size = 10, memory_address_size = 10, memory_content_size = 5, epoch = 100
Test error = 0.232
```

Some TODOs that come to mind:

* TensorBoard, including visualising memory state
* How blurred is the memory state?
* Initialisation of the weights
* Regularisation
* Coding the relevant test functions
* Different optimiser?
* What batch sizes, training size, controller state size, memory size to use?

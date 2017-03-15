# Linear Logic and Recurrent Neural Networks

This is the repository of the paper "Linear Logic and Recurrent Neural Networks" and the associated TensorFlow implementation. At the moment the repo is private. The models that have been implemented so far are

- The ordinary NTM (see class `NTM` in `ntm.py`).
- The pattern NTM (see class `PatternNTM` in `ntm.py`) which is the model described in Section 4.1 of the paper.
- The alternative pattern NTM (see class `PatternNTM_alt` in `ntm.py`) which is the pattern NTM but with the controller allowed to manipulate the read address of the first memory ring directly.

## News

See the [spreadsheet](https://docs.google.com/spreadsheets/d/1GqwW3ma7Cd1W8X8Txph9MPmLSkQ0C-i0tP0YHeINzMs/edit?usp=sharing) of the experiments I have run so far on the following tasks on sequences of length `N = 20`:

- Copy task (as in the NTM paper),
- Repeat copy task (as in the NTM paper),
- Pattern task (defined in Section 4.1 of our paper).

The numbers recorded in the spreadsheet are the percentages of correct predictions for the digits of the output binary sequence (`0.50` meaning as good as chance, `0` meaning perfect predictions) for the test set (which is three times the size of the training set, which is in turn 1% of the sequences, around 10k).

These experiments were done with a version of the code now denoted `v1`. However this version of the code did not implement sharpening (see Eq. (9) of the NTM paper) and had some initialisation choices that made it basically impossible for any of the models to really use the memory in the manner intended. It is somewhat remarkable that they managed to converge to zero error in so many cases, given these handicaps (in hindsight, binary sequences, even of length `20`, may not be very challenging given this many weights). However, the truly useless nature of the `v1` code is made clear by the fact that the models trained using it completely fail to generalise (here we test generalisation by training on length `20` sequences and testing on length `> 20` sequences).

** CURRENT STATUS ** In `v2` of the code, the implementation of which is ongoing as of the 13th of March, sharpening is implemented and the initialisations are fixed. So far this has been done for the NTM, but not the other models. We get convergence on the Copy task now (sometimes) with sharpening turned on, although generalisation is still bad. The best use of the memory we have seen so far is

```
Step 7 of the RNN run on the first input of first batch of this epoch
Read gamma - [ 1.]
Write gamma - [ 1.]
Write address -
[  1.38072655e-01   1.83058560e-01   1.40974551e-01   1.32708490e-01
   6.75634891e-02   4.50365767e-02   1.17327636e-02   5.49054937e-03
   2.56688448e-08   2.63412048e-08   2.62766502e-08   2.71339982e-08
   2.59571529e-08   6.51866547e-04   1.47098606e-03   9.15776752e-03
   1.61451790e-02   4.92556281e-02   6.78959638e-02   1.30784824e-01]
Argmax - 1


Step 8 of the RNN run on the first input of first batch of this epoch
Read gamma - [ 1.]
Write gamma - [ 1.]
Write address -
[  1.46947637e-01   1.43683389e-01   1.63783297e-01   1.18484326e-01
   1.00005426e-01   4.87421304e-02   2.98709143e-02   7.61802401e-03
   3.29333707e-03   2.59189044e-08   2.65678217e-08   2.62732165e-08
   1.91964544e-04   5.02075360e-04   3.24307731e-03   6.60458300e-03
   2.17035543e-02   3.48841548e-02   7.52331242e-02   9.52089354e-02]
Argmax - 2


Step 9 of the RNN run on the first input of first batch of this epoch
Read gamma - [ 1.7574985]
Write gamma - [ 1.65013874]
Write address -
[  1.04503557e-01   1.48825794e-01   1.41839489e-01   1.53002232e-01
   1.08669505e-01   8.80576819e-02   4.24477085e-02   2.51635946e-02
   6.39932044e-03   2.67596846e-03   2.60040558e-08   2.38047851e-05
   7.44253775e-05   5.89667354e-04   1.43234758e-03   5.74376620e-03
   1.10679362e-02   2.91730426e-02   4.49227355e-02   8.53874683e-02]
Argmax - 3
```

which advanced the write address through three consecutive locations. Unfortunately, it then immediately started producing `nan`s and huge error...
 
Also, when we implemented the ability to test on sequences of greater length than during training, we used a save and load of the weight matrices. This doesn't appear to work very well (even if training seems to be low error, testing is often very high error). So there are probably bugs there too.

## TODOs

The TODO list items by category:

- **Implement more models**
    - Multiple Pattern NTM
    - Polynomial step NTM
- **Implement more tasks**
    - Other tasks from NTM, DNC and other papers
    - Multiple pattern task (as in Section 4.2 of the paper)
    - Polynomial pattern task (as in Section 4.3 of the paper)
- **Inspection and visualisation**
    - Setting up Tensorboard
    - How to visualise the memory state?
    - Check in the copy and pattern copy examples that the "algorithm" being learned is comprehensible
    - How blurred is the memory state?
- **Details of training**
    - Regularisation (to e.g. force the memory to be used "properly")
	- Gradient clipping? This seems standard in the augmented RNN literature
	- Noise?
	- Curriculum learning?
- **Various**
    - Train on `N = 20` and test on `N > 20` length sequences. For sequences of length over `20` it is not feasible to construct the list of all sequences, shuffle it, and take some part, as we are currently doing. We need to rewrite the code to sample from the set of all sequences. This is also necessary if we want to increase `num_classes` (i.e. the number of symbols in our alphabet) and run on sequences of length `N = 20`
    - Run experiments on non-binary sequences

The next few things on my immediate TODO list:

- **TODO track 1**
    - Rewrite the code to sample from the set of sequences
    - Run all models on Copy, Repeat copy and Pattern tasks on `num_classes = 2` and `N =   25`
    - Run all models on Copy, Repeat copy and Pattern tasks on `num_classes = 3` and `N = 20`
    - Rewrite code to allow for training on `N = 20` and test on `N > 20`
    - Run all models on Copy, Repeat copy and Pattern tasks, training on `N = 20` and testing on `N = 30`, at `num_classes = 2` and `num_classes = 3`

- **TODO track 2**
    - Get TensorBoard visualisation working, analyse experiments
    - Gradient clipping?
    - Implement more models and harder tasks

## Some lessons learned

- We default to `controller_state_size = 100` in all our experiments now. In the beginning we tried `50` or even `30` but the models often failed to converge, and this is not allowing us to really distinguish the NTM and Pattern NTM. This is also the same dimension as the controller in the original NTM paper.

- On the Repeat Copy and Pattern tasks the convergence was unreliable for both the NTM and Pattern NTM using the Adam optimiser. However, things are much better with RMSProp. So the current situation is that on all tasks we have currently implemented, both the NTM, Pattern NTM and alternative Pattern NTM converge to zero error on some large percentage of runs (arguably the percentage is somewhat higher for the pattern NTM on the Pattern task, but this is unlikely to be statistically significant). To distinguish the models, therefore, we need to implement *harder tasks* and run them on *longer sequences*.

- All weights are initialised with the default `glorot_uniform_initializer` (see [the TensorFlow docs](https://www.tensorflow.org/api_docs/python/tf/get_variable)) and biases are initialised to zero. For more on initialisation see [here](https://plus.google.com/+SoumithChintala/posts/RZfdrRQWL6u) and [here](http://stackoverflow.com/questions/40318812/tensorflow-rnn-weight-matrices-initialization).

## Setting up TensorFlow on AWS

Following the instructions [here](https://aws.amazon.com/blogs/ai/the-aws-deep-learning-ami-now-with-ubuntu/) for the AWS Deep Learning AMI with Ubuntu. Our current machines are

```
Name        Type        Port    vCPU    Mem     Price   
=========================================================
[Tesla]     p2.xlarge	8880    4       61      $0.9 per Hour
[Frege]     g2.2xlarge	8881    8	    60 SSD	$0.65 per Hour
[Leibniz]   g2.2xlarge	8882    8	    60 SSD	$0.65 per Hour
[Turing]    p2.xlarge	8883    4       61      $0.9 per Hour
[Wiener]    p2.xlarge	8884    4       61      $0.9 per Hour
```

Here the "Port" denotes the port that we should use when creating an `ssh` tunnel to the remote server, in order to run Jupyter. That is, you should connect to the server with Port `<Port>` and IP `<IP>` using

```
ssh -L localhost:<Port>:localhost:8888 -i Virginia.pem ubuntu@<IP>
```

For convenience of cut and paste here are the commands expanded in each case:

```
[Tesla] ssh -L localhost:8880:localhost:8888 -i Virginia.pem ubuntu@limitordinal.org
[Frege] ssh -L localhost:8881:localhost:8888 -i Virginia.pem ubuntu@34.206.99.116
[Leibniz] ssh -L localhost:8882:localhost:8888 -i Virginia.pem ubuntu@52.21.99.86
[Turing] ssh -L localhost:8883:localhost:8888 -i Virginia.pem ubuntu@34.206.82.20
[Wiener] ssh -L localhost:8884:localhost:8888 -i Virginia.pem ubuntu@34.199.65.56
```

To verify that the GPUs are actually being used by TensorFlow within your Jupyter session, run the code [here](https://www.tensorflow.org/tutorials/using_gpu). Note that the output they describe there will appear in the *Jupyter log* not in your notebook. What we see for the `p2.xlarge` machines is

```
name: Tesla K80
major: 3 minor: 7 memoryClockRate (GHz) 0.8235
pciBusID 0000:00:1e.0
Total memory: 11.17GiB
```

The other `g2.2xlarge` machines have

```
name: GRID K520
major: 3 minor: 0 memoryClockRate (GHz) 0.797
pciBusID 0000:00:03.0
Total memory: 3.94GiB
Free memory: 3.91GiB
```

See [these instructions](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-attaching-volume.html) for adding more persistent disk, and [these](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/UsingIAM.html) for granting other AWS accounts access to your instances.

### TensorBoard

The instructions are [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard). You can launch TensorBoard with `tensorboard --logdir=path/to/log-directory` and then navigate to `localhost:6006`.

### Upgrading TensorFlow 0.12 to 1.0

The problem with the Deep Learning AMI is that it has TensorFlow v0.12 installed, and we want v1.0 (particularly for `tf.tensordot`). That means we have to upgrade. First install CUDA 0.8 by running ([from here](http://expressionflow.com/2016/10/09/installing-tensorflow-on-an-aws-ec2-p2-gpu-instance/))

```
wget https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64-deb
rm cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64-deb
sudo apt-get update
sudo apt-get install -y cuda
```

Then add the following to `~/.profile` and run `source ~/.profile`

```
export CUDA_HOME=/usr/local/cuda
export CUDA_ROOT=/usr/local/cuda
export PATH=$PATH:$CUDA_ROOT/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64
```
Then follow the `pip` upgrade [instructions](https://www.tensorflow.org/install/install_linux) on the TensorFlow webpage by running

```
sudo pip uninstall tensorflow
sudo pip install tensorflow-gpu
```

Then follow the instructions on the TensorFlow webpage to check the GPU is working. Then run `jupyter notebook` as usual.

## Notes on other implementations

The most robust implementation we are aware of is `NTM-Lasagne` for which see [this](https://medium.com/snips-ai/ntm-lasagne-a-library-for-neural-turing-machines-in-lasagne-2cdce6837315#.arp7npxt3) blog post and the [GitHub repository](https://github.com/snipsco/ntm-lasagne). It is written for Theano. There is also [carpedm20](https://github.com/carpedm20/NTM-tensorflow) which we have looked at less. Essentially we confirmed that the `NTM-Lasagne` implementation makes the same initialisation choices that we made on our own, which the exception of the controller internal state. The following remarks pertain entirely to `NTM-Lasagne`.

Some general notes: they train the Copy task on `num_classes = 256 + 1` that is, an alphabet of `256` symbols plus a terminal symbol, and on sequences of length `N = 5`. The allowed rotations of the read and write addresses default to `3`, i.e. `[-1,0,1]`. The final output layer uses a sigmoid.

Note that the actual calculation of the weights (including sharpening) takes place in the function `get_weights` of `heads.py`. They train on about `1000000` samples.

#### Read and write addresses

From `init.py` we see that `init.OneHot` contains

```
def sample(self, shape):
M = np.min(shape)
arr = np.zeros(shape)
arr[:M,:M] += 1 * np.eye(M)
return arr
```

From `heads.py > class Head > self.weights_init` we see that the weight of a generic head is initialised using `init.OneHot( self, (1, self.memory_shape[0]) )` so `M = 1` and therefore the return value of `init.OneHot` will be a tensor of shape `[1, self.memory_shape[0] ]` with value `(1,0,0,...)`. That is, all read and write heads are by default initialised to be sharply focused at the zero position of the memory. This is not subject to learning (i.e. the initialisation vector is not a weight vector).

#### Memory state

Memory shapes default to `(128,20)` and are intialised according to `memory.py` the initialisation is a weight vector with `memory_init=lasagne.init.Constant(1e-6)`.

#### Controller internal state

The recurrent controller is the class `RecurrentController` in `controller.py` and is initialised using `lasagne.init.GlorotUniform` with no parameter. For the details of this initialiser in Theano see [here](http://lasagne.readthedocs.io/en/latest/modules/init.html). The evolution equation uses `lasagne.nonlinearities.rectify`. Note that the nonlinearity *defaults* are defined in `controller.py` but you really need to check `copy-task.py` to verify that these defaults are not overwritten (they are not, in this case).

#### Recurrent matrices H, U, B

In the notation of their library these weight matrices are respectively `W_hid_to_hid`, `W_in_to_hid` and `b_hid_to_hid`. See the definition of the class `RecurrentController`. The initialisers are respectively `GlorotUniform()`, `GlorotUniform()` and `Constant(0.0)`.

#### Weights and biases for s, q, e, a and gamma

Both of `s,q` are described in class `Head` of `heads.py` and the relevant weights are `W_hid_to_shift` and `b_hid_to_shift`. The former is initialised with `GlorotUniform()` and the latter with `Constant(0.0)`. The nonlinearity is `lasagne.nonlinearities.softmax`.

The weight and bias for `e, a` are given in class `WriteHead` of the same file. In both cases, the weights are `GlorotUniform()` and the biases are `Contant(0.0)`. Similarly for `gamma`. The nonlinearities are respectively

```
Erase:  nonlinearities.hard_sigmoid
Add:    nonlinearities.ClippedLinear(low=0., high=1.)
Gamma:  lambda x: 1. + lasagne.nonlinearities.rectify(x)
```

Note that `hard_sigmoid` and `ClippedLinear` are defined in `nonlinearities.py`. The former is `theano.tensor.nnet.hard_sigmoid` and `ClippedLinear` is `theano.tensor.clip(x, low, high)`. See [this Theano page](http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html) for details: hard sigmoid is just a piecewise linear `ReLu` like approximation to the sigmoid, whereas `clip` just does what it says: it is like the `ReLu` but where values of `x >= 1` are sent to `1`.

**NOTE** in `copy-task.py` the nonlinearity default for the add vector (given above) is overridden with `lasagne.nonlinearities.rectify`.

### Tasks

See `utils/generators.py` and `examples/copy-task.py`. The Copy task uses `size = 8` and length `5` which in `utils/generators.py` means that we uniformly sample from the set of sequences of length `5` in the set `{0,1}^8`. This set has `2^8 = 256` elements.

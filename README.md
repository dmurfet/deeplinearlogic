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

These experiments were done with a version of the code now denoted `v1`. However this version of the code did not implement sharpening (see Eq. (9) of the NTM paper) and had some initialisation choices that made it basically impossible for any of the models to really use the memory in the manner intended. It is somewhat remarkable that they managed to converge to zero error in so many cases, given these handicaps (in hindsight, binary sequences, even of length `20`, may not be very challenging given this many weights). However, the truly useless nature of the `v1` code is made clear by the fact that the models trained using it completely fail to generalise (here we test generalisation by training on length `20` sequences and testing on length `> 20` sequences). So improvements are necessary.

In `v2` of the code, the implementation of which is ongoing as of the 13th of March, sharpening is implemented and the initialisations are fixed. So far this has been done for the NTM, but not the other models. But the tests of the `v2` NTM as currently implemented show it often failing to converge, and still not really using memory properly. So, **HELP**!

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

Some general notes: they train the Copy task on `num_classes = 8` that is, an alphabet of `8` symbols, and on sequences of length `N = 5`. They include an end marker symbol. 

### Questions

- How are the read and write addresses initialised?
- How is the memory state initialised?
- How is the controller internal state initialised?
- How are the recurrent matrices H, U, B initialised?
- How are the weights and biases generating the s, q, e, a and gamma initialised?
- Which rotations do they allow? e.g. `[-1,0,1]`.

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

Memory shapes default to `(128,20)` and are intialised according to `memory.py` the initialisation is a weight vector with 

```
memory_init=lasagne.init.Constant(1e-6)
trainable=True
```

I don't know exactly what trainable means here as I can't find the class `InputLayer`. Well, no matter, because in the `README.md` of the GitHub repo they specify that in fact they do not make the initial memory state learnable.

#### Controller internal state

The recurrent controller is the class `RecurrentController` in `controller.py` and is initialised using `lasagne.init.GlorotUniform` with no parameter. For the details of this initialiser in Theano see [here](http://lasagne.readthedocs.io/en/latest/modules/init.html).

#### Recurrent matrices H, U, B

In the notation of their library these weight matrices are respectively `W_hid_to_hid`, `W_in_to_hid` and `b_hid_to_hid`. See the definition of the class `RecurrentController`. The initialisers are respectively `GlorotUniform()`, `GlorotUniform()` and `Constant(0.0)`.

#### Weights and biases for s, q, e, a and gamma

Both of `s,q` are described in class `Head` of `heads.py` and the relevant weights are `W_hid_to_shift` and `b_hid_to_shift`. The former is initialised with `GlorotUniform()` and the latter with `Constant(0.0)`. 

The weight and bias for `e, a` are given in class `WriteHead` of the same file. In both cases, the weights are `GlorotUniform()` and the biases are `Contant(0.0)`. Similarly for `gamma`.

#### Allowed rotations

Defaults to `3`, i.e. `[-1,0,1]`.

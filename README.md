# Linear Logic and Recurrent Neural Networks

This is the repository of the paper "Linear Logic and Recurrent Neural Networks" and the associated TensorFlow implementation. At the moment the repo is private. The models that have been implemented so far are

- The ordinary NTM (see class `NTM` in `ntm.py`).
- The pattern NTM (see class `PatternNTM` in `ntm.py`) which is the model described in Section 4.1 of the paper.
- The alternative pattern NTM (see class `PatternNTM_alt` in `ntm.py`) which is the pattern NTM but with the controller allowed to manipulate the read address of the first memory ring directly.

## Results

See the [spreadsheet](https://docs.google.com/spreadsheets/d/1GqwW3ma7Cd1W8X8Txph9MPmLSkQ0C-i0tP0YHeINzMs/edit?usp=sharing) of the experiments I have run so far, on the ordinary NTM and Pattern NTM, on the following tasks on sequences of length `N = 20`:

- Copy task (as in the NTM paper),
- Repeat copy task (as in the NTM paper),
- Pattern task (defined in Section 4.1 of our paper).

The numbers recorded in the spreadsheet are the percentages of correct predictions for the digits of the output binary sequence (`0.50` meaning as good as chance, `0` meaning perfect predictions) for the test set (which is three times the size of the training set, which is in turn 1% of the sequences, around 10k).

## TODOs

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
    - Initialisation of the weights (see [here](https://plus.google.com/+SoumithChintala/posts/RZfdrRQWL6u) and [here](http://stackoverflow.com/questions/40318812/tensorflow-rnn-weight-matrices-initialization))
    - Regularisation (to e.g. force the memory to be used "properly")
	- Gradient clipping? This seems standard in the augmented RNN literature
	- Noise?
	- Curriculum learning?
- **Various**
    - Train on `N = 20` and test on `N > 20` length sequences (at the moment we are enumerating all sequences and then taking 1% for training and 3% for testing, but it would be better for larger `N` to just sample what we need?)

## Some lessons learned

- We default to `controller_state_size = 100` in all our experiments now. In the beginning we tried `50` or even `30` but the models often failed to converge, and this is not allowing us to really distinguish the NTM and Pattern NTM. This is also the same dimension as the controller in the original NTM paper.

- Even the "difficult" Pattern task is learned by both the NTM and Pattern NTM on some runs (other runs terminate with awful levels of error). It seems necessary to use generalisation as a discriminator between the two models; that is, as in the NTM paper, we should train on `N = 20` and test on longer sequences, to see if the algorithm that is being learned is correct.

- The current tests indicate that the Pattern NTM doesn't live up to the expectation that its special architecture makes it much better than the NTM at the pattern task. We have to both try harder patterns (i.e. the Multiple Pattern NTM) and develop the introspection and visualisation tools to try and diagnose what the Pattern NTM is doing. Maybe a small modification will help (for example, at the moment the controller can only modify the read address of the first memory ring via the second memory ring. Perhaps it should be able to *both* manipulate it directly *and* via the second memory ring).

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
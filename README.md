# Linear Logic and Recurrent Neural Networks

This is the repository of the paper "Linear Logic and Recurrent Neural Networks" and the associated TensorFlow implementation. At the moment the repo is private.

## Results

See the [spreadsheet](https://docs.google.com/spreadsheets/d/1GqwW3ma7Cd1W8X8Txph9MPmLSkQ0C-i0tP0YHeINzMs/edit?usp=sharing) of the experiments I have run so far, on the ordinary NTM and Pattern NTM, on two tasks. On the Copy task (which asks the controller to simply repeat a given input sequence) the NTM converges to zero easily. On the Pattern task (described in the section of the paper which discusses the Pattern NTM) both the NTM and Pattern NTM converge when the size of the controller state space is made sufficiently large. Generally it looks like the Pattern NTM outperforms the NTM on this task, but there is also a difference in the number of weights in the networks (with the Pattern NTM having several times more) so the difference between the models is not clear yet.

## TODOs

* TensorBoard, including visualising memory state
* How blurred is the memory state?
* Initialisation of the weights
* Regularisation
* Coding the relevant test functions
* Different optimiser?
* What batch sizes, training size, controller state size, memory size to use?

## Setting up TensorFlow on AWS

Following the instructions [here](https://aws.amazon.com/blogs/ai/the-aws-deep-learning-ami-now-with-ubuntu/) for the AWS Deep Learning AMI with Ubuntu. Note the P2 GPU compute instances are not available in all regions (we use US West (Oregon)) and that if your AWS account is new you may not be able to use the P2 instances. We use 

```
p2.xlarge   4   12  61  EBS Only    $0.9 per Hour
```

which is a machine with `4` vCPUs, `61`Gb of memory. We modified the `ssh` command given in the aforelinked instructions to

```
ssh -L localhost:8880:localhost:8888 -i Virginia.pem ubuntu@limitordinal.org
```

since we are running Jupyter locally on port `8888` and we want to be able to access both sessions (remote on AWS and local) at the same time. To verify this is working, that is, that the GPUs of the P2 instance are actually being used by TensorFlow within your Jupyter session, run the code [here](https://www.tensorflow.org/tutorials/using_gpu). Note that the output they describe there will appear in the *Jupyter log* not in your notebook. What we see is

```
name: Tesla K80
major: 3 minor: 7 memoryClockRate (GHz) 0.8235
pciBusID 0000:00:1e.0
Total memory: 11.17GiB
```

See [these instructions](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-attaching-volume.html) for adding more persistent disk.

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
# Linear Logic and Recurrent Neural Networks

This is the repository of the paper "Linear Logic and Recurrent Neural Networks" and the associated TensorFlow implementation. At the moment the repo is private, and this Readme will be used to flag some current TODOs in the code.

As of Friday 3rd of March, the NTM runs and on the Copy task (which asks the controller to simply repeat a given input sequence) with the following result.

```
N = 20, controller_state_size = 20, memory_address_size = 10, memory_content_size = 5, epoch = 100
Test error = 0.06
```

Some TODOs that come to mind:

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

First install CUDA 0.8 by running ([from here](http://expressionflow.com/2016/10/09/installing-tensorflow-on-an-aws-ec2-p2-gpu-instance/))

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
Then follow the Anaconda installation [instructions](https://www.tensorflow.org/install/install_linux) on the TensorFlow webpage (one could reasonably ask at this point if we needed to start with the Amazon AMI at all. Oh well, who knows). This has the advantage that it matches precisely the Anaconda distribution on our desktop. But this means that before running any commands you have to activate the container with

```
source activate tensorflow
```

Then follow the instructions on the TensorFlow webpage to check the GPU is working.

### Running a Jupyter notebook

After SSH-ing into the remote machine, activate the tensorflow virtual environment with `source activate tensorflow`, and then run `jupyter notebook`.
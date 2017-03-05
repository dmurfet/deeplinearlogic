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
p2.xlarge	4	12	61	EBS Only	$0.9 per Hour
```

which is a machine with `4` vCPUs, `61`Gb of memory. We modified the `ssh` command given in the aforelinked instructions to

```
ssh -L localhost:8880:localhost:8888 –i Virginia.pem ubuntu@limitordinal.org
```

since we are running Jupyter locally on port `8888` and we want to be able to access both sessions (remote on AWS and local) at the same time. To verify this is working, that is, that the GPUs of the P2 instance are actually being used by TensorFlow within your Jupyter session, run the code [here](https://www.tensorflow.org/tutorials/using_gpu). Note that the output they describe there will appear in the *Jupyter log* not in your notebook. What we see is

```
name: Tesla K80
major: 3 minor: 7 memoryClockRate (GHz) 0.8235
pciBusID 0000:00:1e.0
Total memory: 11.17GiB
```

See [these instructions](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-attaching-volume.html) for adding more persistent disk.

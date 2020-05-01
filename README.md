# CS5296 Group11 Course Project

## Introduction
The project aims to conduct machine learning algorithms on distributed machine learning platforms including Spark, TensorFlow and PMLS. The different performances of distributed platforms will be compared.

## Setup
1. You need to setup at least three AWS EC2 instances for experiment.</br> 
&ensp;In our experiment, we use three m4.large instances and each instance contains 4 vCores and 8 GiB memory.</br></br> 
2. You don't need to worry about the dateset since it is prepared. For the experiment on Spark and Tensorflow, the dateset will be downloaded automatically if it is not exist.</br> 


## Usage

### Spark
In the Spark, please make sure you have installed the time, pandas, sklearn, pyspark.ml and pyspark.sql package.
Run the following file to launch the program. 
```
./Spark/upload.py
```

### Tensorflow
Pre-requisites: python 2.7, tensorflow 1.10 (other version maybe work, but have not been tested), sklearn, time, numpy, math.</br> 

Run the following file to try nondistributed training (It takes about 300s for one epoch training). 
```
./Tensorflow/Code/Non_distributed_training.py
```
For Distributed training, you need to modify the distributed training parameter (i.e. change the ip address for ps_hosts and workers) in the ```./Tensorflow/Code/Distributed_training.py``` before you run it.</br> 

To conduct distributed training, run the following command for PS node.
```
python Distributed_training.py --job_name=ps --task_index=0
```
And the following command for worker 1
```
python Distributed_training.py --job_name=worker --task_index=0
```
And the following command for worker 2
```
python Distributed_training.py --job_name=worker --task_index=1
```
It takes about 150s for one epoch training.


### PMLS (Petuum)

Install PMLS following the illustration provided by the [documents of Petuum](https://pmls.readthedocs.io/en/latest/index.html).


Run the following command to launch the demo program. 


```
./PMLS/app/mlr/script/launch.py
```

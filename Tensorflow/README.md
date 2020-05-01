## Distributed Machine Learning on Tensorflow

## Pre-requisites
python 2.7, tensorflow 1.10 (other version maybe work, but have not been tested), sklearn, time, numpy, math. 

## Introduce
The code can be found here -> ```./Code/``` </br> 
The result(Screenshot) can be found here -> ```./Result/```

## Usage
Run the following file to try nondistributed training (It takes about 300s for one epoch training). 
```
./Code/Non_distributed_training.py
```
For Distributed training, you need to modify the distributed training parameter (i.e. change the ip address for ps_hosts and workers) in the ```./Code/Distributed_training.py``` before you run it.</br> 

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
It takes about 150s for one epoch training.<br/>

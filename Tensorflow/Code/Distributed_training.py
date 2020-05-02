# encoding:utf-8
import math
import tempfile
import time
import tensorflow as tf
import numpy as np
import time
from sklearn import *
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def next_batch(train_data, train_target, batch_size):  
    #Shuffle the data set
    index = [ i for i in range(0,len(train_target)) ]  
    np.random.shuffle(index);  
    
    batch_data = []; 
    batch_target = [];  

    #add
    for i in range(0,batch_size):  
        batch_data.append(train_data[index[i]]);  
        batch_target.append(train_target[index[i]])  
    return batch_data, batch_target 

flags = tf.app.flags


flags.DEFINE_integer('batch_size', 128, 'Training batch size ')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')

#Distributed training parameter
#one PS
flags.DEFINE_string('ps_hosts', '34.226.245.134:22221', 'Comma-separated list of hostname:port pairs')
#two worker
flags.DEFINE_string('worker_hosts', '3.84.91.252:22221,54.237.84.20:22221',
                    'Comma-separated list of hostname:port pairs')
#job name
flags.DEFINE_string('job_name', None, 'job name: worker or ps')
#index
flags.DEFINE_integer('task_index', None, 'Index of task within the job')
#issync
flags.DEFINE_integer("issync", None, "Whether to use distributed synchronous mode, 1:synchronous mode, 0:asynchronous mode")


FLAGS = flags.FLAGS


def main(unused_argv):
    
    #get data
    (data, target) = fetch_covtype(return_X_y=True)

    onehot_encoder = OneHotEncoder(sparse=False)
    target = onehot_encoder.fit_transform(target.reshape(len(target), 1))

    min_max_scaler = MinMaxScaler()
    data = min_max_scaler.fit_transform(data)

    #split
    trainX, testX, trainY, testY = train_test_split(data, target, train_size=0.8, test_size=0.2, random_state=4488)


    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('Must specify an explicit job_name !')
    else:
        print 'job_name : %s' % FLAGS.job_name
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('Must specify an explicit task_index!')
    else:
        print 'task_index : %d' % FLAGS.task_index

    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')

    # create cluster
    num_worker = len(worker_spec)
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == 'ps':
        server.join()

    is_chief = (FLAGS.task_index == 0)
    

    with tf.device(tf.train.replica_device_setter(
            cluster=cluster
    )):

        global_step = tf.Variable(0, name='global_step', trainable=False)  #global_step

        batch_size = FLAGS.batch_size     

        #bulid model
        X = tf.placeholder(tf.float32, [batch_size, 54], name='X_placeholder') 
        Y = tf.placeholder(tf.int32, [batch_size, 7], name='Y_placeholder')

        w = tf.Variable(tf.random_normal(shape=[54, 7], stddev=0.01), name='weights')
        b = tf.Variable(tf.zeros([1, 7]), name="bias")

        logits = tf.matmul(X, w) + b 

        #cross entropy
        entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')
        #mean
        loss = tf.reduce_mean(entropy)    

        learning_rate = FLAGS.learning_rate
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        #predit
        preds = tf.nn.softmax(logits)   
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))   
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))  



        #init
        init_op = tf.global_variables_initializer()
        train_dir = tempfile.mkdtemp()
        sv = tf.train.Supervisor(is_chief=is_chief, logdir=train_dir, init_op=init_op, recovery_wait_secs=1,
                                 global_step=global_step)

        if is_chief:
            print 'Worker %d: Initailizing session...' % FLAGS.task_index
        else:
            print 'Worker %d: Waiting for session to be initaialized...' % FLAGS.task_index
        sess = sv.prepare_or_wait_for_session(server.target)
        print 'Worker %d: Session initialization  complete.' % FLAGS.task_index

        time_begin = time.time()
        print 'Traing begins @ %f' % time_begin

        local_step = 0
        n_epochs = 1
        c_epochs = 0

        while True:
            batch_xs, batch_ys = next_batch(trainX,trainY,batch_size)
            train_feed = {X: batch_xs, Y: batch_ys}

            _, step = sess.run([optimizer, global_step], feed_dict=train_feed)
            local_step += 1

            c_epochs = math.ceil(step/(int(trainX.shape[0]/batch_size)))+1

            print 'Epoch: %d/%d : Worker %d: traing step %d (global step:%d)' % (c_epochs,n_epochs,FLAGS.task_index, local_step, step)

            if c_epochs > n_epochs:
                break

        time_end = time.time()
        print 'Training ends @ %f' % time_end
        train_time = time_end - time_begin
        print 'Training elapsed time:%f s' % train_time
            
  
        #Test
        n_batches = int(testX.shape[0]/batch_size) 
        total_correct_preds = 0

        for i in range(n_batches):
            X_batch, Y_batch = next_batch(testX,testY,batch_size)    
            accuracy_batch = sess.run([accuracy], feed_dict={X: X_batch, Y:Y_batch}) 
            total_correct_preds += accuracy_batch[0]
            
        print('Accuracy {0}'.format(total_correct_preds/testX.shape[0])) 

    sess.close()

if __name__ == '__main__':
    tf.app.run()

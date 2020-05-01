
# coding: utf-8
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



if __name__ == "__main__":
    
    #get Data
    (data, target) = fetch_covtype(return_X_y=True)

    #onehot encoding
    onehot_encoder = OneHotEncoder(sparse=False)
    target = onehot_encoder.fit_transform(target.reshape(len(target), 1))

    #normalization 
    min_max_scaler = MinMaxScaler()
    data = min_max_scaler.fit_transform(data)

    #split the data(2:8)
    trainX, testX, trainY, testY = train_test_split(data, target, train_size=0.8, test_size=0.2, random_state=4488)


    batch_size = 128     


    X = tf.placeholder(tf.float32, [batch_size, 54], name='X_placeholder') 
    Y = tf.placeholder(tf.int32, [batch_size, 7], name='Y_placeholder')

    w = tf.Variable(tf.random_normal(shape=[54, 7], stddev=0.01), name='weights')
    b = tf.Variable(tf.zeros([1, 7]), name="bias")

    logits = tf.matmul(X, w) + b 

    #cross entropy
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')
    #get mean
    loss = tf.reduce_mean(entropy)   

    learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


    #number of epochs
    n_epochs = 1

    with tf.Session() as sess:

        #writer = tf.summary.FileWriter('./LR/log', sess.graph)

        #start training
        start_time = time.time()
        sess.run(tf.global_variables_initializer()) 
        n_batches = int(trainX.shape[0]/batch_size)
        for i in range(n_epochs): 
            total_loss = 0

            for _ in range(n_batches):
                X_batch, Y_batch = next_batch(trainX,trainY,batch_size)
                #print(X_batch.shape)
                _, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y:Y_batch}) 
                total_loss += loss_batch
            print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

        print('Total time: {0} seconds'.format(time.time() - start_time))

        print('Training Finished!')
        
        
        #Test
        preds = tf.nn.softmax(logits)   
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))   
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))    

        n_batches = int(testX.shape[0]/batch_size) 
        total_correct_preds = 0

        for i in range(n_batches):
            X_batch, Y_batch = next_batch(testX,testY,batch_size)    #此处是从测试集中按批次的输入数据
            accuracy_batch = sess.run([accuracy], feed_dict={X: X_batch, Y:Y_batch}) 
            total_correct_preds += accuracy_batch[0]
        
        #calculate accuracy
        print('Accuracy {0}'.format(total_correct_preds/testX.shape[0]))


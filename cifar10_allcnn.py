# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 13:13:53 2016
@author: jeandut
"""

from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
import cPickle as pickle
import argparse
import dataset_class as dataset
import  sys
#
parser=argparse.ArgumentParser(description="Testing All-CNN on CIFAR-10 global contrast normalised and whitened without data-augmentation.")
parser.add_argument('--learning_rate', default=0.05, help="Initial Learning Rate")
parser.add_argument('--weight_decay',default=0.001, help="weight decay")
parser.add_argument('--data_dir',default="/home/shared/alham/grad_drop_experiment/", help="Directory with cifar-10-batches-py")

args=parser.parse_args()

WD=float(args.weight_decay)
starter_learning_rate=float(args.learning_rate)
batchdir=os.path.join(args.data_dir,"cifar-10-batches-py")

workdir='/home/shared/alham/grad_drop_experiment/cifar-10-batches-py/'
 
BATCH_SIZE=128
EPSILON_SAFE_LOG=np.exp(-50.)
SEED=321
PROPAGATE_ERROR = True
GRADIENT_DROP = True
CLASS_DIM = 16
DROP_RATE = 50

# tf.set_random_seed(SEED)

# note that you will still get slightly undeterministic result when you run the code under GPUs due to the GPUs nature. 

def grad_drop(input_grad, rate):
    old_shape = input_grad.get_shape()
    grad = tf.reshape(input_grad, [-1])
    size = grad.get_shape().as_list()[0]
    k = tf.maximum(1, (size * (100 - rate) ) // 100)
    v, _ = tf.nn.top_k(abs(grad), k)
    cut_point = v[-1]
    dropped_grad = tf.select(tf.less(abs(grad), cut_point), tf.zeros_like(grad), grad)
    return tf.reshape(dropped_grad, old_shape)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
#Loading CIFAR-10 from data_dir directory
#The full code for the preprocessing is taken from nervana neon repository Image.py which is also inspired from pylearn2

train_batches=[os.path.join(batchdir, "data_batch_"+str(i)) for i in range(1,6)]

Xlist, ylist=[], []
for batch in train_batches:
    with open(batch,'rb') as f:
        d=pickle.load(f)
        Xlist.append(d['data'])
        ylist.append(d['labels'])
        
X_train=np.vstack(Xlist)
y_train=np.vstack(ylist)

with open(os.path.join(batchdir,"test_batch"),'rb') as f:
    d=pickle.load(f)
    X_test, y_test= d['data'], d['labels']
    
y_train=np.reshape(y_train,(-1,1))
y_test=np.array(y_test).reshape(-1, 1)



#Applying gcn followed by whitening

def global_contrast_normalize(X, scale=1., min_divisor=1e-8):
    
    X=X-X.mean(axis=1)[:, np.newaxis]
    
    normalizers=np.sqrt((X**2).sum(axis=1))/ scale
    normalizers[normalizers < min_divisor]= 1.
    
    X /= normalizers[:, np.newaxis]
    
    return X
    
def compute_zca_transform(imgs, filter_bias=0.1):
    
    meanX=np.mean(imgs,0)
    
    covX=np.cov(imgs.T)
    
    D, E =np.linalg.eigh(covX+ filter_bias * np.eye(covX.shape[0], covX.shape[1]))
    
    assert not np.isnan(D).any()
    assert not np.isnan(E).any()
    assert D.min() > 0
    
    D= D** -0.5
    
    W=np.dot(E, np.dot(np.diag(D), E.T))
    return meanX, W

def zca_whiten(train, test, cache=None):
    if cache and os.path.isfile(cache):
        with open(cache,'rb') as f:
            (meanX, W)=pickle.load(f)
    else:
        meanX, W=compute_zca_transform(train)
        
        with open(cache,'wb') as f:
            pickle.dump((meanX,W), f , 2)
            
    train_w=np.dot(train-meanX, W)
    test_w=np.dot(test-meanX, W)
    
    return train_w, test_w
    

norm_scale=55.0
X_train=global_contrast_normalize(X_train, scale= norm_scale)
X_test=global_contrast_normalize(X_test, scale= norm_scale)

zca_cache=os.path.join(workdir,'cifar-10-zca-cache.pkl')
X_train, X_test=zca_whiten(X_train, X_test, cache=zca_cache)

#Reformatting data as images
X_train=X_train.reshape((X_train.shape[0],3,32,32)).transpose((0,2,3,1))
X_test=X_test.reshape((X_test.shape[0],3,32,32)).transpose((0,2,3,1))

#Reformatting labels with 16 one-hot encoding
one_hot_train=np.zeros((y_train.shape[0],CLASS_DIM),dtype="int64")
one_hot_test=np.zeros((y_test.shape[0],CLASS_DIM),dtype="int64")

for i in xrange(y_train.shape[0]):
    one_hot_train[i,y_train[i]]=1
for i in xrange(y_test.shape[0]):
    one_hot_test[i,y_test[i]]=1
    
y_train=one_hot_train.astype("float32")
y_test=one_hot_test.astype("float32")

CIFAR10=dataset.read_data_sets(X_train,y_train, X_test, y_test, None, False)
 

grad_errors = []

def _variable_with_weight_decay(shape,wd=WD):
    
    initial=tf.random_normal(shape,stddev=0.05,  seed=SEED)
    
    var=tf.Variable(initial)
    
    if wd is not None:
        weight_decay=tf.mul(tf.nn.l2_loss(var), wd, name="weight_loss")
        tf.add_to_collection("losses", weight_decay)

    # for gradient drop error
    grad_errors.append( tf.Variable(tf.zeros(shape), trainable=False) )
        
    return var
    
def conv(input_tensor, W, b, stride=1, padding="SAME"):
    _x = tf.nn.conv2d(input_tensor, W, strides=[1, stride, stride, 1], padding=padding)
    return tf.nn.relu(_x + b)
    
def avg_pool(input_tensor, k=2, stride=2):
    return tf.nn.avg_pool(input_tensor, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding="VALID")
    
def safelog(input_tensor):
    return tf.log(input_tensor+EPSILON_SAFE_LOG)


x=tf.placeholder(dtype=tf.float32, shape= [None, 32, 32, 3])
y_=tf.placeholder(dtype=tf.float32, shape= [None, CLASS_DIM])

#Placeholders for the dropout probabilities
keep_prob_input=tf.placeholder(tf.float32)
keep_prob_layers=tf.placeholder(tf.float32)

global_step=tf.Variable(0, trainable=False)


#Scheduling learning rate to drop from starter_learning_rate by a factor 10 after 200, 250 and 300 epochs with Momentum optimizer with momentum=0.9
def get_optimizer(global_step):
    
    NUM_EPOCHS_PER_DECAY_1=200
    NUM_EPOCHS_PER_DECAY_2=250
    NUM_EPOCHS_PER_DECAY_3=300
    
    LEARNING_RATE_DECAY_FACTOR=0.1
    num_batches_per_epoch=50000/128
    
    decay_steps_1=int(num_batches_per_epoch*NUM_EPOCHS_PER_DECAY_1)
    decay_steps_2=int(num_batches_per_epoch*NUM_EPOCHS_PER_DECAY_2)
    decay_steps_3=int(num_batches_per_epoch*NUM_EPOCHS_PER_DECAY_3)
    
    decayed_learning_rate_1=tf.train.exponential_decay(starter_learning_rate, 
                                                     global_step, 
                                                     decay_steps_1, 
                                                     LEARNING_RATE_DECAY_FACTOR,
                                                     staircase=True)
                                                     
    decayed_learning_rate_2=tf.train.exponential_decay(decayed_learning_rate_1, 
                                                     global_step, 
                                                     decay_steps_2, 
                                                     LEARNING_RATE_DECAY_FACTOR,
                                                     staircase=True)
                                                     
    decayed_learning_rate_3=tf.train.exponential_decay(decayed_learning_rate_2, 
                                                     global_step, 
                                                     decay_steps_3, 
                                                     LEARNING_RATE_DECAY_FACTOR,
                                                     staircase=True)
                                                     
    lr=decayed_learning_rate_3
    
    return tf.train.MomentumOptimizer(lr,0.9)


def connect_conv(input_tensor, shape, stride=1, padding = "SAME"):
    bias_size = shape[3]
    __w = _variable_with_weight_decay(shape)
    __b = _variable_with_weight_decay([bias_size])
    return conv(input_tensor, __w, __b, stride, padding)

def inference(x_input):
    x_dropped=tf.nn.dropout(x_input, keep_prob=keep_prob_input, seed= SEED)
    
    conv1 = connect_conv(x_dropped, [3, 3, 3, 96] )
    conv2 = connect_conv(conv1, [3, 3, 96, 96] )
    conv3 = connect_conv(conv2, [3, 3, 96, 96] , stride=2)
    conv3_dropped=tf.nn.dropout(conv3, keep_prob=keep_prob_layers, seed= SEED)

    conv4 = connect_conv(conv3_dropped, [3, 3, 96, 192] )
    conv5 = connect_conv(conv4, [3, 3, 192, 192] )
    conv6 = connect_conv(conv5, [3, 3, 192, 192],  stride=2)
    conv6_dropped=tf.nn.dropout(conv6, keep_prob=keep_prob_layers, seed= SEED)
    
    conv7 = connect_conv(conv6_dropped, [3, 3, 192, 192])
    conv8 = connect_conv(conv7, [1, 1, 192, 192], padding="VALID")
    conv9 = connect_conv(conv8, [1, 1, 192, CLASS_DIM], padding="VALID")
    logits=avg_pool(conv9, 8, 8)

    return logits

    
def ce(y_pred, labels):
    cross_entropy=tf.reduce_sum(labels*safelog(y_pred),1)
    cross_entropy_mean=-tf.reduce_mean(cross_entropy)
    tf.add_to_collection("losses", cross_entropy_mean)
    
    return tf.add_n(tf.get_collection("losses"),"total_loss")
    
def acc(y_pred, labels):
    correct_prediction =tf.equal(tf.argmax(y_pred,1), tf.argmax(labels,1))
    return tf.reduce_mean(tf.cast(correct_prediction,tf.float32),0)



logits=inference(x)

pred=tf.nn.softmax(tf.reshape(logits,(-1,CLASS_DIM)))

loss=ce(pred, y_)

optimizer=get_optimizer(global_step)
grad_and_vars = optimizer.compute_gradients(loss) 


if GRADIENT_DROP:  
    drop_rate = tf.placeholder(tf.int32)
    print('Training with gradient drop rate of ', DROP_RATE)
    final_grads = [grad + (0.5 * prev_grad) for prev_grad, (grad, _) in zip(grad_errors, grad_and_vars)]
    isflush = tf.placeholder(tf.bool)
    
    dropped_grads = [(grad_drop(final_grad, drop_rate), var) for final_grad, (_, var) in zip(final_grads, grad_and_vars) ]
 
    error_update_ops = []
    if PROPAGATE_ERROR:
        print('Propagating the Error')
        idx = 0
        for (drop_grad, _ ),  final_grad in zip(dropped_grads, final_grads):
            error_update_ops.append( tf.assign(grad_errors[idx] , final_grad - drop_grad) )
            idx += 1
    train_step = optimizer.apply_gradients(dropped_grads, global_step = global_step)
else:
    print('Training without gradient drop')
    train_step = optimizer.apply_gradients(grad_and_vars, global_step = global_step)

accuracy=acc(pred, y_)

sess=tf.Session()
sess.run(tf.initialize_all_variables())
#Going through 350 epochs (there is 390 batches by epoch)
STEPS, ACC_TRAIN, COST_TRAIN=[], [], []
dr = DROP_RATE
for i in xrange(0,136500):
    batch=CIFAR10.train.next_batch(128)
    if i%1000==0:
        FINAL_ACC=0.
        for j in xrange(0,10):
            FINAL_ACC+=0.1*sess.run(accuracy, feed_dict={x: CIFAR10.test.images[j*1000:(j+1)*1000], y_: CIFAR10.test.labels[j*1000:(j+1)*1000], keep_prob_input: 1., keep_prob_layers: 1.}) 
        print("current eval accuracy :", FINAL_ACC)
        eprint("current eval accuracy :", FINAL_ACC)

    if i%100==0:
        
        acc_batch, loss_batch = sess.run([accuracy, loss], feed_dict={x: batch[0], y_: batch[1], keep_prob_input: 1., keep_prob_layers: 1.})
        print("Step: %s, Acc: %s, Loss: %s"%(i,acc_batch, loss_batch))
        eprint("Step: %s, Acc: %s, Loss: %s"%(i,acc_batch, loss_batch))
        
        STEPS.append(i)
        ACC_TRAIN.append(acc_batch)
        COST_TRAIN.append(loss_batch)
    ops = [dropped_grads, train_step]

    if GRADIENT_DROP and PROPAGATE_ERROR:
        for u in error_update_ops:
            ops.append(u)

    ret = sess.run(ops, feed_dict={x: batch[0], y_: batch[1], keep_prob_input: 0.8, keep_prob_layers: 0.5, drop_rate:dr })


    
FINAL_ACC=0.
for i in xrange(0,10):
    FINAL_ACC+=0.1*sess.run(accuracy, feed_dict={x: CIFAR10.test.images[i*1000:(i+1)*1000], y_: CIFAR10.test.labels[i*1000:(i+1)*1000], keep_prob_input: 1., keep_prob_layers: 1.}) 
   
    
print("Final accuracy on test set:", FINAL_ACC)  
    
 

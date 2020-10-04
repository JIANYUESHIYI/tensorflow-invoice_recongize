#-*- coding:utf-8 -*-
import sys
import os
import time
import random
import numpy as np
import tensorflow as tf
import cv2 as cv

SIZE = 256
WIDTH = 16
HEIGHT = 16
NUM_CLASSES = 36
iterations = 800

SAVER_DIR = "model_file/"
LOG_DIR = SAVER_DIR + "logfile/"
DATASET_DIR = "/"


def get_train():
    time_begin = time.time()
    #Get the total number of pictures in the first iteration
    input_count = 0
    for i in range(0, NUM_CLASSES):
        # dir = 'train/%s/' % i
        dir = DATASET_DIR + 'train/%s/' % i
        for root,dirs,files in os.walk(dir):
            for filename in files:
                if filename == "Thumbs.db":
                    continue
                # if filename == 
                input_count = input_count + 1
    
    input_images = np.array([[0]*SIZE for i in range(input_count)])     
    input_labels = np.array([[0]*NUM_CLASSES for i in range(input_count)])    
    #Get data content in the second iteration
    index = 0
    for i in range(0, NUM_CLASSES):
        dir = DATASET_DIR + 'train/%s/' % i
        a = 0
        for root,dirs,files in os.walk(dir):
            for filename in files:
                if filename == "Thumbs.db":
                    continue
                filename = dir + filename
                img = cv.imread(filename,0)
 
                height = img.shape[0]       
                width = img.shape[1]       
                a = a + 1
                for h in range(0,height):
                    for w in range(0,width):
                        m = img[h][w]
                        if m > 150:
                            input_images[index][w+h*width] = 1
                        else:
                            input_images[index][w+h*width] = 0
                input_labels[index][i] = 1
                index = index + 1
    print('[---------------]')
    print('train data successfully loaded!')
    return time_begin, input_count, input_images, input_labels

    
def get_test():
    tes_count = 0
    for i in range(0, NUM_CLASSES):
        dir = DATASET_DIR + 'test/%s/' % i
        for root,dirs,files in os.walk(dir):
            for filename in files:
                if filename == "Thumbs.db":
                    continue
                tes_count = tes_count + 1
    tes_images = np.array([[0]*SIZE for i in range(tes_count)])     
    tes_labels = np.array([[0]*NUM_CLASSES for i in range(tes_count)])   
    index = 0
    for i in range(0, NUM_CLASSES):
        dir = DATASET_DIR + 'test/%s/' % i
        for root,dirs,files in os.walk(dir):
            for filename in files:
                if filename == "Thumbs.db":
                    continue
                filename = dir + filename
                img = cv.imread(filename,0)
                height = img.shape[0]      
                width = img.shape[1]       
                for h in range(0,height):
                    for w in range(0,width):
                        m = img[h][w]
                        if m > 150:
                            tes_images[index][w+h*width] = 1
                        else:
                            tes_images[index][w+h*width] = 0
                tes_labels[index][i] = 1
                index = index + 1
    print('[---------------]')
    print('test data successfully loaded!')
    return tes_count, tes_images, tes_labels


class CNN:
    def __init__(self,iterations, time_begin, input_count, input_images, 
                input_labels, tes_count, tes_images, tes_labels):
        self.W_conv1 = None
        self.b_conv1 = None
        self.W_conv2 = None
        self.b_conv2 = None
        self.W_fc1 = None
        self.b_fc1 = None
        self.W_fc2 = None
        self.b_fc2 = None
        self.keep_prob = None 
        self.iterations = iterations 
        self.time_begin = time_begin
        self.input_count = input_count
        self.input_images = input_images
        self.input_labels = input_labels
        self.tes_count = tes_count
        self.tes_images = tes_images
        self.tes_labels = tes_labels

    def conv_layer(self,inputs, W, b):
        L1_conv = tf.nn.conv2d(inputs,W,strides=[1,1,1,1],padding='SAME')                         
        L1_relu = tf.nn.relu(L1_conv + b)                                                               
        return tf.nn.max_pool(L1_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    def full_connect(self,inputs,W,b):
        return tf.nn.relu(tf.matmul(inputs,W)+b)

    def average(self, seq):
        return float(sum(seq)) / len(seq)

    def fit(self):
        # with tf.name_scope('Input'):
        x = tf.placeholder(tf.float32,shape=[None,SIZE], name = "x_input")   
        y_ = tf.placeholder(tf.float32,shape=[None,NUM_CLASSES])    
        self.keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
        x_image = tf.reshape(x,[-1,WIDTH,HEIGHT,1])      

        #The first convolution layer, 16*16*1-> 8*8*12
        self.W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,12],stddev=0.1),name="W_conv1")    
        self.b_conv1 = tf.Variable(tf.constant(0.1,shape=[12]),name="b_conv1")                       
        L1_pool = CNN.conv_layer(self, x_image, self.W_conv1, self.b_conv1)  

        #The second convolution layer, 8*8*12-> 4*4*24
        self.W_conv2 = tf.Variable(tf.truncated_normal([5,5,12,24],stddev=0.1),name="W_conv2")
        self.b_conv2 = tf.Variable(tf.constant(0.1,shape=[24]),name="b_conv2")
        L2_pool = CNN.conv_layer(self, L1_pool, self.W_conv2, self.b_conv2)

        # #The third convolution layer, 4*4*24-> 2*2*36
        # self.W_conv3 = tf.Variable(tf.truncated_normal([5,5,24,36],stddev=0.1),name="W_conv3")
        # self.b_conv3 = tf.Variable(tf.constant(0.1,shape=[36]),name="b_conv3")
        # L3_pool = CNN.conv_layer(self, L2_pool, self.W_conv3, self.b_conv3)

        #The first fullconnect layer, 2*2*36-> 64
        self.W_fc1 = tf.Variable(tf.truncated_normal([4*4*24,128],stddev=0.1),name="W_fc1")   
        self.b_fc1 = tf.Variable(tf.constant(0.1,shape=[128]),name="b_fc1")
        h_pool2_flat = tf.reshape(L2_pool,[-1,4*4*24])                
        h_fc1 = CNN.full_connect(self, h_pool2_flat, self.W_fc1, self.b_fc1)                   
        h_fc1_drop = tf.nn.dropout(h_fc1,self.keep_prob)

        #readout layer, 64->NUM_CLASSES
        self.W_fc2 = tf.Variable(tf.truncated_normal([128,NUM_CLASSES],stddev=0.1),name="W_fc2")
        self.b_fc2 = tf.Variable(tf.constant(0.1,shape=[NUM_CLASSES]),name="b_fc2")
        y_conv = tf.matmul(h_fc1_drop, self.W_fc2) + self.b_fc2      
        pyx = tf.nn.softmax(y_conv, name = "y_pred")
        
        with tf.name_scope('loss'):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
            loss_summary = tf.summary.scalar('loss', cross_entropy)
            
        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer((1e-5)).minimize(cross_entropy)                                        
         
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))                                         
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))                                          
                acc_summary = tf.summary.scalar('accuracy', accuracy)
        
        #Use tensorboard to record loss and accuracy charts
        merged = tf.summary.merge([loss_summary, acc_summary])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())   

            time_elapsed = time.time() - self.time_begin  
            print('[=============]')
            print("Reading data takes time %d second." % time_elapsed)
            print ("Read a total of %s training data, %s tags." % (self.input_count, self.input_count))
            print ("Read a total of %s test data %s tags." % (self.tes_count, self.tes_count))

            # saver = tf.compat.v1.train.Saver()
            saver = tf.train.Saver()
            writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

            tr_start = time.time()
            batch_size = 64       
            batches_count = int(self.input_count/batch_size)
            remainder = self.input_count % batch_size
            print ("Training set divided into %s batchs, Each batch before have %s datas, Last Batch have %s datas." % (batches_count+1, batch_size, remainder))

            print('[=============]')
            print('Train start!')
            for it in range(self.iterations + 1):
                sum_loss = []
                for n in range(batches_count):
                    summary, loss, out = sess.run([merged, cross_entropy, train_step], feed_dict = {x:self.input_images[n*batch_size:(n+1)*batch_size],
                                                                                                    y_:self.input_labels[n*batch_size:(n+1)*batch_size],self.keep_prob:0.5})   
                    sum_loss.append(loss)
                if remainder > 0:
                    start_index = batches_count * batch_size
                    loss, out = sess.run([cross_entropy, train_step], feed_dict = {x:self.input_images[start_index:input_count-1],
                                                                                   y_:self.input_labels[start_index:input_count-1],
                                                                                   self.keep_prob:0.5})
                    sum_loss.append(loss)
                writer.add_summary(summary, it)
                avg_loss = CNN.average(self, sum_loss)
                iterate_accuracy = 0
                if it % 5 == 0:
                    loss1, iterate_accuracy = sess.run([cross_entropy,accuracy], feed_dict = {x :self.tes_images,
                                                                                              y_ :self.tes_labels,self.keep_prob : 1.0})
                    print('The %0.4d iteration training accuracy is: %0.5f%% ' % (it,iterate_accuracy*100) + 
                          '    Loss value is: %0.5f' % avg_loss + '    Test loss value is: %0.5f.' % loss1)
                    if iterate_accuracy >= 0.9999999:
                        break
                    acc = iterate_accuracy

            print('[=============]')
            print ('Train end!')
            tr_end = time.time() - tr_start
            print('The final accuracy of training is: %0.3f%%!' % (acc*100))
            print('The final loss of training is: %0.5f!' % (avg_loss))
            print ("This training takes a total of time: %d second." % tr_end)

            if not os.path.exists(SAVER_DIR) :
                print ('The training model save path does not exist, Automatically create a path now.')
                os.makedirs(SAVER_DIR)
            saver_path = saver.save(sess, SAVER_DIR)
            print("The model save path is: %s" % saver_path)
            print('The graph save path is: %s' % LOG_DIR)
            writer.close()



if __name__ == "__main__":
    
    # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    time_begin, input_count, input_images, input_labels = get_train()
    tes_count, tes_images, tes_labels = get_test()

    cnn = CNN(iterations, time_begin, input_count, input_images, 
              input_labels, tes_count, tes_images, tes_labels)
    cnn.fit()























#????锟斤�?????????????????????
    # val_count = 0
    # for i in range(NUM_CLASSES):
    #     dir = 'valid/%s/' % i
    #     for root,dirs,files in os.walk(dir):
    #         for filename in files:
    #             if filename == "Thumbs.db":
    #                 continue
    #             val_count = val_count + 1
    # val_images = np.array([[0]*SIZE for i in range(val_count)])     
    # val_labels = np.array([[0]*NUM_CLASSES for i in range(val_count)])    
    # index = 0
    # for i in range(NUM_CLASSES):
    #     dir = 'valid/%s/' % i
    #     for root,dirs,files in os.walk(dir):
    #         for filename in files:
    #             if filename == "Thumbs.db":
    #                 continue
    #             filename = dir + filename
    #             img = cv.imread(filename,0)
    #             height = img.shape[0]      
    #             width = img.shape[1]       
    #             for h in range(0,height):
    #                 for w in range(0,width):
    #                     m = img[h][w]
    #                     if m > 150:
    #                         val_images[index][w+h*width] = 1
    #                     else:
    #                         val_images[index][w+h*width] = 0
    #             val_labels[index][i] = 1
    #             index = index + 1
    #     print('[---------------]')
    #     print('%s 锟斤拷图片锟斤拷锟斤拷锟斤拷锟�??' % i)

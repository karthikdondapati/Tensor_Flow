##Homework For Karthik Reddy Dondapati X145608

import tensorflow as tf
import numpy as np
A=tf.placeholder(tf.float32,shape=None,name="A")
A_rand = tf.random_normal([100],mean=1,stddev=2,name="A_rand")
A_rand_np=A_rand.eval()
D=tf.reduce_sum(A,name="D")
C=tf.reduce_mean(A,name="C")
B=tf.reduce_prod(A,name="B")
logs_dir = '/Users/karthikreddydondapati/events_logs_file'
E=tf.add(C,B)
F=tf.multiply(E,D)

with tf.Session() as sess:
    writer=tf.summary.FileWriter(logs_dir,sess.graph)
    sess.run([F],feed_dict={A:A_rand_np})



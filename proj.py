import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

#use normal distribution to initialize weights
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#use normal distribution to initialize bias
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#input and output size
x = tf.placeholder(tf.float32, shape=[None, 4])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

W_fc1 = weight_variable([4, 16])
b_fc1 = bias_variable([16])

#second layer
h_fc1 = tf.nn.sigmoid(tf.matmul(x, W_fc1) + b_fc1)

W_fc15= weight_variable([16, 16])
b_fc15= bias_variable([16])

#third layer
h_fc15 = tf.nn.sigmoid(tf.matmul(h_fc1, W_fc15) + b_fc15)

W_fc2 = weight_variable([16, 2])
b_fc2 = bias_variable([2])

#output layer
y_pre = tf.matmul(h_fc15, W_fc2) + b_fc2

#regularization
reg_constant=0
l2_error = tf.reduce_mean(tf.pow(y_pre-y_,2))
+reg_constant*tf.nn.l2_loss(W_fc1)
+reg_constant*tf.nn.l2_loss(W_fc2)
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(l2_error)
l2_dis=tf.reduce_mean(tf.pow(y_pre-y_,2))

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

X = pd.read_csv('data.csv')
X=X.values

Y=X[:,4:]
X=X[:,0:4]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.20)

#iteration number:30000
for i in range(30000):
    sess.run(train_step,feed_dict={x: Xtrain, y_: ytrain})
    print i
    print("train error %g"%(sess.run(l2_error,feed_dict={x: Xtrain,y_: ytrain})))
print("test error %g"%(sess.run(l2_dis,feed_dict={x: Xtest,y_: ytest})))
    
start_point=np.array([0.0,0.0,0.707106781187,0.658106781187])

pdata=np.zeros([101,4])
pdata[0]=start_point

for i in range(100):
    pdata[i+1,0:2]=pdata[i,2:4]
    pdata[i+1,2:4]=sess.run(y_pre,feed_dict={x: [pdata[i]]})
    if(pdata[i+1,3]<=0):break
    
pdata=pdata[:,0:2]
out=pd.DataFrame(pdata,columns=['x','y'])
out.to_csv('result.csv')

import tensorflow as tf  

print(tf.version)

#skaler olarak da bilinen, derece 0 olan bir tensördür.
string = tf.Variable("this is a string", tf.string) 
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)
#print(string)

rank1_tensor = tf.Variable(["Test"], tf.string) 
rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)
#print(rank1_tensor)

tf.rank(rank2_tensor)

rank2_tensor.shape

tensor1 = tf.ones([1,2,3])  
tensor2 = tf.reshape(tensor1, [2,3,1]) 
tensor3 = tf.reshape(tensor2, [3, -1])
#print(tensor1)
#print(tensor2)
#print(tensor3)

# Creating a 2D tensor
matrix = [[1,2,3,4,5],
          [6,7,8,9,10],
          [11,12,13,14,15],
          [16,17,18,19,20]]

tensor = tf.Variable(matrix, dtype=tf.int32) 
print(tf.rank(tensor))
print(tensor.shape)

import tensorflow as tf
import numpy as np


sess=tf.Session()    
saver = tf.train.import_meta_graph('model/-38.meta')
saver.restore(sess,tf.train.latest_checkpoint('model/'))
print("Model Restored")

#input
test_data = np.load("../test_data.npy") 
test_data = np.float32(test_data) 
labels = np.load("../test_labels.npy")

#init
eval_labels = []
count  = 0

# Now, let's access and create placeholders variables and
# create feed-dict to feed new data
 
graph = tf.get_default_graph()
input = graph.get_tensor_by_name("input:0")
print("Input Restored")
feed_dict ={input:test_data}
 
#Now, access the op that you want to run. 
op_to_restore = graph.get_tensor_by_name("prediction:0")
print("Operation Restored")
 
results = sess.run(op_to_restore,feed_dict)


for i in range(0,len(results)):
	if(labels[i] == np.argmax(results[i])):
		count+=1
	eval_labels.append(np.argmax(results[i]))

accuracy = float(float(count)/float(len(results))*100)
print(accuracy)
np.save("eval_labels.npy",eval_labels)
#This will print 60 which is calculated 
#using new values of w1 and w2 and saved value of b1. 

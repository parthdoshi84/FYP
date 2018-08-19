from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from darkflow.net.build import TFNet
import cv2
from PIL import Image
import binascii
from threading import Thread
import operator


app = Flask(__name__)


#init dictionaries



#yolo load 20 classes
options = {"pbLoad": "darkflow-master/built_graph_original/tiny-yolo-voc.pb", "metaLoad": "darkflow-master/built_graph_original/tiny-yolo-voc.meta", "threshold": 0.2}
tfnet = TFNet(options)

#yolo load 10 classes
options = {"pbLoad": "darkflow-master/built_graph_3000_images_10_classes/tiny-yolo-voc-10-classes-3000-images.pb", "metaLoad": "darkflow-master/built_graph_3000_images_10_classes/tiny-yolo-voc-10-classes-3000-images.meta", "threshold": 0.2}
tfnet1 = TFNet(options)

#emotion_recogniton load
sess=tf.Session()    
saver = tf.train.import_meta_graph('Expression_Recognition/Model6/model/-38.meta')
saver.restore(sess,tf.train.latest_checkpoint('Expression_Recognition/Model6/model/'))
print("Model Restored")
graph = tf.get_default_graph()
input = graph.get_tensor_by_name("input:0")
print("Input Restored")
op_to_restore = graph.get_tensor_by_name("prediction:0")



def object_recognition_function(desc, object_dict,person_dict):
	
	imgcv = cv2.imread("hope.png")
	results = tfnet.return_predict(imgcv)
	
	dictionary = dict()
	
	
	for i in range(0,len(results)):
		dictionary = dict()
		if(results[i]['label'] == 'person'):
			dictionary['label'] = results[i]['label']
			dictionary['confidence'] = str(results[i]['confidence'])
			dictionary['top_left_x'] = str(results[i]['topleft']['x'])
			dictionary['top_left_y'] = str(results[i]['topleft']['y'])
			dictionary['bottom_right_x'] = str(results[i]['bottomright']['x'])
			dictionary['bottom_right_y'] = str(results[i]['bottomright']['y'])
			person_dict[str(i)] = dictionary
		else:
			dictionary['label'] = results[i]['label']
			dictionary['confidence'] = str(results[i]['confidence'])
			dictionary['topleft'] = str(results[i]['topleft']['x']) + "," + str(results[i]['topleft']['y'])
			dictionary['bottomright'] = str(results[i]['bottomright']['x']) + "," + str(results[i]['bottomright']['y'])
			object_dict[str(i)] = dictionary
		
	
def object_recognition_function_10_classses(desc, object_dict):
	
	imgcv = cv2.imread("hope.png")
	results = tfnet1.return_predict(imgcv)
	
	dictionary = dict()
	
	
	for i in range(0,len(results)):
		dictionary = dict()
		
			
		dictionary['label'] = results[i]['label']
		dictionary['confidence'] = str(results[i]['confidence'])
		dictionary['topleft'] = str(results[i]['topleft']['x']) + "," + str(results[i]['topleft']['y'])
		dictionary['bottomright'] = str(results[i]['bottomright']['x']) + "," + str(results[i]['bottomright']['y'])
		object_dict[str(i)] = dictionary
	
def pre_pocessing_function():
	f = cv2.imread("hope.png")
	test_data = []
	face_cascade = cv2.CascadeClassifier(r"C:\Users\Parth123\Downloads\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml")
	W, H, D = f.shape
	gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
	
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	cords = []
	for (x, y, w, h) in faces:
		new_arr = []
		individual_cords = []
		crop_img = gray[y:y + h, x:x + w]
		print(x,y,x+w,y+h)
		resized_image = cv2.resize(crop_img, (48, 48))
		cv2.imwrite("saved.png",resized_image)
		reshaped = np.reshape(resized_image, 48*48)
		mean = np.mean(reshaped)
		std = np.std(reshaped)
		new_arr = (reshaped - mean)/256
		
		test_data.append(new_arr)
		individual_cords.append(x)
		individual_cords.append(y)
		individual_cords.append(x+w)
		individual_cords.append(y+h)
		cords.append(individual_cords)
		
	return test_data, cords
	
def emotion_recognition_function(desc,emotion):
	test_data,cords = pre_pocessing_function()
	
	if(test_data):
		feed_dict ={input:test_data}
		
		results = sess.run(op_to_restore,feed_dict)
		labels = []
		for i in range(0,len(results)):
			labels.append(np.argmax(results[i]))
				
		print(labels)
		for i in range(0,len(labels)):
			emotion_dict = dict()
			if labels[i] == 0:
				emotion_dict["emotion"] = "angry"
			elif labels[i] == 1:
				emotion_dict["emotion"] = "disgust"
			elif labels[i] == 2:
				emotion_dict["emotion"] = "fear"
			elif labels[i] == 3:
				emotion_dict["emotion"] = "happy"
			elif labels[i] == 4:
				emotion_dict["emotion"] = "sad"
			elif labels[i] == 5:
				emotion_dict["emotion"] = "surprise"
			else:
				emotion_dict["emotion"] = "neutral"
		
			emotion_dict["top_left_x"] = str(cords[i][0])
			emotion_dict["top_left_y"] = str(cords[i][1])
			emotion_dict["bottom_right_x"] = str(cords[i][2])
			emotion_dict["bottom_right_y"] = str(cords[i][3])
			
			emotion[str(i)] = emotion_dict
		


					
	
	


	

@app.route('/', methods=['POST'])
def main():
	
	if request.method == 'POST':
		content = request.form['pic']
		
		#save image
		new_content = binascii.a2b_base64(content)
		f = open('hope.png', 'wb')
		f.write(bytearray(new_content))
		f.close()
	
		#init dictionaries
		object_dict = dict()
		object_dict_10_classes = dict()
		emotion_dict = dict()
		person_dict = dict()
		mapping = dict()
		main_dict = dict()
		
		response_count = 0
		
		
		emotion_recognition = Thread(target = emotion_recognition_function, args=("1",emotion_dict))
		object_recognition = Thread(target = object_recognition_function, args=("2",object_dict,person_dict))
		object_recognition_10_classes = Thread(target = object_recognition_function_10_classses, args=("3",object_dict_10_classes))
		
		
		
		#start yolo thread
		object_recognition.start()
		
		#start yolo thread 10 classes
		object_recognition_10_classes.start()

		#start emotion thread
		emotion_recognition.start()
		
		
		
		#wait for completion
		emotion_recognition.join()
		object_recognition.join()
		object_recognition_10_classes.join()
		
		
		for key_person, value_person in person_dict.items():
			for key_emotion, value_emotion in emotion_dict.items():
				diff = abs(int(value_person["top_left_x"]) - int(value_emotion["top_left_x"])) + abs(int(value_person["top_left_y"]) - int(value_emotion["top_left_y"])) + abs(int(value_person["bottom_right_x"]) - int(value_emotion["bottom_right_x"])) + abs(int(value_person["bottom_right_y"]) - int(value_emotion["bottom_right_y"])) 
				mapping[str(key_person) + "," + str(key_emotion)] = diff 
		
		sorted_mapping = sorted(mapping.items(), key=operator.itemgetter(1))
		
		
		iter_range = min(len(person_dict),len(emotion_dict))
		used_indexes = []
		for i in range(0,len(sorted_mapping)):
			person_index, emotion_index = sorted_mapping[i][0].split(",")
			used_indexes.append(person_index)
			temp = person_dict[person_index]
			temp['emotion'] = emotion_dict[emotion_index]['emotion'] 
			main_dict[str(response_count)]= temp
			response_count+=1
			if response_count == len(emotion_dict):
				break
		for key,value in person_dict.items():
			if key not in used_indexes:
				main_dict[str(response_count)] = person_dict[key]
				response_count+=1 
		
		'''for key,value in emotion_dict.items():
			main_dict[str(response_count)] = emotion_dict[key]
			response_count+=1'''
		
		for key,value in object_dict.items():
			main_dict[str(response_count)] = object_dict[key]
			response_count+=1
		
		for key,value in object_dict_10_classes.items():
			main_dict[str(response_count)] = object_dict_10_classes[key]
			response_count+=1
			
		print("Main Answer ",main_dict)
		
				
		return jsonify(main_dict)


if __name__ == '__main__':
    app.run(host= '192.168.43.63', port=5000, debug=True)


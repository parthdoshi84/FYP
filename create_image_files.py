import numpy as np
import sys
import os

textfile = open("class_text_files/" + sys.argv[1], "r")
print "class_text"
argument = sys.argv[1].split(".")
image_id = []
class_description_file = open("../../classes/class-descriptions.txt","r")

encrypted_keys = []
annotations_file = open("annotations-human-bbox.csv","r")
print "annotation open"
classes = []
count  = 0
for line in textfile:
	temp = line.split(",")
	if temp[0] not in  image_id:
		image_id.append(temp[0])
		count = count + 1 
print "1 for done"


for line in class_description_file:
	modified = line.replace(r"\r\n","")
	temp = modified.split(",")
	encrypted_keys.append(temp[0])
	t = temp[1].replace(r'\r','')
	classes.append(t)
count  = 0

print "Last"

for line in annotations_file:
	temp = line.split(",")
	id_name = temp[0]
	if id_name in image_id:
		new_file = open("image_files/"+argument[0]+"/"+str(id_name)+ ".txt","a")
		encrypted_key = temp[2]
		for key in range(0,len(encrypted_keys)):
			if encrypted_keys[key] == encrypted_key:
				class_name = classes[key].replace("\n","")
				break
		new_line = class_name + "," + str(temp[4]) + "," + str(temp[5]) + "," + str(temp[6]) + ","+ str(temp[7])
		new_file.write(new_line)
		files_n = os.listdir("image_files/" + argument[0] + "/")
		if len(files_n) == 3001:
			break
		else:
			print len(files_n)		
		

			

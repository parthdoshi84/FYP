from PIL import Image
import numpy as np
import os
import urllib
import sys
files = os.listdir("image_files/"+sys.argv[1]+ "/")

classes = ["tree","book","building","food","footwear","mobile phone","door","window","bed","clothing"]

template = open("templates_for_annotation/template.xml","r")
object_template = open("templates_for_annotation/object_template.xml","r")
fails = open("../../../data_test/" + sys.argv[1]+ "/failed_urls.txt","a")
unOpened = open("../../../data_test/" + sys.argv[1] + "/unOpened_images.txt","a")

start_string = ""
object_string = ""
for line in template:
	start_string+=line
for line in object_template:
	object_string+=line

def get_image(id_name):
	file_csv = open("../../images/train/images.csv","r")
    	for line in file_csv:
        	image_id = line.split(",")[0]
        	if str(image_id) == str(id_name):
			#print image_id
			
            		url = line.split(",")[2]
            		try:
				file_jpg = urllib.URLopener()
	    			file_jpg.retrieve(url,"../../../data_test/"+ sys.argv[1] +"/images/"+str(image_id)+ ".jpg")
				return True
			except:
				fails.write(str(id_name)+".txt" + "\n")
				return False
			break
		
def create_annotations(id_name,file_name):
	#print "annotate function"
	annotation_file = open("../../../data_test/"+sys.argv[1]+"/annotations/"+ str(id_name) + ".xml", "a")
	custom_start_string = start_string.replace("<filename>1.jpg</filename>", "<filename>"+str(id_name)+".jpg" + "</filename>")
	try:
		img = Image.open("../../../data_test/"+sys.argv[1]+"/images/"+ str(id_name)+".jpg")
	except:
		print "Error in opening file"
		unOpened.write(str(id_name) + ".jpg" + "\n")
		return 
	width = img.size[0]
	height = img.size[1]
	custom_start_string = custom_start_string.replace("<width>500</width>","<width>"+str(width)+"</width>")
	custom_start_string = custom_start_string.replace("<height>375</height>", "<height>"+str(height)+"</height>")
	file_n = open("image_files/"+sys.argv[1]+"/"+ file_name,"r")
	for line in file_n:
		
		line_arr = line.split(",")
		class_name = line_arr[0].lower()
		if class_name == "woman" or class_name == "man":
			class_name = "person"
		if class_name == "bike":
			class_name = "motorbike"
		if class_name == "plant":
			class_name = "potted plant"
		#print type(class_name)
		for class_n in classes: 	
			if class_name == class_n:
				#print "Inside class"
				custom_object_string = object_string.replace("<name>person</name>","<name>"+str(class_name)+"</name>")
				custom_object_string = custom_object_string.replace("<xmin>135</xmin>","<xmin>"+str(int(float(line_arr[1])*width))+"</xmin>")
				custom_object_string = custom_object_string.replace("<ymin>25</ymin>","<ymin>"+str(int(float(line_arr[3])*height))+"</ymin>")
				custom_object_string = custom_object_string.replace("<xmax>236</xmax>","<xmax>"+str(int(float(line_arr[2])*width))+"</xmax>")
				custom_object_string = custom_object_string.replace("<ymax>188</ymax>","<ymax>"+str(int(float(line_arr[4].replace("\n",""))*height))+"</ymax>")				
				custom_start_string += custom_object_string
	custom_start_string += "</annotation>" 
	annotation_file.write(custom_start_string)
count  = 1
for file_name in files:
	id_name = file_name.split(".")[0]
    	print id_name
	
	if get_image(id_name) == True:
		create_annotations(id_name,file_name)	
	else:
		print "File = "+ str(count)+ " failed"
	print "Count = " + str(count) + " done"
	
	count = count + 1 

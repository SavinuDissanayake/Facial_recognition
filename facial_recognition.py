import cv2
import os
import numpy as np

####get the image that you want to test####
#test_image = cv2.imread(image.jpg)
#### and send that value a function that reconize the name for the face
#face_prediction = pridict(test_image)



### Making a function to train the program to identify the group of pictures of the same person that we provide ###
# def traing_data(training_data_folder_path):
#	faces = []     --- make a empty list to hold each training image of the person's face
#	labels = []    --- make a empty list to hold the label number of the specific person (that we are looking through the pictures of)
# 	dirs = os.listdirs(training_data_foler_path)          --- make a list of all the files in the training folder (we need a proper path address for the [os.listdirs()])

#	for dir_name in dirs:       --- we use a FOR loop to access each file individualy from the list of files we made earlier (list of files in the our training data folder)
#		label =  dir_name.replace("s","")   --- In the training folder for each individual we named their folder S1 and S2 and S3 and so on. but here we are getting rid of that 's' cuz in python we dont like mixed values

# 		subject_folder = training_data_folder_path + "\\" + dir_name          --- this is the adress for the each individual person's folder (so we can open up each person's folder)
#		subject_images = os.listdirs(subject_folder)         --- this is going to make a list for the images in each individual's folder
#		for image in subject_images:        --- we are making a FOR loop to access all of the pictures (of that person that person that we open the file of) in the list individually
#			face, rect = detect_face(image)    --- here we are calling out the [detect_face()] function to get just the face part of each training image of that person and also we get the measurement for the rectangle
#			if face, rect == None, None:   --- if there is no face just continue the program
#				continue
#			faces.append(face)   --- each face we get of the same person we going to add to the faces list (so we have a list of faces of the same person)
#			labels.append(label)  --- now for the label we just have a number cuz we got rid of the 's' earlier, so here we just add the label number for the person to the labels list
	####  THE REASON WHY I PUT THE [FACES.APPEND] AND [LABELS.APPEND] TOGETHER CUZ THEN WE HAVE A LABEL FOR EACH PICTURE THAT WE RAN (EVEN THOUGH ALL THE VALUES IN THE LABELS LIST GOING TO BE THE SAME)
	#### IT DOES MAKE SENCE WHY WE WOULD HAVE THE SAME LABEL VALUE WHEN WE RUN ALL THE PICTURES IN THE FOLDER BECAUSE THEY ARE PICTURES OF THE SAME PERSON

#	return faces, labels   --- return the faces list and the labels list of the person



####	pridict if there is a face and get just the face part in the image and the sizes for the rectangle size for the face####
# def dectect_face(image):
#	image_gray = cv2.cvtColor(image, COLOR_BGR2GRAY)
#	haar_face_detecter = cv2.CascadeClassifier("C:\\Users\\savin\\OneDrive\\Desktop\\python\\Lib\\site-packages\\cv2\\data\\lbpcascade_frontalface.xml") --- activating the haar_face_detecter

#	faces  = haar_face_detecter.detectMultiScale(image_gray, 1.3, 5) --- geting all the faces in the image

#	if len(faces) == 0:
# 		return None, None

#	(x,y,w,h) = face[0]
# 	return image_gray[y:y+h, x:x+w], face[0]




### make the function that going to determine the name for the face and display the output###
# def pridict_image(image, subject_names):
#	face,rect = detect_face(image)

###		use the face recognizer to identify the id value for each person ###
#	face_recognition_tuple_value = face_recognizer.pridict(face)

#	label = face_recognition_tuple_value[0] --- get the id value
#	subject = subject_names[label]

#	draw_rectangle(img,rect)                        --- drawing the rectangle around the face
#	put_text_label(img, subject,rect[0], rect[1])   --- label the person's name

# return image, subject  --- becuase in this function we got the test image and we predicted the name and we labeled it on the actual test image and also drew a box around the face in the actual test image
### so we did all of those changes to that test image and now we send the changed test image back (in the test image we drew a box and wrote something)



### create a function that will draw the box around the test image
# def draw_rectangle(image, rect): 
#	(x,y,w,h) = rect  --- extract the measurments for the rectangle,, so basically the [rect] looks like this ((1,2,4,6), (2,3,5,6)), so rather than extracting the values one at a time-- we just extracting it in one step
#	cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)   --- draw the actual rectangle



### create a function that going to label the test image
# def put_text(image, x, y)
#	cv2.putText(image, (x,y), the text, font family, font color, thickness)



### the main code that call run the program ###

# subject_names = ["name1", "name2", "name3"]   --- make a list of all the people's names that we need to know 

# faces, labels = training_data(path to the training folder)  --- run the [traing_data()] function to get the list of all the faces of the subject and the label list associated with these pictures(just one number whole time)

# face_recognizer = cv2.face.LBPHFaceRecognizer_create()   --- activate the [LBPHFaceRecognizer_create()]
# face_recognizer.train(faces, np.array(labels))    --- train the program, so that every time the picture has this person's face give them the same label value

# test_image = cv2.imread(test_image)   --- load the test image
# prediction, subject = predict_image(test_image, subject_names)  --- call out the [predict_image()] function to identify how is in the picture

# cv2.imshow(subject, prediction)   --- show the result of the prediction
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def detect_face(img):
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	haar_face_detecter = cv2.CascadeClassifier("C:\\Users\\savin\\OneDrive\\Desktop\\python\\Lib\\site-packages\\cv2\\data\\lbpcascade_frontalface.xml")
	face = haar_face_detecter.detectMultiScale(img_gray,1.3,5)
	if len(face) == 0:
		return None, None

	(x,y,w,h) = face[0]

	return img_gray[x:x+w, y:y+h], face[0]


def train_data(train_folder_path):
	faces = []
	labels = []

	dirs = os.listdir(train_folder_path)

	for dir_name in dirs:
		label = int(dir_name.replace("s", ""))

		subject_file_path = train_folder_path + "\\" + dir_name
		subject_image_list = os.listdir(subject_file_path)

		for subject_image in subject_image_list:

			if subject_image.startswith("."):
				continue

			subject_image_path = subject_file_path + "\\" + subject_image
			image = cv2.imread(subject_image_path)
			face, rect = detect_face(image)

			if face is None:
				continue
			else:
				faces.append(face)
				labels.append(label)
	return faces, labels



def draw_rectangle(image, measurement):
	img = image
	(x,y,w,h) = measurement
	cv2.rectangle(img, (x,y), (w+x, h+y), (0,255,0), 2)

	return img



def label_text(image, text, x, y):
	img = image
	cv2.putText(img,text, (x,y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2 )

	return img


def predict_face(test_image, subject_list):

	image = test_image.copy()
	face, rect = detect_face(image)

	face_recognizer_tuple = face_recognizer.predict(face)

	confidence = face_recognizer_tuple[1]
	if confidence < 80:
		label = face_recognizer_tuple[0]
		person_name  = subject_list[label]
	else:
		name = input("what is your name: ")
		person_name = name




	print(face_recognizer_tuple)
	
	draw_rectangle(image, rect)
	label_text(image, person_name,rect[0], rect[1])

	return image, person_name



subject_list = ["", "Bill Gates", "Steve Jobs"]
faces,labels = train_data("C:\\Users\\savin\\OneDrive\\Desktop\\py_projects\\Facial_recognition\\training_data")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()     
face_recognizer.train(faces, np.array(labels))
print(faces)
print(labels)
test_image = cv2.imread("shaq.jpg")
predict_img, person_name = predict_face(test_image, subject_list)

cv2.imshow(person_name,predict_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
import numpy as np
import json
import cv2
import sys
import time
import pickle
import openface
import os
import socket
import struct 
import matplotlib
import matplotlib.pyplot as plt
import collections

NEW_CLICK_FLAG = False 
MOUSE_LAST_CLICK_X_AND_Y = (0, 0)

def mouse_events(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		global NEW_CLICK_FLAG 
		global MOUSE_LAST_CLICK_X_AND_Y
		NEW_CLICK_FLAG = True
		MOUSE_LAST_CLICK_X_AND_Y = (x, y)

class person:
	def __init__(self, inf_json_file_name):
		file_base_name = os.path.basename(inf_json_file_name)
		self.name = os.path.splitext(file_base_name)[0]
		
		with open(inf_json_file_name) as data_file:    
			self.dict = json.load(data_file)

		self.infoTypes = list(self.dict.keys())

		self.personal_informations = self.dict['personal_informations']
		self.professional_informations = self.dict['professional_informations']
		self.academic_informations = self.dict['personal_informations']
		self.info_mode = 'main_menu'

class people:
	def __init__(self, files_dir):
		self.names = []
		self.people_dict = {}
		self.num_people = 0

		for file in os.listdir(files_dir):
			if file.endswith(".json"):
				json_file_name = os.path.join(files_dir, file)
				new_person = person(inf_json_file_name = json_file_name)
				
				self.names.append(new_person.name)
				self.people_dict[new_person.name] = new_person
				self.num_people += 1

	def get_person(self, person_name):
		ret_flag = False
		ret_val = 'Unknown Person'
		if person_name in self.people_dict.keys():
			ret_flag = True
			ret_val = self.people_dict[person_name]
		return [ret_flag, ret_val]

	def get_person_info_types_list(self, person_name):
		ret_flag = False
		ret_val = 'Unknown Person'
		
		if person_name in self.people_dict.keys():
			ret_flag = True
			ret_val = self.people_dict[person_name].infoTypes
		return [ret_flag, ret_val]

	def set_person_current_info_mode(self, person_name, new_info_mode):
		ret_flag = False
		if person_name in self.people_dict.keys():
			ret_flag = True
			self.people_dict[person_name].info_mode = new_info_mode
		return ret_flag

class EYE:
	def __init__(	self, casc_path, classifier_model_path, 
					align, net,
					people_json_dir, 
					classifier_img_length = 3,
					video_size_x = 480, video_size_y = 540,
					face_rectangle_padding = 25,
					classifier_face_img_dim = 96,
					time_search_face_found = 0.5, time_search_face_not_found = 0.25,
					search_padding_x = 50, search_padding_y = 50,
					face_cascade_scale_factor = 1.05, face_cascade_min_neighbors = 5,
					face_cascade_min_size = (30, 30), 
					face_cascade_flags = cv2.cv.CV_HAAR_SCALE_IMAGE,
					x_menu_padding = 5):

		self.video_size_x = video_size_x
		self.video_size_y = video_size_y

		self.video_capture = cv2.VideoCapture(0)
		self.face_cascade = cv2.CascadeClassifier(casc_path)

		#### CLASSIFIER ARGS
		with open(classifier_model_path, 'rb') as classifier_file_connection:
			(labels, classifier) = pickle.load(classifier_file_connection)
		self.align = align
		self.net = net
		self.classifier = classifier
		self.labels = labels
		self.classifier_face_img_dim = classifier_face_img_dim 


		self.frame_rate_per_second = 0  
		self.frame_count_in_second = 0
		self.last_update_time_frame_count = time.time()

		self.last_search_time = -1
		self.time_search_face_found = time_search_face_found
		self.time_search_face_not_found = time_search_face_not_found

		self.known_face = False
		self.face_name = 'Unknown'

		self.face_in_image = False
		self.face_in_image_min_x = 0
		self.face_in_image_max_x = video_size_x
		self.face_in_image_min_y = 0
		self.face_in_image_max_y = video_size_y

		self.search_region_min_x = 0
		self.search_region_max_x = video_size_x
		self.search_region_min_y = 0
		self.search_region_max_y = video_size_y

		self.search_padding_x = search_padding_x
		self.search_padding_y = search_padding_y

		self.face_rectangle_padding = face_rectangle_padding
		self.face_rectangle_min_x = -1
		self.face_rectangle_max_x = -1
		self.face_rectangle_min_y = -1
		self.face_rectangle_max_y = -1

		self.face_cascade_scale_factor = face_cascade_scale_factor
		self.face_cascade_min_neighbors = face_cascade_min_neighbors
		self.face_cascade_min_size = face_cascade_min_size
		self.face_cascade_flags = face_cascade_flags

		self.video_capture.set(3, video_size_x)
		self.video_capture.set(4, video_size_y)

		[self.ret , self.img_bgr] = self.video_capture.read()
		self.img_gray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)

		self.people_DB = people(people_json_dir)

		self.classifier_img_window = classifier_img_length * ['Unknown']

		self.x_menu_padding = x_menu_padding

		self.clickable_menu_item_list = []

	def draw_new_clickable_menu_item(	self,
										init_x, init_y, info,
										color = (0, 255, 0),
										clickable_rectangle_x = 20, 
										x_padding = 5, y_padding = 5,
										font_scale = 1, thickness = 1):
		y_size = y_padding
		max_x_size = 0
		infoList = str(info).split('_')

		for text in infoList:
			text = text.title()
			[[x, y], baseLine] = cv2.getTextSize(	text = text, 
													fontFace = cv2.FONT_HERSHEY_PLAIN,
													fontScale = font_scale, 
													thickness = thickness)
			y_size += y + y_padding
			### Writes menu_items
			cv2.putText(img = self.img_bgr, 
						text = text,
						org = (	init_x + clickable_rectangle_x + x_padding,
							 	init_y + y_size - y_padding),
						fontFace = cv2.FONT_HERSHEY_PLAIN,
						fontScale = 1, color = color, thickness = 1)

			if x > max_x_size :
				max_x_size = x

		max_x_size += init_x + clickable_rectangle_x + 2 * x_padding

		### Draws clickable rectangle
		cv2.rectangle(	self.img_bgr, 
						pt1 = (	init_x,
								init_y),
		 				pt2 = (	init_x + clickable_rectangle_x,
		 				 		init_y + y_size), 
		 				color = color,
		 				thickness = -1)

		cv2.rectangle(	self.img_bgr, 
						pt1 = (init_x + clickable_rectangle_x, init_y),
		 				pt2 = (max_x_size, init_y + y_size), 
		 				color = color,
		 				thickness = 1)

		return [init_x,
		 		init_y,
			 	init_x + clickable_rectangle_x,
		 		init_y + y_size]

	def draw_menu_title(self, menu_title_name):
		cv2.putText(img = self.img_bgr, text = menu_title_name,
					org = (	self.face_rectangle_max_x + self.x_menu_padding,
						 	self.face_rectangle_min_y),
					fontFace = cv2.FONT_HERSHEY_PLAIN,
					fontScale = 1.5, color = (0,255,0), thickness = 2)

	def draw_main_menu(self):
		self.draw_menu_title(menu_title_name = 'Main Menu')
		[ret_flag, info_types_list] = self.people_DB.get_person_info_types_list(person_name = self.face_name)
		x = self.face_rectangle_max_x + self.x_menu_padding
		y = self.face_rectangle_min_y

		for info_type in info_types_list :
			[init_x, init_y, end_x, end_y] = self.draw_new_clickable_menu_item( init_x = x,
				 																init_y = y,
				 																info = info_type )
			self.clickable_menu_item_list.append((info_type, init_x, init_y, end_x, end_y))
			y = end_y

	def draw_return_menu_item(self, x, y):
		[init_x, init_y, end_x, end_y] = self.draw_new_clickable_menu_item( init_x = x,
				 															init_y = y,
				 															info = 'Return' )
		self.clickable_menu_item_list.append(('main_menu', init_x, init_y, end_x, end_y))

	def draw_info_menu(self, info_name, info):
		formated_menu_name = ' '.join([ x.capitalize() for x in info_name.split('_') ])
		self.draw_menu_title(menu_title_name = formated_menu_name)

		x = self.face_rectangle_max_x + self.x_menu_padding
		y = self.face_rectangle_min_y

		for info_key in info.keys() :
			info_text = str(info_key) + ' : ' + info[str(info_key)]
			[init_x, init_y, end_x, end_y] = self.draw_new_clickable_menu_item( init_x = x,
				 																init_y = y,
				 																info = info_text,
				 																color = (0, 0, 255) 
				 																)
			y = end_y
		self.draw_return_menu_item(x = x, y = y)

	def draw_menu(self):
		if self.face_in_image:
			self.clickable_menu_item_list = []
			[ret_flag, person] = self.people_DB.get_person(person_name = self.face_name)

			if ret_flag :
				if person.info_mode in person.infoTypes:
					self.draw_info_menu( info_name = person.info_mode,
										 info = person.dict[str(person.info_mode)] )
				else :
					self.draw_main_menu()

	def detec_clicked_menu_item(self, x_clicked_pos, y_clicked_pos):
		for menu_item in self.clickable_menu_item_list:
			[info_type, init_x, init_y, end_x, end_y] = menu_item
			if ((x_clicked_pos >= init_x) and (x_clicked_pos <= end_x)) and ((y_clicked_pos >= init_y) and (y_clicked_pos <= end_y)):
				print 'info_type :' + str(info_type)
				self.people_DB.set_person_current_info_mode(person_name = self.face_name,
															new_info_mode = info_type)
		return 'None'

	def get_representatives(self):
		if self.face_in_image:
			face_img = self.img_bgr[self.search_region_min_y:self.search_region_max_y, self.search_region_min_x:self.search_region_max_x] 
			face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
			aligened_face_pre_process = self.align.getLargestFaceBoundingBox(face_img)
			aligned_face = self.align.align( self.classifier_face_img_dim,
										face_img, aligened_face_pre_process,
										landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE)
			if aligned_face == None :
				return None 
			else :
				rep = self.net.forward(aligned_face)
				return (aligened_face_pre_process.center(), rep)
		else :
			return None

	def update_face_classification(self, face_name):
		self.classifier_img_window.pop()
		self.classifier_img_window.insert(0, face_name)

		count = collections.Counter(self.classifier_img_window)
		self.face_name = str( count.most_common()[0][0] )

		if self.face_name == 'Unkown':
			self.known_face = False
		else : 
			self.known_face = True

	def classify_face(self):
		if self.face_in_image:
			rep = self.get_representatives()
			if rep:
				predictions = self.classifier.predict_proba(rep[1].reshape(1, -1)).ravel()
				maxI = np.argmax(predictions)
				person = self.labels.inverse_transform(maxI)
				confidence = predictions[maxI]
				self.update_face_classification(face_name = str(person) )

	def draw_face_name(self):
		if self.face_in_image:
			cv2.putText(img = self.img_bgr, text = self.face_name,
						org = (self.face_rectangle_min_x, self.face_rectangle_min_y),
						fontFace = cv2.FONT_HERSHEY_PLAIN,
						fontScale = 2, color = (0,255,0), thickness = 2)

	def update_frame_count(self):
		aux_time = time.time()
		self.frame_count_in_second += 1

		if (aux_time - self.last_update_time_frame_count) > 1.0:
			self.last_update_time_frame_count = aux_time
			self.frame_rate_per_second = self.frame_count_in_second
			self.frame_count_in_second = 1

		cv2.putText(img = self.img_bgr, text = str(self.frame_rate_per_second) + ' FPS',
					org = (5, 10),
					fontFace = cv2.FONT_HERSHEY_PLAIN,
					fontScale = 1, color = (255,255,255), thickness = 1)

	def get_lower_limit_x(self, original_x, padding_x):
		new_x = original_x - padding_x
		if new_x > 0 :
			return new_x
		else :
			return 0

	def get_lower_limit_y(self, original_y, padding_y):
		new_y = original_y - padding_y
		if new_y > 0 :
			return new_y
		else :
			return 0

	def get_upper_limit_x(self, original_x, padding_x):
		new_x = original_x + padding_x
		if new_x < self.video_size_x :
			return new_x
		else :
			return self.video_size_x

	def get_upper_limit_y(self, original_y, padding_y):
		new_y = original_y + padding_y
		if new_y < self.video_size_y :
			return new_y
		else :
			return self.video_size_y

	def set_face_rectangle(self):
		if self.face_in_image :
			self.face_rectangle_min_x = self.get_lower_limit_x(	original_x = self.face_in_image_min_x,
			 													padding_x = self.face_rectangle_padding)

			self.face_rectangle_max_x = self.get_upper_limit_x(	original_x = self.face_in_image_max_x,
			 													padding_x = self.face_rectangle_padding)

			self.face_rectangle_min_y = self.get_lower_limit_y(	original_y = self.face_in_image_min_y,
			 													padding_y = self.face_rectangle_padding)

			self.face_rectangle_max_y = self.get_upper_limit_y(	original_y = self.face_in_image_max_y,
			 													padding_y = self.face_rectangle_padding)
		else :
			self.face_rectangle_min_x = -1
			self.face_rectangle_max_x = -1
			self.face_rectangle_min_y = -1
			self.face_rectangle_max_y = -1


	def draw_face_rectangle(self):
		if self.face_in_image :
			cv2.rectangle(	img = self.img_bgr,
						 	pt1 = (self.face_rectangle_min_x, self.face_rectangle_min_y),
			 				pt2 = (self.face_rectangle_max_x, self.face_rectangle_max_y),
			 				color = (0, 255, 0),
			 				thickness = 2 )

	def set_face_not_found(self):
		self.face_in_image = False
		self.face_in_image_min_x = 0
		self.face_in_image_max_x = self.video_size_x
		self.face_in_image_min_y = 0
		self.face_in_image_max_y = self.video_size_y

		self.search_region_min_x = 0
		self.search_region_max_x = self.video_size_x
		self.search_region_min_y = 0
		self.search_region_max_y = self.video_size_y
		self.set_face_rectangle()

	def set_face_found(	self, 
						face_in_image_min_x, face_in_image_max_x,
	 					face_in_image_min_y, face_in_image_max_y):

		self.face_in_image = True
		self.face_in_image_min_x = face_in_image_min_x
		self.face_in_image_max_x = face_in_image_max_x
		self.face_in_image_min_y = face_in_image_min_y
		self.face_in_image_max_y = face_in_image_max_y

		self.search_region_min_x = self.get_lower_limit_x(	original_x = self.face_in_image_min_x, 
															padding_x = self.search_padding_x)

		self.search_region_max_x = self.get_upper_limit_x(	original_x = self.face_in_image_max_x, 
															padding_x = self.search_padding_x)

		self.search_region_min_y = self.get_lower_limit_y(	original_y = self.face_in_image_min_y, 
															padding_y = self.search_padding_y)

		self.search_region_max_y = self.get_upper_limit_y(	original_y = self.face_in_image_max_y, 
															padding_y = self.search_padding_y)

		self.set_face_rectangle()

	def get_bigger_face(self, faces):
		n_faces = faces.shape[0]
		
		if n_faces == 1 :
			return faces[0]

		face_area = 0
		bigger_face = faces[0]
		for face in faces:
			if (face[2] * face[3]) > face_area :
				bigger_face = face

		return bigger_face


	def check_for_face(self):

		diff_time = time.time() - self.last_search_time
		condition_01 = (diff_time > self.time_search_face_found) and (self.face_in_image)
		condition_02 = (diff_time > self.time_search_face_not_found) and (not self.face_in_image) 

		if condition_01 or condition_02 :
			self.last_search_time = time.time()
			faces = self.face_cascade.detectMultiScale( self.img_gray[self.search_region_min_y:self.search_region_max_y, self.search_region_min_x:self.search_region_max_x],
														scaleFactor = self.face_cascade_scale_factor,
														minNeighbors = self.face_cascade_min_neighbors,
														minSize = self.face_cascade_min_size,
														flags = self.face_cascade_flags )

			if type(faces).__module__ == np.__name__ :
				[face_in_image_min_x, face_in_image_min_y, face_padding_x, face_padding_y] = self.get_bigger_face(faces)
				
				face_in_image_min_x = self.search_region_min_x + face_in_image_min_x
				face_in_image_min_y = self.search_region_min_y + face_in_image_min_y
				face_in_image_max_x = face_in_image_min_x + face_padding_x
				face_in_image_max_y = face_in_image_min_y + face_padding_y

				self.set_face_found(face_in_image_min_x = face_in_image_min_x,
								 	face_in_image_max_x = face_in_image_max_x,
		 							face_in_image_min_y = face_in_image_min_y,
		 							face_in_image_max_y = face_in_image_max_y)

			else :
				self.set_face_not_found()

	def run(self):
		aux_X = 100
		aux_Y = 100

		global MOUSE_LAST_CLICK_X_AND_Y
		global NEW_CLICK_FLAG 
		
		cv2.namedWindow('EyE')
		cv2.setMouseCallback('EyE', mouse_events)

		fps = 15
		capSize = (self.img_bgr.shape[0], self.img_bgr.shape[1])
		fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
		#out = cv2.VideoWriter()
		#out.open('output.m4v', fourcc, fps, capSize, True) 


		print "Press \'q\' on the video window to exit!"
		while(True):
			[self.ret, self.img_bgr] = self.video_capture.read()
			self.img_gray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)

			self.check_for_face()
			self.classify_face()
			self.update_frame_count()
			self.draw_face_rectangle()
			self.draw_face_name()
			self.draw_menu()

			cv2.imshow('EyE', self.img_bgr)
			#out.write(self.img_bgr)

			if NEW_CLICK_FLAG :
				NEW_CLICK_FLAG = 0
				print 'New Click Detected at : ' + str(MOUSE_LAST_CLICK_X_AND_Y)
				[aux_X, aux_Y] = MOUSE_LAST_CLICK_X_AND_Y
				self.detec_clicked_menu_item(x_clicked_pos = aux_X, y_clicked_pos = aux_Y)
			
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		self.video_capture.release()
    	cv2.destroyAllWindows()

def main(	casc_path, classifier_model_path, align, net, people_json_dir):
	eye = EYE(	casc_path = casc_path, classifier_model_path = classifier_model_path,
				align = align, net = net, people_json_dir = people_json_dir)
	eye.run()

if __name__ == "__main__":

	people_json_dir = "./data/json"

	casc_path = "./util/haarcascade_frontalface_default.xml"
	
	classifier_model_path = "./generateSnaps/features/classifier.pkl"
	dlib_face_predictor_path = "./models/dlib/shape_predictor_68_face_landmarks.dat"
	open_face_model_dir = "./models/openface"
	network_model_path = "./models/openface/nn4.small2.v1.t7"


 	align = openface.AlignDlib(dlib_face_predictor_path)
 	net = openface.TorchNeuralNet(network_model_path, imgDim = 96)

	main(casc_path, classifier_model_path, align, net, people_json_dir)

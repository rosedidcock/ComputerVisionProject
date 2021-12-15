import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from os import listdir
from numpy import zeros
from numpy import asarray
from matplotlib.patches import Rectangle
from Mask_RCNN.mrcnn.utils import Dataset
from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from math import floor, ceil





##################################################### Defining the Dataset Class ##############################################################



class VideoFrameDataset(Dataset):
	# class that defines and loads the dataset


	def load_dataset(self, dataset_dir):
		""" Input: string dataset_dir containing the name of the directory containing annotation and frame data
			Loads the dataset definitions for the frame image data contained in that directory
		"""

		# define one class, classifying object vs background 
		self.add_class("dataset", 1, "object")

		images_dir = dataset_dir + '/zips/'
		annotations_dir = dataset_dir + '/anno/'


        # looping through the video folders in the zips folder
		for folder in listdir(images_dir):


			# reading in annotation data
			with open (annotations_dir+folder+'.txt', "r") as myfile:
				data=myfile.readlines()


			bboxes = [None for line in data]
			for (i,line) in enumerate(data):
				line = line.replace('\n', '')
				nums = line.split(',')
				nums_float = [floor(float(num)) for num in nums]

				xmin = nums_float[0]
				ymin = nums_float[1]
				xmax = xmin + nums_float[2]
				ymax = ymin + nums_float[3]

				bboxes[i] = [xmin, ymin, xmax, ymax]


			frames = [None for image in listdir(images_dir + folder)]
			for (i,filename) in enumerate(listdir(images_dir + folder)):
				im = Image.open(images_dir + folder+"/"+filename)
				im2arr = np.array(im)
				frames[i] = im2arr


			# generating predicited bounding boxes for testing data from frame differencing method
			if dataset_dir == "baby_test":
				bboxes = [bboxes[0]] + frameDifferencingMethod(frames,bboxes[0])
				

			for (i,frame) in enumerate(frames):
				# adding the image to the model dataset
				self.add_image('dataset', image_id=folder+str(i), path=frame, annotation=bboxes[i])


	# load the masks
	def load_mask(self, image_id):
		""" Using the image_id to extract corresponding frame and annotation info, loads a mask for the frame
		"""

		# get details of image
		info = self.image_info[image_id]
		# define box 
		box = info['annotation']
		image = info["path"]

		# ensuring bounding box doesn't go outside image boundaries
		if box[0] < 0:
			xdiff = 0-box[0]
			box[0] = 0
			box[2] += xdiff
		if box[1] < 0: 
			ydiff = 0-box[1]
			box[1] = 0
			box[3] += ydiff

		h,w,dims = image.shape

		# create one array for all masks, each on a different channel
		mask = zeros([h, w], dtype='uint8')
		# create masks
		class_ids = list()
		row_s, row_e = box[1], box[3]
		col_s, col_e = box[0], box[2]
		mask[row_s:row_e, col_s:col_e] = 1
		class_ids.append(self.class_names.index('object'))

		return mask, asarray(class_ids, dtype='int32')
 


	def load_image(self, image_id):
		# Loads the frame data in array form for a given image id

		info = self.image_info[image_id]
		return info['path']

	


##################################################### Defining the Model Configurations ##############################################################



class ObjConfig(Config):
	NAME = "obj_cfg"
	# (background + object)
	NUM_CLASSES = 1 + 1
	# Number of training steps per epoch
	STEPS_PER_EPOCH = 131
 

class PredictionConfig(Config):
	NAME = "obj_cfg"
	# (background + object)
	NUM_CLASSES = 1 + 1
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1



#################################################### Frame Differencing Method  ##############################################################




def frameDifferencingMethod(frames, annotation, plot=False):
	""" Inputs: a list containing frame data for a video, and a bounding box annotation for the first frame in the video
		Output: solution, a list containing the generated bounding box differences for the rest of the frames in the list
		Uses frame differences between 3 consecutive frames, followed by a closing, then ignores contours that are not within 20 pixels 
		of the previous frame's bounding box annotation. 
		Note: When plot is set to true, can be used for investigating the generated frame differencing bounding boxes
	"""

	solution = []

	for i in range(1, len(frames)): 

		if solution == []:
			try: 		
				xmin = annotation[0]
				ymin = annotation[1]
				xmax = annotation[2]
				ymax = annotation[3]
			except: print(annotation)
		else: 
			# reset new area to search in
			xmin = solution[-1][0]
			ymin = solution[-1][1]
			xmax = solution[-1][2]
			ymax = solution[-1][3]

		bbox = []


		frame = frames[i]
		prevFrame = frames[i - 1]
		try: nextFrame = frames[i + 1]
		except: nextFrame = frames[-1]


		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		prevFrame = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)
		nextFrame = cv2.cvtColor(nextFrame, cv2.COLOR_BGR2GRAY)

		# Frame Differencing 
		diff_frames_1 = cv2.absdiff(prevFrame, frame)
		diff_frames_2 = cv2.absdiff(frame, nextFrame)
		imgray = cv2.bitwise_or(diff_frames_1, diff_frames_2)


		try:
			# Adding thresholding, followed by a closing with kernel size 9
			ret, thresh = cv2.threshold(imgray, 30, 255, cv2.THRESH_BINARY)
			kernel = np.ones((9,9),np.uint8)
			thresh_morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

			mask = np.zeros(thresh_morph.shape,np.uint8)
			mask[max(0,ymin-20):min(thresh_morph.shape[0], ymax+20), max(0,xmin-20):min(thresh_morph.shape[1], xmax+20)] = thresh_morph[max(0,ymin-20):min(thresh_morph.shape[0], ymax+20), max(0,xmin-20):min(thresh_morph.shape[1], xmax+20)]

			contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


			if len(contours) != 0:
				c = max(contours, key = cv2.contourArea)                           # find the largest contour
				x,y,w,h = cv2.boundingRect(c)                                      # get bounding box of largest contour
				bbox = [floor(x), floor(y), ceil(x + w), ceil(y + h)]
			else:
				bbox = solution[-1]

			if plot == True:

				# plot the contour and bounding box pair and display every 60 frames

				if i in [1,60,120,180,240,300]:

					# Create figure and axes
					fig, ax = plt.subplots()
					# Display the image
					ax.imshow(mask, cmap = 'gray')
					# Create a Rectangle patch
					rect = Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='r', facecolor='none')
					# Add the patch to the Axes
					ax.add_patch(rect)

					plt.show()

		except:
			if solution == []:
				bbox = annotation
			else:
				bbox = solution[-1]

	
		solution += [bbox]


	return solution





##################################################### Plotting and Visualization Helper Functions ############################################################




def load_video_data(dataset_dir, folder_name, plot = False):
	""" Inputs: directory containing video folder zips and annotation, designated video folder to print from
		Output: the frames and bounding box annotations for the given video
		This helper function gives the bounding boxes from the given annotations in the training case, and the 
		predicited frame differencing method bounding boxes for the test data.
	"""

	images_dir = dataset_dir + '/zips/'
	annotations_dir = dataset_dir + '/anno/'


	with open (annotations_dir+folder_name+'.txt', "r") as myfile:
		data=myfile.readlines()


	bboxes = [None for line in data]

	for (i,line) in enumerate(data):
		line = line.replace('\n', '')
		nums = line.split(',')
		nums_float = [floor(float(num)) for num in nums]

		xmin = nums_float[0]
		ymin = nums_float[1]
		xmax = xmin + nums_float[2]
		ymax = ymin + nums_float[3]

		bboxes[i] = [xmin, ymin, xmax, ymax]


	frames = [None for image in listdir(images_dir + folder_name)]
	for (i,filename) in enumerate(listdir(images_dir + folder_name)):
		im = Image.open(images_dir + folder_name+"/"+filename)
		im2arr = np.array(im)
		frames[i] = im2arr


	if dataset_dir == "baby_test":
		if plot == True:
			bboxes = [bboxes[0]] + frameDifferencingMethod(frames,bboxes[0], True)
		else: 
			bboxes = [bboxes[0]] + frameDifferencingMethod(frames,bboxes[0])
				

	return frames, bboxes
        



def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):
	""" Inputs: dataset, model and configuration, as well as how many frames you want to see in the output image
		Output: Plot with actual and predicted bounding boxes displayed on n_images frames.
		Visual comparison between ground truth and output of prediciton model.
	"""

	seed = 2010
	step = 40
	k = 0

	for i in range(seed, seed+(step*n_images), step):

		# load the image and mask
		image = dataset.load_image(i)
		mask, _ = dataset.load_mask(i)
		mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		sample = np.expand_dims(scaled_image, 0)

		# make prediction
		yhat = model.detect(sample, verbose=0)[0]

		# Plotting actual bounding boxes
		plt.subplot(n_images, 2, k*2+1)
		plt.imshow(image)
		plt.title('Actual')

		# plot masks
		for j in range(mask.shape[2]):
			plt.imshow(mask[:, :, j], cmap='gray', alpha=0.3)


		# Plotting predicted bounding boxes from the model
		plt.subplot(n_images, 2, k*2+2)
		plt.imshow(image)
		plt.title('Predicted')
		ax = plt.gca()

		if len(yhat['rois']) >= 15:
			list =  yhat['rois'][:15]
		else: list = yhat['rois']


		p = 0
		for box in list:
			if p == 0:
				print(box)
			p+=1


			# get coordinates
			y1, x1, y2, x2 = box
			width, height = x2 - x1, y2 - y1

			# create the shape
			rect = Rectangle((x1, y1), width, height, fill=False, color='red')
			ax.add_patch(rect)

		k += 1


	plt.show()

	return





##################################################### Main Script: Creating and Training the Model #############################################################




def main():
	# called to define model, configuration, testing and training datasets

	data = VideoFrameDataset()
	data.load_dataset('baby')
	data.prepare()
	print('Train: %d' % len(data.image_ids))
	test_set = VideoFrameDataset()
	test_set.load_dataset('baby_test')
	test_set.prepare()
	print('Test: %d' % len(test_set.image_ids))



	################# UNCOMMMENT BELOW FOR TRAINING #########################
	
	
	# # create training config
	# config = ObjConfig()
	# config.display()

	# # define the model
	# model = MaskRCNN(mode='training', model_dir='./', config=config)


	# # load weights (mscoco)
	# model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

	# # train weights  ->  saves as .h5 file
	# model.train(data, test_set, learning_rate=config.LEARNING_RATE, epochs=3, layers='heads')
	# print("Done Training!")


	################### Model Inference Code ############################


	# create inference config
	cfg = PredictionConfig()
	# define the model
	model = MaskRCNN(mode='inference', model_dir='./', config=cfg)

	return model, data, test_set, cfg


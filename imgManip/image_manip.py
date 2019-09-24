import os, sys, cv2
import numpy as np

class ColorException(Exception):
    pass

class GradientFilters():
	def __init__(self, uint8=False, kernel_size=3):
		self.uint8 = uint8
		self.kernel_size = kernel_size

	def laPlacian(self, img_sample):
		laplacian = cv2.Laplacian(img_sample, cv2.CV_64F)

		if self.uint8:
			laplacian = np.uint8(np.absolute(laplacian))

		return img_sample + laplacian

	def sobel(self, img_sample):
		sobelx = cv2.Sobel(img_sample, cv2.CV_64F,1,0,ksize=self.kernel_size)
		sobely = cv2.Sobel(img_sample, cv2.CV_64F,0,1,ksize=self.kernel_size)

		if self.uint8:
			sobelx = np.uint8(np.absolute(sobelx))
			sobely = np.uint8(np.absolute(sobely))

		return img_sample + sobelx + sobely

	def scharr(self, img_sample):
		scharrx = cv2.Scharr(img_sample, cv2.CV_64F,1,0)
		scharry = cv2.Scharr(img_sample, cv2.CV_64F,0,1)

		if self.uint8:
			scharrx = np.uint8(np.absolute(scharrx))
			scharry = np.uint8(np.absolute(scharry))

		return img_sample + scharrx + scharry

class Thresholder():
	def __init__(self, th_type=cv2.THRESH_BINARY):
		self.th_type=th_type
		'''
		Possible types:
			- cv2.THRESH_BINARY
			- cv2.THRESH_BINARY_INV
			- cv2.THRESH_TRUNC
			- cv2.THRESH_TOZERO
			- cv2.THRESH_TOZERO_INV
		'''
	def otsu(self, img_sample):
		return cv2.threshold(img_sample, 0, 255, self.th_type + cv2.THRESH_OTSU)[1]

	def mean_c(self, img_sample):
		return cv2.adaptiveThreshold(img_sample, 255, cv2.ADAPTIVE_THRESH_MEAN_C, self.th_type, 15, 1)

	def gaussian_c(self, img_sample):
		return cv2.adaptiveThreshold(img_sample, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, self.th_type, 15, 1)

	def global_th(self, img_sample, threshold):
		return cv2.threshold(img_sample, threshold, 255, cv2.THRESH_BINARY)[1]

def gs2rgb(img_sample):
	if (len(img_sample.shape) > 2 ):
		raise ColorException("Grayscale image sample needed! Input: RGB")
	else:
		return cv2.cvtColor(img_sample, cv2.COLOR_GRAY2RGB)

def apply_mask(img, mask):
	return cv2.bitwise_and(img, img, mask=mask)

def grayscale_converter(input_folder = "./02_Glia_Images/", output_folder = "./Alternative_Datasets/02_Glia_Images_BW/"):
	if not os.path.isdir(output_folder):
		os.mkdir(output_folder)
		
	cont = 1
	for sample in os.listdir(input_folder):
		print(sample, ' #', cont)
		image = cv2.imread(input_folder + sample)
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		cv2.imwrite(output_folder + sample, gray_image)
		cont +=1

'''
def noise_reduction(input_folder, output_folder, output_mode='RGB'):
	if not os.path.isdir(output_folder):
		os.mkdir(output_folder)

	RGB = (output_mode == 'RGB')

	cont = 1
	for sample in os.listdir(input_folder):
		print(sample, ' #', cont)
		
		img_gs = cv2.imread(input_folder + sample, 0)

		if RGB:
			img_un = cv2.imread(input_folder + sample, -1)
			cv2.imwrite(output_folder + sample, cv2.bitwise_and(img_un, gs2rgb(toBinary(img_gs))))
		else:
			cv2.imwrite(output_folder + sample, cv2.bitwise_and(img_gs, toBinary(img_gs)))

		cont +=1 
'''

'''
This function resizes an image sample to a proportion of it

Parameters:
	- input_folder = directory containing all the image samples to be resized
	- output_folder = directory where the new image samples will be saved
	- proportion = proportion used to resize the image sample. Default = 0.1 (10% of its original size)

'''
def resize_by_prop(input_folder, output_folder, size=0.1):
	if not os.path.isdir(output_folder):
		os.mkdir(output_folder)

	cont = 1
	for sample in os.listdir(input_folder):
		print(sample, ' #', cont)

		img_un = cv2.imread(input_folder + sample, -1)
		cv2.imwrite(output_folder + sample, cv2.resize(img_un, None, fx=size, fy=size))
		
		cont +=1 
	
def resize(input_folder, output_folder, size=(128,96)):
	if not os.path.isdir(output_folder):
		os.mkdir(output_folder)

	cont = 1
	for sample in os.listdir(input_folder):
		print(sample, ' #', cont)

		img_un = cv2.imread(input_folder + sample, -1)
		cv2.imwrite(output_folder + sample, cv2.resize(img_un, size))
		
		cont +=1 

def split_by_img_prop(input_folder, output_folder, img_size=0.1, proportion=0.05):
	if not os.path.isdir(output_folder):
		os.mkdir(output_folder)

	count_img = 1
	for sample in os.listdir(input_folder):
		print(sample, ' #', count_img)

		img_un = cv2.imread(input_folder + sample, 0)
		img_rgb = cv2.imread(input_folder + sample, -1)
		otsu = glial_cells_get_background(img_un)
		height, width = img_un.shape

		wid_win_size = int(width * img_size)
		hei_win_size = int(height * img_size)

		count = 0
		for i in range(0, height, hei_win_size):
			y = i + hei_win_size
			for j in range(0, width, wid_win_size):
				x = j + wid_win_size
				name = sample.split('.')[0]

				# picks a portion of the threshold image
				otsu_crop = otsu[i:y, j:x]

				# if it's a valid crop
				if otsu_crop.shape[0] == hei_win_size and otsu_crop.shape[1] == wid_win_size:
					# count the number of non black pixels
					num_non_black_px = cv2.countNonZero(otsu_crop)
					# if it isnt a full black crop
					if  num_non_black_px > 0:
						proportion_black_px = num_non_black_px/(wid_win_size*hei_win_size)
						# if 95% of the image is composed of non-black pixels
						if proportion_black_px > proportion:
							cv2.imwrite("{0}{1}${2}.png".format(output_folder, name, (count)), img_rgb[i:y, j:x])
				
				count += 1
		
		count_img += 1
				

def split_by_prop(input_folder, output_folder, size=(64, 64), proportion=0.05):
	if not os.path.isdir(output_folder):
		os.mkdir(output_folder)

	count_img = 1
	for sample in os.listdir(input_folder):
		print(sample, ' #', count_img)

		img_un = cv2.imread(input_folder + sample, 0)
		img_rgb = cv2.imread(input_folder + sample, -1)
		otsu = glial_cells_get_background(img_un)
		height, width = img_un.shape

		wid_win_size = size[0]
		hei_win_size = size[1]

		count = 0
		for i in range(0, height, hei_win_size):
			y = i + hei_win_size
			for j in range(0, width, wid_win_size):
				x = j + wid_win_size
				name = sample.split('.')[0]

				# picks a portion of the threshold image
				otsu_crop = otsu[i:y, j:x]

				# if it's a valid crop
				if otsu_crop.shape[0] == hei_win_size and otsu_crop.shape[1] == wid_win_size:
					# count the number of non black pixels
					num_non_black_px = cv2.countNonZero(otsu_crop)
					# if it isnt a full black crop
					if  num_non_black_px > 0:
						proportion_black_px = num_non_black_px/(wid_win_size*hei_win_size)
						# if 95% of the image is composed of non-black pixels
						if proportion_black_px > proportion:
							cv2.imwrite("{0}{1}${2}.png".format(output_folder, name, (count)), img_rgb[i:y, j:x])
				
				count += 1
		
		count_img += 1

'''
This function pseudo color an grayscale image

Parameters:
	- input_folder = directory containing all the image samples to be pseudo colored
	- output_folder = directory where the new image samples will be saved
	- color_map = color map to be aplied to the samples. Options available:
		0 = cv2.COLORMAP_AUTUMN
		1 = cv2.COLORMAP_BONE
		2 = cv2.COLORMAP_JET
		3 = cv2.COLORMAP_WINTER
		4 = cv2.COLORMAP_RAINBOW
		5 = cv2.COLORMAP_OCEAN
		6 = cv2.COLORMAP_SUMMER
		7 = cv2.COLORMAP_SPRING
		8 = cv2.COLORMAP_COOL
		9 = cv2.COLORMAP_HSV
		10 = cv2.COLORMAP_PINK
		11 = cv2.COLORMAP_HOT
'''	
def pseudo_coloring(img, color_map=9):
	return cv2.applyColorMap(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), color_map)

def pseudo_coloring_path(input_folder, output_folder, color_map=9):
	if not os.path.isdir(output_folder):
		os.mkdir(output_folder)

	cont = 1
	for sample in os.listdir(input_folder):
		print(sample, ' #', cont)

		img_gs = cv2.imread(input_folder + sample, 0)
		cv2.imwrite(output_folder + sample, cv2.applyColorMap(img_gs, color_map))
		cont +=1 	

def glial_cells_get_background(img):
	th = Thresholder()
	gf = GradientFilters(uint8=True)

	return apply_mask(th.otsu(gf.sobel(img)), th.otsu(img))

def remove_background(img):
	return  apply_mask(img, glial_cells_get_background(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))

def general(input_folder, output_folder,kernel_sz=3):
	if not os.path.isdir(output_folder):
		os.mkdir(output_folder)
	'''
	th = Thresholder()
	gf = GradientFilters(uint8=True)

	y = [th.otsu, th.mean_c, th.gaussian_c]
	'''
	cont = 1
	for sample in os.listdir(input_folder):
		print(sample, ' #', cont)

		img_rgb = cv2.imread(input_folder + sample, -1)
		name = sample.split('.')[0]

		# Flips Image in 3 different ways
		cv2.imwrite('{0}/{1}$0.jpg'.format(output_folder, name), img_rgb)
		cv2.imwrite('{0}/{1}$1.jpg'.format(output_folder, name), cv2.flip(img_rgb, 1))
		cv2.imwrite('{0}/{1}$2.jpg'.format(output_folder, name), cv2.flip(img_rgb, 0))
		cv2.imwrite('{0}/{1}$3.jpg'.format(output_folder, name), cv2.flip( cv2.flip(img_rgb, 1), 0))

		'''
		otsu = th.otsu(img_sample)
		img_sample = cv2.imread(input_folder + sample, 0)
		cv2.imwrite(output_folder + sample, apply_mask(gf.sobel(img_rgb), glial_cells_get_background(img_sample)))
		cv2.imwrite(output_folder +'sobel_'+ sample, apply_mask(img_rgb, glial_cells_get_background(img_sample)))
		print(otsu.shape)
		'''
		cont +=1 	

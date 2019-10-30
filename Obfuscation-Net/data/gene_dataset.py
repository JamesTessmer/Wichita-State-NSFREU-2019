import torch
from torch.utils import data
import taxotree
import tensorflow as tf
import os
from keras.utils import to_categorical
import numpy as np
from PIL import Image
from random import shuffle
"""
Custom dataset made using the GeNet data, to be used with obfuscation net. Uses taxotree class from GeNet code to load genome data
"""
class gene_dataset(data.Dataset):
	'''
	Opens and reads the data files required, populates a list of labels and genomes
	'''
	def __init__(self, taxo_tree, labels, classes, training=False, transform = None):
		
		self.classes = classes
		self.labels = labels
		self.transform = transform
		self.taxo_tree=taxo_tree
		self.read_length = 1024 #number of pixels required
		
		#creating different lists for training vs testing set
		num_labels = self.taxo_tree.num_labels
		taxo_genomes = self.taxo_tree.genomes
	

		if(training):
			self.genomes = [None] * 900
			self.labels = [None] *900
			for i in range (0,900): #the first 900 of the genomes loaded
				self.genomes[i] = taxo_genomes[i]
				self.labels[i] = labels[i]
			#print(self.genomes)
		else:
			self.genomes = [None]*99
			self.labels = [None]*99
			counter = 0
			for i in range (901,1000): #the last 100
				self.genomes[counter] = taxo_genomes[i]
				self.labels[counter] = labels[i]
				counter += 1
			#print(self.genomes)
		

	def __len__(self):
		return len(self.genomes)
 
	def __getitem__(self, index):

		genome = self.genomes[index]
		
		#creating image that correlates to bp in genome
		pixel_dic = {	0: [255,0,0],
				1: [0,255,0],
				2: [0,0,255],
				3: [255,255,255]
				}
		pixels = np.zeros((32,32,3), dtype=np.uint8) #32x32 pixel image to work with ResNet18
		counter = 0
		
		for i in range(0,32):
			for p in range(0,32):
				pixels[i, p] = (pixel_dic[genome[counter]])
				counter += 1
			
		
		genome_img = Image.fromarray(pixels)
		#genome_img.show()
		#transforming to tensor along with any other desired transforms
		if self.transform is not None:
			genome_img=self.transform(genome_img)
		#return the genome image and the genus label
		#since the label is a path through the taxo tree we convert it to an integer, using the index of the label in the class list		
		path = self.labels[index]
		label = self.classes.index(path)
		
		return genome_img, label


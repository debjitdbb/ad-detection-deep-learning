import glob
import os

path = os.getcwd()+"\\labels\\"
labels = [f for f in glob.glob(path + "*.txt", recursive=True)]
for i in range(len(labels)):
	labels[i] = labels[i][22:-4]

path = os.getcwd()+"\\images\\"
images = [f for f in glob.glob(path + "*.png", recursive=True)]
images2 = [f for f in glob.glob(path + "*.jpg", recursive=True)]
images.append(images2)

for i in range(len(images)):
	images[i] = images[i][22:-4]

flag = 0
for i in range (len (labels)):
	# for j in range (len(labels)):
	if (labels[i] in images):
		flag = 1
	else: flag = 1
		# print(labels[i])
		

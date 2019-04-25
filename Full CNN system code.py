#importing libraries and functions
#begin imports
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras import backend as K
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import random
from IPython.display import display # to display images
#end imports

#Configuration Settings
#begin config
train_imgs_original = "crackforest-data/fixed_dataset_imgs_train.hdf5"
train_imgs_groudTruth = "crackforest-data/fixed_dataset_groundTruth_train.hdf5"
test_imgs_original = "crackforest-data/fixed_dataset_imgs_test.hdf5"
test_imgs_groudTruth = "crackforest-data/fixed_dataset_groundTruth_test.hdf5"
patch_height = 48
patch_width = 48
# total patches to extract
N_subimgs = 71000
# patches with cracks to extract eg. 60% of patches have cracks
patch_ratio = 0.6
batch_size = 32
#end config

#Get training and validation data and pre-process images
#begin training and validation data
def pred_to_imgs(pred, patch_height, patch_width, mode="original"):
	assert (len(pred.shape)==3)  #3D array: (Npatches,height*width,2)
	assert (pred.shape[2]==2 )  #check the classes are 2
	print(pred.shape)
	pred_images = np.empty((pred.shape[0],pred.shape[1]))  #(Npatches,height*width)
	if mode=="original":
		#for i in range(pred.shape[0]):
		#    for pix in range(pred.shape[1]):
		#        pred_images[i,pix]=pred[i,pix,1]
		pred_images[:]=pred[:,:,1]
	elif mode=="threshold":
		for i in range(pred.shape[0]):
			for pix in range(pred.shape[1]):
				if pred[i,pix,1]>=0.5:
					pred_images[i,pix]=1
				else:
					pred_images[i,pix]=0
	else:
		print("mode " +str(mode) +" not recognized, it can be 'original' or 'threshold'")
		exit()
	pred_images = np.reshape(pred_images,(pred_images.shape[0],1, patch_height, patch_width))
	return pred_images


def recompone_overlap(preds, img_h, img_w, stride_h, stride_w):
	assert (len(preds.shape)==4)  #4D arrays
	assert (preds.shape[1]==1 or preds.shape[1]==3)  #check the channel is 1 or 3
	patch_h = preds.shape[2]
	patch_w = preds.shape[3]
	N_patches_h = (img_h-patch_h)//stride_h+1
	N_patches_w = (img_w-patch_w)//stride_w+1
	N_patches_img = N_patches_h * N_patches_w
	print("N_patches_h: " +str(N_patches_h))
	print("N_patches_w: " +str(N_patches_w))
	print("N_patches_img: " +str(N_patches_img))
	assert (preds.shape[0]%N_patches_img==0)
	N_full_imgs = preds.shape[0]//N_patches_img
	print("According to the dimension inserted, there are " +str(N_full_imgs) +" full images (of " +str(img_h)+"x" +str(img_w) +" each)")
	full_prob = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))  #itialize to zero mega array with sum of Probabilities
	full_sum = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))

	k = 0 #iterator over all the patches
	for i in range(N_full_imgs):
		for h in range((img_h-patch_h)//stride_h+1):
			for w in range((img_w-patch_w)//stride_w+1):
				full_prob[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=preds[k]
				full_sum[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=1
				k+=1
	assert(k==preds.shape[0])
	assert(np.min(full_sum)>=1.0)  #at least one
	final_avg = full_prob/full_sum
	assert(np.max(final_avg)<=1.0) #max value for a pixel is 1.0
	assert(np.min(final_avg)>=0.0) #min value for a pixel is 0.0
	return final_avg



def visualize(data):
	assert (len(data.shape)==3) #height*width*channels
	img = None
	if data.shape[2]==1:  #in case it is black and white
		data = np.reshape(data,(data.shape[0],data.shape[1]))
	if np.max(data)>1:
		img = Image.fromarray(data.astype(np.uint8))   #the image is already 0-255
	else:
		img = Image.fromarray((data*255).astype(np.uint8))  #the image is between 0-1
	return img
#group a set of images row per columns
def group_images(data,per_row):
	assert data.shape[0]%per_row==0
	assert (data.shape[1]==1 or data.shape[1]==3)
	data = np.transpose(data,(0,2,3,1))  #corect format for imshow
	all_stripe = []
	for i in range(int(data.shape[0]/per_row)):
		stripe = data[i*per_row]
		for k in range(i*per_row+1, i*per_row+per_row):
			stripe = np.concatenate((stripe,data[k]),axis=1)
		all_stripe.append(stripe)
	totimg = all_stripe[0]
	for i in range(1,len(all_stripe)):
		totimg = np.concatenate((totimg,all_stripe[i]),axis=0)
	return totimg



#My pre processing (use for both training and testing!)
def my_PreProc(data):
	assert(len(data.shape)==4)
	assert (data.shape[1]==1)  #Use the original images
	#black-white conversion
	#train_imgs = rgb2gray(data)
	train_imgs = data
	#my preprocessing:
	train_imgs = dataset_normalized(train_imgs)
	train_imgs = clahe_equalized(train_imgs)
	train_imgs = adjust_gamma(train_imgs, 1.2)
	train_imgs = train_imgs/255.  #reduce to 0-1 range
	return train_imgs

#==== histogram equalization
def histo_equalized(imgs):
	assert (len(imgs.shape)==4)  #4D arrays
	assert (imgs.shape[1]==1)  #check the channel is 1
	imgs_equalized = np.empty(imgs.shape)
	for i in range(imgs.shape[0]):
		imgs_equalized[i,0] = cv2.equalizeHist(np.array(imgs[i,0], dtype = np.uint8))
	return imgs_equalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
#adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def clahe_equalized(imgs):
	assert (len(imgs.shape)==4)  #4D arrays
	assert (imgs.shape[1]==1)  #check the channel is 1
	#create a CLAHE object (Arguments are optional).
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	imgs_equalized = np.empty(imgs.shape)
	for i in range(imgs.shape[0]):
		imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
	return imgs_equalized


# ===== normalize over the dataset
def dataset_normalized(imgs):
	assert (len(imgs.shape)==4)  #4D arrays
	assert (imgs.shape[1]==1)  #check the channel is 1
	imgs_normalized = np.empty(imgs.shape)
	imgs_std = np.std(imgs)
	imgs_mean = np.mean(imgs)
	imgs_normalized = (imgs-imgs_mean)/imgs_std
	for i in range(imgs.shape[0]):
		imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
	return imgs_normalized


def adjust_gamma(imgs, gamma=1.0):
	assert (len(imgs.shape)==4)  #4D arrays
	assert (imgs.shape[1]==1)  #check the channel is 1
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	new_imgs = np.empty(imgs.shape)
	for i in range(imgs.shape[0]):
		new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
	return new_imgs

#data consinstency check
def data_consistency_check(imgs,masks):
	assert(len(imgs.shape)==len(masks.shape))
	assert(imgs.shape[0]==masks.shape[0])
	assert(imgs.shape[2]==masks.shape[2])
	assert(imgs.shape[3]==masks.shape[3])
	assert(masks.shape[1]==1)
	assert(imgs.shape[1]==1 or imgs.shape[1]==3)

def load_hdf5(infile):
    with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands
	       return f["image"][()]


def extract_random_cracks(full_imgs,full_masks, patch_h,patch_w, N_patches,patch_ratio,augmentation=False):
	if (N_patches%full_imgs.shape[0] != 0):
		print("N_patches: plase enter a multiple of 20")
		exit()
	assert (len(full_imgs.shape)==4 and len(full_masks.shape)==4)  #4D arrays
	assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
	assert (full_masks.shape[1]==1)   #masks only black and white
	assert (full_imgs.shape[2] == full_masks.shape[2] and full_imgs.shape[3] == full_masks.shape[3])
	patches = np.empty((N_patches,full_imgs.shape[1],patch_h,patch_w))
	patches_masks = np.empty((N_patches,full_masks.shape[1],patch_h,patch_w))
	img_h = full_imgs.shape[2]  #height of the full image
	img_w = full_imgs.shape[3] #width of the full image
	# (0,0) in the center of the image
	patch_per_img = int(N_patches/full_imgs.shape[0])  #N_patches equally divided in the full images
	print("patches per full image: " +str(patch_per_img))
	iter_tot = 0   #iter over the total numbe rof patches (N_patches)
	for i in range(full_imgs.shape[0]):  #loop over the full images

		if (full_masks[i] == 1).sum() == 0:
			continue
		k=0
		C=0
		Z=0
		while k < (patch_per_img):
			x_center = random.randint(0+int(patch_w/2),img_w-int(patch_w/2))
			y_center = random.randint(0+int(patch_h/2),img_h-int(patch_h/2))
			patch = full_imgs[i,:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
			patch_mask = full_masks[i,:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
			if (k < (patch_ratio*patch_per_img)):
				# patch
				if (np.sum(patch_mask)) >= 1:
					#if (np.sum(patch_middle)) >=1:
					patches[iter_tot]=patch
					patches_masks[iter_tot]=patch_mask
					iter_tot +=1   #total
					k+=1  #per full_img
					C+=1
			else:
				# no patch
				if (np.sum(patch_mask)) == 0:

					patches[iter_tot]=patch
					patches_masks[iter_tot]=patch_mask
					iter_tot +=1   #total
					k+=1  #per full_img
					Z+=1
	
	print("Patches with cracks: %s" %str(C))
	print("Patches without cracks: %s" %str(Z))


	return patches, patches_masks

def extract_ordered_overlap(full_imgs, patch_h, patch_w,stride_h,stride_w):
	assert (len(full_imgs.shape)==4)  #4D arrays
	assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
	img_h = full_imgs.shape[2]  #height of the full image
	img_w = full_imgs.shape[3] #width of the full image
	assert ((img_h-patch_h)%stride_h==0 and (img_w-patch_w)%stride_w==0)
	N_patches_img = ((img_h-patch_h)//stride_h+1)*((img_w-patch_w)//stride_w+1)  #// --> division between integers
	N_patches_tot = N_patches_img*full_imgs.shape[0]
	#print("Number of patches on h : " +str(((img_h-patch_h)//stride_h+1)))
	#print("Number of patches on w : " +str(((img_w-patch_w)//stride_w+1)))
	#print("number of patches per image: " +str(N_patches_img) +", totally for this dataset: " +str(N_patches_tot))
	patches = np.empty((N_patches_tot,full_imgs.shape[1],patch_h,patch_w))
	iter_tot = 0   #iter over the total number of patches (N_patches)
	for i in range(full_imgs.shape[0]):  #loop over the full images
		for h in range((img_h-patch_h)//stride_h+1):
			for w in range((img_w-patch_w)//stride_w+1):
				patch = full_imgs[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]
				patches[iter_tot]=patch
				iter_tot +=1   #total
	assert (iter_tot==N_patches_tot)
	return patches  #array with all the full_imgs divided in patches
#Extend the full images becasue patch divison is not exact
def paint_border(data,patch_h,patch_w):
	assert (len(data.shape)==4)  #4D arrays
	assert (data.shape[1]==1 or data.shape[1]==3)  #check the channel is 1 or 3
	img_h=data.shape[2]
	img_w=data.shape[3]
	new_img_h = 0
	new_img_w = 0
	if (img_h%patch_h)==0:
		new_img_h = img_h
	else:
		new_img_h = ((int(img_h)/int(patch_h))+1)*patch_h
	if (img_w%patch_w)==0:
		new_img_w = img_w
	else:
		new_img_w = ((int(img_w)/int(patch_w))+1)*patch_w
	new_data = np.zeros((data.shape[0],data.shape[1],new_img_h,new_img_w))
	new_data[:,:,0:img_h,0:img_w] = data[:,:,:,:]
	return new_data



def get_data_val(DRIVE_test_imgs_original, DRIVE_test_groudTruth, Imgs_to_test=42):
	test_imgs_original = load_hdf5(DRIVE_test_imgs_original)
	test_masks = load_hdf5(DRIVE_test_groudTruth)

	test_imgs = my_PreProc(test_imgs_original)
	test_masks = test_masks/255.

	#extend both images and masks so they can be divided exactly by the patches dimensions
	test_imgs = test_imgs[0:Imgs_to_test,:,:,:]
	test_masks = test_masks[0:Imgs_to_test,:,:,:]

	test_imgs = paint_border(test_imgs,patch_height,patch_width)
	test_masks = paint_border(test_masks,patch_height,patch_width)

	data_consistency_check(test_imgs, test_masks)

	#check masks are within 0-1
	assert(np.max(test_masks)==1  and np.min(test_masks)==0)

	print("\ntest images shape:")
	print(test_imgs.shape)
	print("\ntest mask shape:")
	print(test_masks.shape)
	print("test images range (min-max): " +str(np.min(test_imgs)) +' - '+str(np.max(test_imgs)))
	print("test masks are within 0-1\n")
	
	original_images = test_imgs
	original_masks = test_masks

	#extract the TEST patches from the full images
	patches_imgs_test = extract_ordered_overlap(test_imgs,patch_height,patch_width,patch_height/2,patch_width/2)
	patches_masks_test = extract_ordered_overlap(test_masks,patch_height,patch_width,patch_height/2,patch_width/2)


	print("\ntest PATCHES images shape:")
	print(patches_imgs_test.shape)
	print("test PATCHES images range (min-max): " +str(np.min(patches_imgs_test)) +' - '+str(np.max(patches_imgs_test)))

	return patches_imgs_test, patches_masks_test, original_images, original_masks

def get_data_training(train_imgs_original,train_imgs_groudTruth):
	train_imgs_original = load_hdf5(train_imgs_original)
	train_masks = load_hdf5(train_imgs_groudTruth) #masks always the same

	train_imgs = my_PreProc(train_imgs_original)
	train_masks = train_masks/255.

	train_imgs = train_imgs[:,:,9:574,:]  #cut bottom and top so now it is 565*565
	train_masks = train_masks[:,:,9:574,:]  #cut bottom and top so now it is 565*565
	data_consistency_check(train_imgs,train_masks)

	#check masks are within 0-1
	assert(np.min(train_masks)==0 and np.max(train_masks)==1)

	print "train images/masks shape:"
	print train_imgs.shape
	print "train images range (min-max): " +str(np.min(train_imgs)) +' - '+str(np.max(train_imgs))
	print "train masks are within 0-1"

	#extract the TRAINING patches from the full images
	patches_imgs_train, patches_masks_train = extract_random_cracks(train_imgs,train_masks,patch_height,patch_width,N_subimgs,patch_ratio)
	data_consistency_check(patches_imgs_train, patches_masks_train)

	print "\ntrain PATCHES images/masks shape:"
	print patches_imgs_train.shape
	print "train PATCHES images range (min-max): " +str(np.min(patches_imgs_train)) +' - '+str(np.max(patches_imgs_train))

	return patches_imgs_train, patches_masks_train

def prepare_masks(masks):
	assert (len(masks.shape)==4)  #4D arrays
	assert (masks.shape[1]==1 )  #check the channel is 1
	im_h = masks.shape[2]
	im_w = masks.shape[3]
	masks = np.reshape(masks,(masks.shape[0],im_h*im_w))
	new_masks = np.empty((masks.shape[0],im_h*im_w,2))
	for i in range(masks.shape[0]):
		new_masks[i,np.where(masks[i]==0),0] = 1
		new_masks[i,np.where(masks[i]==0),1] = 0
		new_masks[i,np.where(masks[i]==1),0] = 0
		new_masks[i,np.where(masks[i]==1),1] = 1

	return new_masks


imgs_train, masks_train = get_data_training(train_imgs_original,train_imgs_groudTruth)
imgs_test, masks_test, original_images, original_masks = get_data_val(test_imgs_original,test_imgs_groudTruth)

masks_train = prepare_masks(masks_train)
masks_test = prepare_masks(masks_test)
#end training and validation data

#Define the neural network
#begin CNN Structure
#CNN structure can be inserted between the above begin and "end CNN Structure"
def get_model(n_ch, patch_height, patch_width):

    inputs = Input(shape=(n_ch,patch_height,patch_width))
    
    LRUA = 0.0375
    # make it channels_last
    inputs_t = Lambda(lambda x: tf.transpose(x, (0,2,3,1)))(inputs)

    conv1 = Conv2D(32, (3, 3), activation='linear', padding='same')(inputs_t)
    
    conv1 = ELU(alpha=LRUA)(conv1)
    
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='linear', padding='same')(conv1)
    
    conv1 = ELU(alpha=LRUA)(conv1)
    
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='linear', padding='same')(pool1)
    
    conv2 = ELU(alpha=LRUA)(conv2)
    
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='linear', padding='same')(conv2)
    
    conv2 = ELU(alpha=LRUA)(conv2)

    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='linear', padding='same')(pool2)
    
    conv3 = ELU(alpha=LRUA)(conv3)
    
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='linear', padding='same')(conv3)
    
    conv3 = ELU(alpha=LRUA)(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2,up1],axis=3)

    conv4 = Conv2D(64, (3, 3), activation='linear', padding='same')(up1)
    
    conv4 = ELU(alpha=LRUA)(conv4)
    
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='linear', padding='same')(conv4)
    
    conv4 = ELU(alpha=LRUA)(conv4)

    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1,up2], axis=3)
    conv5 = Conv2D(32, (3, 3), activation='linear', padding='same')(up2)
    
    conv5 = ELU(alpha=LRUA)(conv5)
    
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='linear', padding='same')(conv5)
    
    conv5 = ELU(alpha=LRUA)(conv5)

    conv6 = Conv2D(2, (1, 1), activation='linear',padding='same')(conv5)
    
    conv6 = ELU(alpha=LRUA)(conv6)

    # due to channels last, reshape
    conv6 = core.Permute((3,1,2))(conv6)
    conv6 = core.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    conv7 = core.Activation('softmax')(conv6)       

    model = Model(input=inputs, output=conv7)

    model.compile(optimizer='sgd', loss='binary_crossentropy',metrics=['accuracy'])

    return model


# list of callbacks to include into model.fit
def get_callbacks():
    callbacks = []
    return callbacks
#End CNN structure
  
#carry out model training and output epoch information
#begin model training
n_ch = imgs_train.shape[1]
patch_height = imgs_train.shape[2]
patch_width = imgs_train.shape[3]
callbacks = get_callbacks()

model = get_model(n_ch, patch_height, patch_width)  
print("Check: final output of the network:")
print(model.output_shape)
model.fit(x=imgs_train,y=masks_train,epochs=100,batch_size=34,verbose=2,validation_data=(imgs_test,masks_test),callbacks=callbacks)
#end model training

#create system prediction
#begin prediction
predictions = model.predict(imgs_test, batch_size=32, verbose=2)
pred_patches = pred_to_imgs(predictions, patch_height, patch_width, "original")

# stitch predicted patches to images
predicted_masks = recompone_overlap(pred_patches,336,480,patch_height/2,patch_width/2)

#reshape images and masks to original size due to overlap
predicted_masks = predicted_masks[:,:,0:320,0:480]
original_images = original_images[:,:,0:320,0:480]
original_masks = original_masks[:,:,0:320,0:480]
#end prediction

#Confusion Matrix calculations
#begin confusion matrix calc
def conf_matrix(predictions, gt_masks, threshold_confusion, threshold_pixel):
    # threshold confusion --> when to count a pixel as a predicted crack
    # threshold pixel --> distance between a gt pixel and a predicted pixel, when
    # smaller it gets counted as a tp
    
    assert (predictions.shape == gt_masks.shape)  # check if shapes are the same

    # set all values over and under threshold 1 or 0
    predictions[predictions >= threshold_confusion] = 1
    predictions[predictions < threshold_confusion] = 0

    TP = 0
    FP = 0
    FN = 0

    height = predictions.shape[2]
    width = predictions.shape[3]

    for image in range(predictions.shape[0]):
        for y in range(height):
            for x in range(width):

                x_start = 0 if x-threshold_pixel <= 0 else x-threshold_pixel
                x_end = x+threshold_pixel+1 if x+threshold_pixel+1<width else width
                y_start = 0 if y-threshold_pixel <= 0 else y-threshold_pixel
                y_end = y+threshold_pixel+1 if y+threshold_pixel+1<width else width

                gt_window = gt_masks[image, 0, y_start:y_end, x_start:x_end]
                pred_window = predictions[image, 0, y_start:y_end, x_start:x_end]

                loc_pred = predictions[image, 0, y, x]
                if loc_pred == 1 and 1 in gt_window:
                    TP += 1
                if loc_pred == 1 and 1 not in gt_window:
                    FP += 1

                loc_gt = gt_masks[image, 0, y, x]
                if loc_gt == 1 and 1 not in pred_window:
                    FN += 1
    
    if TP != 0:
        TP = TP*1.0
        FN = FN*1.0
        
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f1 = (2*precision*recall)/(precision+recall)
    else:
        precision = 0
        recall = 0
        f1 = 0


    return precision, recall, f1


precision_2, recall_2, f1_2 = conf_matrix(predicted_masks,original_masks, 0.1, 2)

print(precision_2)
print(recall_2)
print(f1_2)
#end confusion matrix calc

#display images, raw, masks or predicted image
#begin display images
#e.g. display image 5 by using 4:5
display(visualize(group_images(original_images[6:7,:,:,:],1)))
display(visualize(group_images(original_masks[6:7,:,:,:],1)))
display(visualize(group_images(predicted_masks[6:7,:,:,:],1)))
display(visualize(group_images(original_images[5:6,:,:,:],1)))
display(visualize(group_images(original_masks[5:6,:,:,:],1)))
display(visualize(group_images(predicted_masks[5:6,:,:,:],1)))
display(visualize(group_images(original_images[9:10,:,:,:],1)))
display(visualize(group_images(original_masks[9:10,:,:,:],1)))
display(visualize(group_images(predicted_masks[9:10,:,:,:],1)))
#end display images

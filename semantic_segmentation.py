from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from skimage import io
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D
import os
from tensorflow.keras.applications.vgg16 import VGG16
import multiprocessing
import time
import pickle
from scipy import ndimage as nd
from skimage import exposure
from skimage import io
import gc

def fd_VGG16(img,model):
    input_image_stacked = np.expand_dims(img, axis=-1) 
               #model preparation
    
    #new_model.summary()
    #as it works only with 3 input channels: stack nuclear channel
    stacked_img = np.stack((img,)*3, axis=2)
    stacked_img = np.squeeze(stacked_img)
    stacked_img = stacked_img.astype(np.float32)
    stacked_img = stacked_img.reshape(-1, 1024, 1024, 3)
    
    features=model.predict(stacked_img)
    
    fv_VGG16= np.squeeze(features)
    return fv_VGG16

def init_VGG16():
    VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(1024, 1024, 3))

    #diable training (use pretrained weights)
    for layer in VGG_model.layers:
        layer.trainable = False


    #only use up to last layer where input size is still 1024x1024
    new_model = Model(inputs=VGG_model.input, outputs=VGG_model.get_layer('block1_conv2').output)
    VGG_model = []
    return new_model


#https://github.com/87surendra/Random-Forest-Image-Classification-using-Python/blob/master/Random-Forest-Image-Classification-using-Python.ipynb
#https://github.com/bnsreenu/python_for_microscopists/blob/master/062-066-ML_06_04_TRAIN_ML_segmentation_All_filters_RForest.py

def fd_blur(image):
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(image,-1,kernel)
    #print(np.shape(dst))
    dst = np.expand_dims(dst,axis=2)
    return dst


def fd_gabor(image):
    #Generate Gabor features
    num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
    kernels = []
    
    for theta in range(2):   #Define number of thetas
        theta = theta / 4. * np.pi
        for sigma in (1, 3):  #Sigma with 1 and 3
            for lamda in np.arange(0, np.pi, np.pi / 4):   #Range of wavelengths
                for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5
                    gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
    #                print(gabor_label)
                    ksize=9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                    kernels.append(kernel)
                    #Now filter the image and add values to a new column 
                    fimg = cv2.filter2D(image, cv2.CV_8UC3, kernel)
                    fimg = np.expand_dims(fimg, axis=2)
                    if num == 1:
                        filtered_img = fimg#fimg.reshape(-1)
                    else:
                        filtered_img = np.concatenate((filtered_img,fimg),axis=2)
                    num += 1  #Increment for gabor column label
    return filtered_img

def fd_filter_collection(img):
    
        #CANNY EDGE
        #TODO: change min/max
    edges = cv2.Canny(img.astype(np.uint8), 100,200)   #Image, min and max values
    edges = np.expand_dims(edges, axis=2)
    
    from skimage.filters import roberts, sobel, scharr, prewitt

    #ROBERTS EDGE
    edge_roberts = roberts(img)
    edge_roberts = np.expand_dims(edge_roberts, axis=2)

    #SOBEL
    edge_sobel = sobel(img)
    edge_sobel = np.expand_dims(edge_sobel, axis=2)

    #SCHARR
    edge_scharr = scharr(img)
    edge_scharr = np.expand_dims(edge_scharr, axis=2)

    #PREWITT
    edge_prewitt = prewitt(img)
    edge_prewitt = np.expand_dims(edge_prewitt, axis=2)

    #GAUSSIAN with sigma=3
    from scipy import ndimage as nd
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img = np.expand_dims(gaussian_img, axis=2)

    #GAUSSIAN with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img2 = np.expand_dims(gaussian_img2, axis=2)

    #MEDIAN with sigma=3
    median_img = nd.median_filter(img, size=3)
    median_img = np.expand_dims(median_img, axis=2)

    #VARIANCE with size=3
    variance_img = nd.generic_filter(img, np.var, size=3)
    variance_img = np.expand_dims(variance_img, axis=2)
    filters = [edges,edge_roberts,edge_sobel,edge_scharr,edge_prewitt,gaussian_img,gaussian_img2,median_img,variance_img]
    filters_stacked = np.stack(filters,axis=2)
    return filters_stacked
    
def annotations_to_tensor(feature_matrix,mask):
    #feature matrix dim: [x,y,nb_features]
    #possible mask elements: NaN: not annotated, int[0,1]: class annotation
    y_labels=[] #where class labels are stored
    X_features=[] #where feature vectors are stored
    for x,y in np.argwhere(~np.isnan(mask)):
        y_labels.append(mask[x,y])
        X_features.append(feature_matrix[x,y,:])
    #turn list into np array
    X_features = np.asarray(X_features)
    return X_features,y_labels

def extract_features(input_image,model_VGG16):
    #extract features from an image, and stack them
    pixel_value = np.expand_dims(input_image, axis=2)
    fv_blur = fd_blur(input_image)
    fv_gabor = fd_gabor(input_image)
    fv_filter_collection = fd_filter_collection(input_image)
    fv_filter_collection = np.squeeze(fv_filter_collection)
    fv_VGG16 = fd_VGG16(input_image, model_VGG16)
    global_feature = np.concatenate([pixel_value, fv_blur,fv_gabor,fv_filter_collection,fv_VGG16],axis=2)
    
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #rescaled_features = scaler.fit_transform(global_features)
    #print("shape of feature layer: "+str(np.shape(global_feature)))
    return global_feature


def apply_clf(args):
    path, clf = args[0],args[1]
    frames = np.load(path)
    output_stack = []
    model_VGG16 = init_VGG16()
    for frame_nb in (range(np.shape(frames)[0])):
        frame = frames[frame_nb,:,:]
        feature_image = extract_features(frame,model_VGG16)
        feature_vector = feature_image.reshape(np.shape(feature_image)[0]*np.shape(feature_image)[1],np.shape(feature_image)[2])
        prediction_vector = clf.predict_proba(feature_vector)[:,0] #[:,] to only use the prob. of the first class
        prediction_image = np.reshape(prediction_vector, (1024, 1024))
        output_stack.append(prediction_image)
        
        feature_image = []
        feature_vector = []
        prediction_vector = []
        prediction_image = []
        gc.collect()
        
    output_stack = np.array(output_stack)
    output = output_stack*255
    output = output.astype('uint8')
    
    return output




def apply_semantic_segmentation(chunk_paths, model):
    arg_stack = []

    #Create pool before loading dataset to avoid memory leaks
    pool = multiprocessing.Pool(processes= len(chunk_paths))
    time_start = time.time()
    for chunk_path in chunk_paths:
        arg_stack.append([chunk_path,model])

    print("Start workers ...")
    #delete the tiff file from memory as child processes will have copy of memory of parent:
    #https://stackoverflow.com/questions/49429368/how-to-solve-memory-issues-problems-while-multiprocessing-using-pool-map
    #tiff_orig = []


    results = pool.imap(apply_clf, arg_stack, chunksize = 1)
    pool.close() # No more work
    print("Wait for completion ...")
    pool.join()  # Wait for completion

    # Extract the results from the iterator object and store as tiff file.
    # The objects are automatically removed from the iterator once they are parsed - don't forget to store them!
    stack = []
    for result in results:
        stack.append(result)
    stack_array = np.array(stack)
    stack_array = np.concatenate(stack_array,axis=0)
    stack_array = stack_array.squeeze()
    
    nb_frames = np.shape(stack_array)[0]
    time_stop = time.time()
    time_total =  time_stop - time_start
    print("Semantic segmentation done. Processing time per frame: "+str(round(time_total/nb_frames, 2) )+" seconds. Total time: "+str(round(time_total/60,2))+" minutes.") 
    return stack_array
    

def interface(input_image, classifier, alpha):
    drawing = False # true if mouse is pressed
    ix,iy = -1,-1
    mode = True
    cv2.destroyAllWindows() #close any windows left 
    #convert 16bit-1channel input image into grayscale uint8 3channel image
    #rescale the intensity so we can see more
    p2, p98 = np.percentile(input_image, (2, 98))
    img8 = exposure.rescale_intensity(input_image, in_range=(p2, p98))
    scaler = MinMaxScaler(feature_range=(0, 255))
    scaler.fit(img8)
    img8 = scaler.transform(img8)
    img8 = np.round(img8)
    img8 = img8.astype('uint8')
    img8 = np.stack((img8,)*3, axis=-1)

   
    
    def draw_circle(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix,iy = x,y
            if mode == True:
                cv2.circle(output,(x,y),2,(255,100,100),-1)
                cv2.circle(mask_one_channel,(x,y),2,(0))
            else:
                cv2.circle(output,(x,y),2,(100,255,100),-1)
                cv2.circle(mask_one_channel,(x,y),2,(1))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    background = img8
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)
    cv2.imshow('image',background)
    output = background.astype('uint8')
    dest = np.empty(np.shape(background),np.uint8)
    dest = dest.astype('uint8')
    update = False
    model_VGG16 = init_VGG16()
    features = extract_features(input_image,model_VGG16)
    reuse_mask = False
    if reuse_mask:
        no_mask_initialized = False
        
    else:
        no_mask_initialized = True
        mask = np.empty(np.shape(background), np.uint8)
        mask_one_channel = np.zeros(np.shape(background)[:2])
        mask_one_channel.fill(np.nan)
        output = background.copy()
        

    no_prediction_initialized = True    
    
    print('\r' + 'Click to label [BACKGROUND]', end='')
    while(1):
        mask3 =  np.stack((mask_one_channel,)*3, axis=-1)
        mask3_isnan = np.isnan(mask3)

        if update:
            print('\r' + '[TRAINING CLASSIFIER]          ', end='')
            X,y = annotations_to_tensor(features,mask_one_channel)
            clf = classifier
            clf.fit(X, y)
            to_predict = features.reshape(np.shape(features)[0]*np.shape(features)[1],np.shape(features)[2])
            predicted = clf.predict_proba(to_predict)[:,0] #use [:,0] if displaying probab.    
            predicted = np.reshape(predicted, (1024, 1024))
            predicted = (predicted*255).astype('uint8')
            prediction3 = cv2.applyColorMap(predicted, cv2.COLORMAP_JET)
            update = False

            no_prediction_initialized = False
        if no_prediction_initialized == True:
            cv2.imshow('image',output)
        else: 
            cv2.addWeighted(output, alpha, prediction3, 1 - alpha,0, dest)
            cv2.imshow('image',dest)
            if mode:
                print('\r' + 'Click to label [BACKGROUND]', end='')
            else: 
                print('\r' + 'Click to label [  NUCLEI  ]', end='')
                
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
            if mode:
                print('\r' + 'Click to label [BACKGROUND]', end='')
            else: 
                print('\r' + 'Click to label [  NUCLEI  ]', end='')
        elif k == ord('q'):
            break
        elif k == ord('u'): #update
            update = True       

    cv2.destroyAllWindows()
    labels = mask_one_channel 
    return clf, mask_one_channel, output, background, prediction3 #return classifier and manual annotations
import cv2
import gc
import glob
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pickle
import tensorflow as tf
import time
from scipy import ndimage as nd
from skimage import exposure
from skimage import io
from skimage import io
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential, Model



#https://github.com/87surendra/Random-Forest-Image-Classification-using-Python/blob/master/Random-Forest-Image-Classification-using-Python.ipynb
#https://github.com/bnsreenu/python_for_microscopists/blob/master/062-066-ML_06_04_TRAIN_ML_segmentation_All_filters_RForest.py


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



def apply_clf(args):
    path, clf = args[0],args[1]
    frames = np.load(path)
    output_stack = []
    
    
    shapes = [(1024,1024),(512,512),(256,256)]
    models = init_VGG16_pyramid(shapes)
    
    for frame_nb in (range(np.shape(frames)[0])):
        frame = frames[frame_nb,:,:]
        features = fd_VGG16_pyramid(frame,models,shapes)
        to_predict = features.reshape(np.shape(features)[0]*np.shape(features)[1],np.shape(features)[2])
        
        prediction_vector = clf.predict_proba(to_predict)[:,0] #[:,] to only use the prob. of the first class
        prediction_image = np.reshape(prediction_vector, (1024, 1024))
        output_stack.append(prediction_image)
        
        feature_image = []
        feature_vector = []
        prediction_vector = []
        prediction_image = []
        
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


def fd_VGG16_scaled(img,model,shape = (1024, 1024)):
    
    #resize image to new shape  
    input_image_stacked = np.expand_dims(img, axis=-1)
    if shape != (1024, 1024):
        img = tf.image.resize(input_image_stacked,shape)
    #new_model.summary()
    #as it works only with 3 input channels: stack nuclear channel
    stacked_img = np.stack((img,)*3, axis=2)
    stacked_img = np.squeeze(stacked_img)
    stacked_img = stacked_img.astype(np.float32)
    
    stacked_img = stacked_img.reshape(-1, shape[0], shape[1], 3)
    
    #predict class in keras for each pixel
    features=model.predict(stacked_img)
    
    #remove extra dim
    fv_VGG16= np.squeeze(features)
    
    #scale up to match original img size
    #fv_VGG16 = resize(fv_VGG16,(1024,1024))
    if shape!= (1024, 1024):
        fv_VGG16 = tf.image.resize(fv_VGG16,(1024,1024))
    return fv_VGG16


def fd_VGG16_pyramid(img,models,shapes):
    #img - input image to calculate vgg response of
    #models - list of all vgg16 models 
    #shapes - corresponding shapes
    
    fv_list = []
    for model,shape in zip(models,shapes):
        fv = fd_VGG16_scaled(img,model,shape)
        fv_list.append(fv)
    
    global_feature = np.concatenate(fv_list,axis=2)
    return global_feature    
    
    

def init_VGG16_pyramid(input_shapes=[(1024, 1024)]):
    models = []
    for shape in input_shapes:
        keras_shape = (shape[0],shape[1],3) #add color channel
        VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=keras_shape)
        #VGG_model.summary()

        #disable training (use pretrained weights)
        for layer in VGG_model.layers:
            layer.trainable = False

        #only use up to last layer where input size is still 1024x1024
        new_model = Model(inputs=VGG_model.input, outputs=VGG_model.get_layer('block1_conv2').output)
        models.append(new_model)
    return models


def interface(input_image, classifier, alpha,mask = None):
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

    
    shapes = [(1024,1024),(512,512),(256,256)]
    models = init_VGG16_pyramid(shapes)
    features = fd_VGG16_pyramid(input_image,models,shapes)
    
    radius = 2
    
    print(features.shape)
    def draw_circle(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix,iy = x,y
            if mode == True:
                cv2.circle(output,(x,y),radius,(255,100,100),-1)
                cv2.circle(mask_one_channel,(x,y),radius,(0))
            else:
                cv2.circle(output,(x,y),radius,(100,255,100),-1)
                cv2.circle(mask_one_channel,(x,y),radius,(1))
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
    
    
    shape = (1024,1024)
    #model_VGG16 = init_VGG16_scaled(shape)
    #features = extract_features(input_image,model_VGG16)
    #features = fd_VGG16_scaled(input_image,model_VGG16,shape)
    if type(mask) != type(None):
        no_mask_initialized = False
        mask_one_channel = mask
        
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
        elif k == ord('u'):
            update = True   
        elif k in list(map(ord,list(map(str,range(0,10))))):
            #change radius of pen to number key clicked
            radius = int(chr(k))      
            

    cv2.destroyAllWindows()
    labels = mask_one_channel 
    return clf, mask_one_channel, output, background, prediction3 #return classifier and manual annotations
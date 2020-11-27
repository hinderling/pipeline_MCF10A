# input is label_stack
# create cytosolic ring
# extract features such as center of mass, probability, area
# return table with features and image stack with rings

import tqdm
import numpy as np
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.measurements import labeled_comprehension
import pandas as pd
from skimage.segmentation import watershed, expand_labels

def extract_ring(labels):
    #safety margin: 3
    #expansion: 7
    
    labels_expanded_3  = expand_labels(labels, distance=3)
    labels_expanded_13 = expand_labels(labels, distance=10)
    labels_ring = np.zeros(np.shape(labels))
    labels_ring_stack = []
    
    for label in range(1,np.max(labels)+1): #iterate through each nucleus, starting from index 1 as label 0 is background
        mask_nucleus_expanded_3 = labels_expanded_3 == label #boolean mask where nucleus is
        mask_nucleus_expanded_13 = labels_expanded_13 == label #boolean mask where extended region is
        mask_ring = np.logical_and(mask_nucleus_expanded_13, ~mask_nucleus_expanded_3) #the ring is where the nucleus is NOT
        labels_ring[mask_ring]=label
        
    return labels_ring.astype(int)


def apply_feature_extraction(instance_stack,tiff_stack,p_map,channels):
    
    labels_ring_stack = []
    results = pd.DataFrame()
    for frame_nb in tqdm.tqdm(range(np.shape(instance_stack)[0])):
        
        p_map_frame = p_map[frame_nb,:,:]
        tiff_frame = tiff_stack[frame_nb,:,:,:]
        instance_frame = instance_stack[frame_nb,:,:]
        nb_centers = np.max(instance_frame)
        
        to_zip = []     #all the features get appended here
        to_col = [] #all the names    get appended here
        
        
        ## EXTRACT FEATURES ##
        
        ## Store frame number
        frame_number_rep = np.repeat([frame_nb], nb_centers)
        
        to_zip.append(frame_number_rep)    
        to_col.append("frame")
        
        
        ## Nuclear centers
        p_map_frame_binary = p_map_frame<(255/2)
        centers = center_of_mass(p_map_frame_binary, labels = instance_frame, index = range(1, nb_centers+1))
        centers_x = np.array(centers)[:,1] #X/Y are inverted, take care!
        centers_y = np.array(centers)[:,0] #     ... are they?? TODO
        
        to_zip.append(centers_x)
        to_zip.append(centers_y)
        to_col.append("x")
        to_col.append("y")
        
        
        ## Mean certainty of nucleus segmentation
        mean_p_nuc = labeled_comprehension(p_map_frame, labels = instance_frame, index = range(1, nb_centers+1), func = np.mean, out_dtype = 'float32', default = float("nan"))
        
        to_zip.append(mean_p_nuc)
        to_col.append("mean_p_nuc")
        
        
        ## Create cytosolic ring
        labels_ring = extract_ring(instance_frame)
        
        
        ## Calculate areas of nucleus and ring
        size_nuc = labeled_comprehension(p_map_frame_binary, labels = instance_frame, index = range(1, nb_centers+1), func = np.sum, out_dtype = 'float32', default = float("nan"))
        
        ring_binary = np.zeros_like(labels_ring)
        ring_binary[labels_ring>0] = 1
        size_ring = labeled_comprehension(ring_binary, labels = labels_ring, index = range(1, nb_centers+1), func = np.sum, out_dtype = 'float32', default = float("nan"))
       
        to_zip.append(size_nuc)
        to_zip.append(centers_x)
        to_col.append("size_nuc")
        to_col.append("size_ring")
    
    
        ## Extract features from additional channels:
        channels_other = {k:v for (k,v) in channels.items() if k!="H2B"} #for all channels that are not H2B
        
        for k,v in channels_other.items():
            #mean intensity of nucleus
            mean_nuc  = labeled_comprehension(tiff_frame[:,:,v], labels = instance_frame, index = range(1, nb_centers+1), func = np.mean, out_dtype = 'float32', default = float("nan"))
            #mean intensity of cytosolic ring
            mean_ring = labeled_comprehension(tiff_frame[:,:,v], labels = labels_ring, index = range(1, nb_centers+1), func = np.mean, out_dtype = 'float32', default = float("nan"))
            #their ratio
            ratio = (mean_nuc / mean_ring).astype('float32')
            
            to_zip.append(mean_nuc)
            to_zip.append(mean_ring)
            to_zip.append(ratio)    
            to_col.append("mean_nuc_" + k)
            to_col.append("mean_ring_"+ k)
            to_col.append("ratio_"+ k)
            

        
        ## STORE FEATURES TO PANDAS DF, ONE ROW / NUCLEUS
        #features = list(zip(centers_x,centers_y,nuclear_area,ring_area,frame_number_rep, range(1, nb_centers+1),
                         #   mean_nucleus_c1, mean_ring_c1, ratio_c1, 
                         #  mean_p_nucleus))
                
        to_zip.append(range(1, nb_centers+1))
        to_col.append("label_frame")        
        
        features = list(zip(*to_zip))
        
        results = results.append(features)  #assign return value as it is a PD
        labels_ring_stack.append(labels_ring) #dont assign return value as it is a list
    
    
         
    
    #columns=['y','x','size_nuc','size_ring','frame','label_frame',
    #       'mean_nuc_c1','mean_ring_c1','ratio_c1',
    #       'mean_p_nuc']
    
    results.columns = to_col
    #results = results.convert_dtypes() #convert to "best" dtype
    
    # downsample precision form 64bit to 32bit
#    results = results.astype({'y':'float32','x':'float32','size_nuc':'float32','size_ring':'float32','frame':'float32', 'label_frame':'float32',
#            'mean_nuc_c1':'float32','mean_ring_c1':'float32','ratio_c1':'float32',
#            'mean_p_nuc':'float32'})
   
    results.astype('float32')
    
    labels_ring_stack = np.asarray(labels_ring_stack)
    
    return results, labels_ring_stack
    
    
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


def apply_feature_extraction(instance_stack,tiff_stack,p_map):
    columns=['y','x','size','frame', 'label_frame',
            'mean_nuc_c1','mean_ring_c1','ratio_c1',
            'mean_nuc_c2','mean_ring_c2','ratio_c2',
            'mean_p_nuc']
    labels_ring_stack = []
    results = pd.DataFrame()
    for frame_nb in tqdm.tqdm(range(np.shape(instance_stack)[0])):
        p_map_frame = p_map[frame_nb,:,:]
        tiff_frame = tiff_stack[frame_nb,:,:,:]
        instance_frame = instance_stack[frame_nb,:,:]
        nb_centers = np.max(instance_frame)

        ## EXTRACT FEATURES ##
        p_map_frame_binary = p_map_frame<(255/2)
        centers = center_of_mass(p_map_frame_binary, labels = instance_frame, index = range(1, nb_centers+1))
        centers_x = np.array(centers)[:,1] #X/Y are inverted, take care!
        centers_y = np.array(centers)[:,0]
        nuclear_area = labeled_comprehension(p_map_frame_binary, labels = instance_frame, index = range(1, nb_centers+1), func = np.sum, out_dtype = 'float32', default = float("nan"))
        mean_p_nucleus = labeled_comprehension(p_map_frame, labels = instance_frame, index = range(1, nb_centers+1), func = np.mean, out_dtype = 'float32', default = float("nan"))
        
        
        # Create cytosolic rings, calculate ratios  
        # First channel
        labels_ring = extract_ring(instance_frame)
        mean_nucleus_c1 = labeled_comprehension(tiff_frame[:,:,1], labels = instance_frame, index = range(1, nb_centers+1), func = np.mean, out_dtype = 'float32', default = float("nan"))
        mean_ring_c1 = labeled_comprehension(tiff_frame[:,:,1], labels = labels_ring, index = range(1, nb_centers+1), func = np.mean, out_dtype = 'float32', default = float("nan"))
        ratio_c1 = (mean_nucleus_c1 / mean_ring_c1).astype('float32')

        # Second channel
        mean_nucleus_c2 = labeled_comprehension(tiff_frame[:,:,2], labels = instance_frame, index = range(1, nb_centers+1), func = np.mean, out_dtype = 'float32', default = float("nan"))
        mean_ring_c2 = labeled_comprehension(tiff_frame[:,:,2], labels = labels_ring, index = range(1, nb_centers+1), func = np.mean, out_dtype = 'float32', default = float("nan"))
        ratio_c2 = (mean_nucleus_c2 / mean_ring_c2).astype('float32')
        frame_number_rep = np.repeat([frame_nb], nb_centers)

        ## STORE FEATURES TO PANDAS DF, ONE ROW / NUCLEUS
        features = list(zip(centers_x,centers_y,nuclear_area,frame_number_rep, range(1, nb_centers+1),
                            mean_nucleus_c1, mean_ring_c1, ratio_c1, 
                            mean_nucleus_c2, mean_ring_c2, ratio_c2,
                            mean_p_nucleus))
        
        results = results.append(features)  #assign return value as it is a PD
        labels_ring_stack.append(labels_ring) #dont assign return value as it is a list
        
    results.columns = columns
    results = results.convert_dtypes() #convert to "best" dtype
    
    # downsample precision form 64bit to 32bit
    results = results.astype({'y':'float32','x':'float32','size':'int32','frame':'int32', 'label_frame':'int32',
            'mean_nuc_c1':'float32','mean_ring_c1':'float32','ratio_c1':'float32',
            'mean_nuc_c2':'float32','mean_ring_c2':'float32','ratio_c2':'float32',
            'mean_p_nuc':'float32'})
    
    labels_ring_stack = np.asarray(labels_ring_stack)
    
    return results, labels_ring_stack
    
    
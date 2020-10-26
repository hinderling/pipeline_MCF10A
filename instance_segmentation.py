import skimage.morphology
import scipy.ndimage as ndi
from skimage import io
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from scipy.ndimage.morphology import binary_fill_holes, binary_erosion
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, expand_labels
import numpy as np
import multiprocessing
import time

def watershed_edt(background_mask):
    # fill holes
    background_mask = binary_fill_holes(background_mask)

    # calculate euclidean distance transform 
    distance = ndi.distance_transform_edt(background_mask)

    
    #generate the footprint for the non-max supression
    circle = skimage.morphology.disk(10)
    
    
    #get local maxi as a list of coordinates
    local_maxi = peak_local_max(distance, min_distance = 0)

    #coordinates to image with maxi as pixels
    peak_mask = np.zeros_like(distance, dtype=bool)
    for x,y in local_maxi:
            peak_mask[x,y] = True

    #label each maxi differently
    markers,nb_features = ndi.label(peak_mask)

    #watershed from seeds
    labels = watershed(image = -distance, markers = markers, mask=background_mask )#,compactness = 100)
    return labels,distance


def watershed_erosion_edt(background_mask, erosion_mask):
    # calculate euclidean distance transform 
    distance = ndi.distance_transform_edt(erosion_mask)

    #get local maxi as a list of coordinates
    local_maxi = peak_local_max(distance, min_distance = 0)
    
    #coordinates to image with maxi as pixels
    peak_mask = np.zeros_like(distance, dtype=bool)
    for x,y in local_maxi:
            peak_mask[x,y] = True

    #label each maxi differently
    markers,nb_features = ndi.label(peak_mask)

    #watershed from seeds
    labels = watershed(image = -distance, markers = markers, mask=background_mask )#,compactness = 100)
    return labels,distance


def recursion(binary_mask, binary_mask_selected, labels, erosion_iteration, min_size, max_size):
    #apply erosion(iteration)
    
    if erosion_iteration == 0:
        binary_mask_eroded = binary_mask_selected
    else:
        binary_mask_eroded = binary_erosion(binary_mask_selected,iterations=erosion_iteration)
        

    #do watershed
    labels_eroded,_ = watershed_erosion_edt(binary_mask_selected,binary_mask_eroded)
    unique, counts = np.unique(labels_eroded, return_counts=True)
    
    #check if there is a nucleus left, if not return last iteration
    if len(unique)==1:
        return labels
    
    #remove 0 label
    unique = unique[1:]
    counts = counts[1:]
    
    #for all elements/labels
    for u,c in zip(unique,counts):
        if c < min_size:
            #to small? return labels from binary_mask
            # do nothin
            continue
        if (c >= min_size and c < max_size):
            #ok size? return labels from binary_mask_erosion
            #update label map
            labels[labels_eroded==u] = np.max(labels)+1
        else:
            #to large? recursion(binary_mask, erosion+1)
            if erosion_iteration == 15: #increase this value to try to segment even further
                labels[labels_eroded==u] = np.max(labels)+1
            #extract just that blob
            else:
                binary_mask_selected = labels_eroded==u
                labels = recursion(binary_mask, binary_mask_selected, labels, erosion_iteration+1, min_size, max_size)
    return labels


def apply_recursion(args):#, raw_stack):
    chunk_path, min_size, max_size = args[0],args[1],args[2]
    segmentation_stack = np.load(chunk_path)
    completed_frames = []
    for frame_nb in range(np.shape(segmentation_stack)[0]):
        mask_f = segmentation_stack[frame_nb,:,:]
     
        #preprocess
        mask_binary_f = mask_f<(255/2)
        mask_binary_f = binary_fill_holes(mask_binary_f)
        labels,distance = watershed_edt(mask_binary_f)
        labels = np.zeros_like(mask_binary_f, dtype = np.uint16)

        #process
        labels = recursion(mask_binary_f, mask_binary_f, labels, 0, min_size, max_size)
        
        #postprocess
        labels = watershed(image = distance, markers = labels, mask=mask_binary_f )
        completed_frames.append(labels)
    completed_frames = np.array(completed_frames)    
    return completed_frames


def apply_instance_segmentation(chunk_paths, min_size, max_size):
    arg_stack = []
    pool = multiprocessing.Pool(processes= len(chunk_paths))
    time_start = time.time()
    for chunk_path in chunk_paths:
        arg_stack.append([chunk_path,min_size,max_size])

    print("Start workers ...")
    results = pool.imap(apply_recursion, arg_stack, chunksize = 1)
    pool.close() # No more work
    print("Wait for completion ...")
    pool.join()  # Wait for completion
    stack = []
    for result in results:
        stack.append(result)
    stack_array = np.array(stack)
    stack_array = np.concatenate(stack_array,axis=0)
    stack_array = stack_array.squeeze()
    
    nb_frames = np.shape(stack_array)[0]
    time_stop = time.time()
    time_total =  time_stop - time_start
    print("Done. Processing time per frame: "+str(round(time_total/nb_frames, 2) )+" seconds. Total time: "+str(round(time_total/60,2))+" minutes.") 
    return stack_array


def apply_instance_segmentation_old(segmented_stack_path, max_size, min_size,nb_cores):
    nuclear_segmentation = segmented_stack
    nb_frames = np.shape(nuclear_segmentation)[0]-1 #last frame index
    borders = np.round(np.linspace(0,nb_frames+1,nb_cores+1))
    stack_indexes=[]
    arg_stack = []
    nuclear_segmentation = []
    pool = multiprocessing.Pool(processes= nb_cores)

    nuclear_segmentation = io.imread(segmented_stack_path)

    time_start = time.time()
    print("Splitting of dataset: ")
    for c in range(nb_cores):
        stack_indexes.append((int(borders[c]),int(borders[c+1]-1)))
        start,end = ((int(borders[c]),int(borders[c+1])))
        #split_stacks.append(tiff_orig[start:end,:,:,0]) 
        #clf_stack.append(model)
        arg_stack.append(nuclear_segmentation[start:end,:,:].copy()) #.copy() creates a new object in memory instead of a view of the full matrix
        print("Core "+str(c)+"/"+str(nb_frames)+": "+str(end-start)+" frames.")
    print(stack_indexes)

    print("Start workers ...")
    #delete the tiff file from memory as child processes will have copy of memory of parent:
    #https://stackoverflow.com/questions/49429368/how-to-solve-memory-issues-problems-while-multiprocessing-using-pool-map
    nuclear_segmentation = []


    results = pool.imap(mpp.apply_recursion, arg_stack, chunksize = 1)
    pool.close() # No more work
    print("Wait for completion ...")
    pool.join()  # Wait for completion

    time_stop = time.time()
    time_total =  time_stop - time_start
    print("Done. Processing time per frame: "+str(time_total/nb_frames)+" seconds. Total time: "+str(time_total/60)+" minutes.")



    # Extract the results from the iterator object and store as tiff file.
    # The objects are automatically removed from the iterator once they are parsed - don't forget to store them!
    stack = []
    for result in results:
        stack.append(result)
    stack_array = np.array(stack)
    stack_array = np.concatenate(stack_array,axis=0)
    stack_array = stack_array.squeeze()
    io.imsave(output_path+'output_instance_segmentation/'+'series_1_Original'+'_morph_post_proc'+'.tiff',stack_array)
    
    return stack_labeled, table_untracked
import cv2
import gc
import math
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import numpy as np
import pandas as pd
import pickle
import scipy.ndimage as ndi
import skimage.morphology
import time
import time

from anytree import Node, RenderTree, LevelOrderGroupIter
from itertools import chain
from scipy.ndimage import label
from scipy.ndimage.measurements import labeled_comprehension
from scipy.ndimage.morphology import binary_fill_holes, binary_erosion
from skimage import exposure
from skimage import io
from skimage.color import label2rgb
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed, expand_labels
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler 


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
    #do it once with the watershed-line mask
    labels_line = watershed(image = -distance, markers = markers, mask=background_mask,watershed_line = True )#,compactness = 100)
    #do it again to remove the line
    #labels = watershed(image = -distance, markers = labels_line, mask=binary )#,compactness = 100)
  
    #labels_line = watershed(image = -distance, markers = markers, mask=binary,watershed_line = True )#,compactness = 100)
    labels_mask = labels_line!=0
    return labels_line,distance,labels_mask

def form_factor(area,perimeter):
    return (math.pi*4*area)/(perimeter**2)

def cost_f(node):
    cost_area = (abs(expected_size-(node.area))/expected_size)
    cost_ff = (max(1-node.formfactor,0))
    return cost_area + cost_ff

def group_cost_f(nodes):
    costs = list(map(cost_f,nodes))
    cost = np.mean(costs)
    return cost

def my_print_node(node):
    for pre, fill, n in RenderTree(node):
            print(f"{pre}ID: {n.name}, A:{int(n.area)}, F: {n.formfactor:.2f}, C:{cost_f(n):.2f}")
    print('\n')

def flatten_list(mixed):
    flat = []
    for elem in mixed:
        if type(elem)==type(list()):
            for node in elem:
                flat.append(node)
        else:
            flat.append(elem)
    return flat
    
def my_unique(lists):
    ret = np.unique(lists)
    return ret
    
#traverse tree
#return id with lowest costs
def traverse_tree(node):
    cost = cost_f(node)
    nb_children = len(node.children)
    if nb_children > 1:
        best_children = []
        for child in node.children:
            best_children.append(traverse_tree(child))
        #turn list of lists into a list
        best_children = flatten_list(best_children)
        
        #print(f'my c:{cost:.2f}; child c:{group_cost_f(best_children):.2f}')
        if (group_cost_f(best_children))<cost:
            #my_print_node(node)
            return best_children
        else:
            return node
    if nb_children == 1:
        if node.area == node.children[0].area:
            return traverse_tree(node.children[0])
        else: 
            return node
    if nb_children == 0:
        return node


def erode(input_image, erosion_levels, min_size,manual_expected_size):
    ######
    #- calculate erosion maps
    #- for all nuclei calculate area and determine unique ID
    #- now link all nuclei in trees for all largest nuclei 
    #- find node with lowest cost for each tree
    #- create map with all lowest costs

    #print(list(erosion_levels))
    binary = input_image/255<0.5
    lvl_labels = [] #image with labels
    lvl_ids = []
    lvl_areas = []
    lvl_contours = []
    lvl_nodes = []
    lvl_children = []
    max_id = 0

    binary = input_image/255<0.5
    binary = binary_fill_holes(binary)
    mask = binary
    for lvl in erosion_levels:
        if lvl == 0:
            er = binary
        else:
            er = (binary_erosion(binary,iterations=lvl))
        remove_small_objects(er, min_size=min_size,in_place =True)

        lb,distance,mask = watershed_erosion_edt(mask,er)
        lb_zero = lb==0
        lb_unique = lb + max_id
        lb_unique[lb_zero]=0
        lvl_labels.append(lb_unique)
        ids = np.unique(lb_unique)
        ids = ids[1:] #remove 0 label
        nb_labels = len(ids)
        if(nb_labels == 0):
            lvl_areas.append([])
        else:
            lvl_areas.append(labeled_comprehension(binary, lb, range(1,nb_labels+1), func = np.sum, out_dtype = float, default = 0))

        lvl_ids.append(ids)
        max_id = max_id+ nb_labels

    #calculate contour for all detected cells      
    for lvl_i in range(len(erosion_levels)):
        labels_i = lvl_labels[lvl_i]
        contours  = []
        for j,id_j in enumerate(lvl_ids[lvl_i]):
            mask_j = labels_i == id_j
            cnt = cv2.findContours(np.uint8(mask_j), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
            arclen = cv2.arcLength(cnt[0], True)
            contours.append(arclen)
        lvl_contours.append(contours)

    #find children for all (future) nodes
    for lvl_i in range(len(erosion_levels)-1):
        if(len(lvl_ids[lvl_i])==0):
            lvl_children.append([])
        else:
            c_lbls = labeled_comprehension(lvl_labels[lvl_i+1],lvl_labels[lvl_i],lvl_ids[lvl_i], out_dtype = list,func = my_unique, default = 0)
            lvl_children.append(c_lbls)

    #create actual nodes
    for lvl_i in range(len(erosion_levels)):
        nodes = []
        for j,id_j in enumerate(lvl_ids[lvl_i]):
            area = lvl_areas[lvl_i][j]
            contour = lvl_contours[lvl_i][j]
            ff = form_factor(area,contour)
            node = Node(id_j,area = area, contour = contour,formfactor = ff)
            nodes.append(node)
        lvl_nodes.append(nodes)
        
    #link nodes
    for lvl_i in range(len(erosion_levels)-1):
        for j,id_j in enumerate(lvl_ids[lvl_i]):
            children_ids = lvl_children[lvl_i][j]
            parent_node = lvl_nodes[lvl_i][j]
            children_node_indexes = np.where(np.isin(lvl_ids[lvl_i+1],children_ids))
            children_node_indexes = children_node_indexes[0].astype('int')
            children_nodes = np.take(lvl_nodes[lvl_i+1],children_node_indexes)
            parent_node.children = children_nodes
        global expected_size
        if manual_expected_size == 0:
            expected_size = np.median(lvl_areas[0])
        else:
            expected_size = manual_expected_size
        
    ids_final_img = []
    final_nodes = []
    for node in lvl_nodes[0]:
        ids = traverse_tree(node)
        if type(ids)==type(list()):
            for node in ids:
                final_nodes.append(node)
        else:
            final_nodes.append(node)
    ids_final_img = final_nodes            
    ids_final_img = [node.name for node in ids_final_img]
    final_labels = np.zeros((1024,1024))
    for labels in lvl_labels:
        indices_to_take = np.where(np.isin(labels,ids_final_img))
        final_labels[indices_to_take] = labels[indices_to_take]
        
    labels,_ = label(final_labels)#,compactness = 100)
    labels = watershed(image = binary, markers = labels, mask=binary,watershed_line = False )#,compactness = 100)
    return labels

def apply_erosion(args):
    #read in arguments
    chunk_path, erosion_levels, min_size, expected_size = args[0],args[1],args[2],args[3]
    #load in stack chunk
    segmentation_stack = np.load(chunk_path)
    completed_frames = []
    for frame_nb in range(np.shape(segmentation_stack)[0]):
        mask_f = segmentation_stack[frame_nb,:,:]
     
        labels = erode(mask_f, erosion_levels, min_size, expected_size)
        
        completed_frames.append(labels)
        gc.collect()

        
    completed_frames = np.array(completed_frames)    
    return completed_frames

def apply_instance_segmentation(chunk_paths, erosion_levels = range(0,14,1), min_size=20,manual_expected_size=0):
    arg_stack = []
    pool = multiprocessing.Pool(processes= len(chunk_paths))
    time_start = time.time()
    for chunk_path in chunk_paths:
        arg_stack.append([chunk_path,erosion_levels, min_size, manual_expected_size])

    print("Start workers ...")
    results = pool.imap(apply_erosion, arg_stack, chunksize = 1)
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
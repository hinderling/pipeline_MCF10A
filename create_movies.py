import pandas as pd
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib
import copy
import multiprocessing
import time

def create_movie(data_table, tiff_file, variable_name, min_value, max_value, cmap = plt.cm.Spectral):
    """
    Create a colormovie, where the color is given by a variable in the data table.

    variable_name -- what variable should be mapped
    min_value -- everything lower is mapped to lowest value of cmap
    max_value -- everything higher is mapped to highest value of cmap
    cmap -- what cmap to use (default: plt.cm.Spectral)

    """ 
    ## CREATE COLOR MAPPER AND CHECK DATA
    palette = copy.copy(cmap)
    palette.set_bad(alpha = 0.0)
    norm = colors.Normalize(vmin=min_value,vmax = max_value, clip = True)
    mapper = matplotlib.cm.ScalarMappable(norm = norm, cmap = palette)           
    try:
        data_table['frame']
        data_table['label_frame']
        data_table[variable_name]
    except KeyError:
        raise ValueError("Data table must contain columns 'frame', 'particle' and "+ variable_name)


    ## PARSE STACK AND LOCATE WANTED VALUES
    img_col_stack = []
    first_frame = data_table['frame'].min()
    for frame_i in range(tiff_file.shape[0]):
        frame_nb = first_frame + frame_i
        img_labels = tiff_file[frame_i,:,:]
        img_col = np.zeros(np.shape(img_labels))
        img_col[:] = np.nan #init empty array with nan

        for index, row in data_table[data_table['frame']==frame_nb].iterrows():
            #extract the variable for that nucleus and frame
            ratio,label = (row[variable_name], row['label_frame'])
            #fill all nucleus pixels with the value
            img_col[img_labels==label] = ratio

        img_col = mapper.to_rgba(img_col) #apply cmap
        img_col_stack.append(img_col)        
    img_col_stack = np.asarray(img_col_stack) #convert list to array
    img_col_stack = img_col_stack*255
    img_col_stack = img_col_stack.astype('uint8')
    return img_col_stack
    
def create_movie_from_path(args):
    """
    Helper function for create_movie(). Useful when using chunks in multiprocessing.
    Loads the np.array given in the chunk path, then calls create_movie().

    tiff_file_path -- path to a tiff file
    other args -- see create_movie()
    
    """     
    data_table = args[0]
    chunk_path = args[1] 
    variable_name = args[2]
    min_value = args[3]
    max_value = args[4]
    cmap = args[5]
    
    
    tiff_file = np.load(chunk_path)
    img_col_stack = create_movie(data_table, tiff_file, variable_name, min_value, max_value, cmap)
    return img_col_stack


def apply_create_movie(data_table, chunk_paths, variable_name, min_value, max_value, cmap = plt.cm.Spectral):
    """
    Helper function for multiprocessing for create_movie(). Takes a list of paths, pointing to stored np.arrays and runs a separate process for each of them. Gathers the results and stacks them.

    tiff_file_paths -- a list of paths to chunks of a tiff file
    other args -- see create_movie()
    
    """   
    arg_stack = []
    pool = multiprocessing.Pool(processes= len(chunk_paths))
    time_start = time.time()
    chunk_frame_start = 0
    for chunk_path in chunk_paths:
        chunk = np.load(chunk_path)
        nb_frames_in_chunk = chunk.shape[0]
        chunk_frame_stop = chunk_frame_start+nb_frames_in_chunk
        data_table_chunk = data_table[(data_table['frame'] >= chunk_frame_start) & (data_table['frame'] < chunk_frame_stop)]
        chunk_frame_start = chunk_frame_stop
        arg_stack.append([data_table_chunk,chunk_path,variable_name,min_value,max_value,cmap])

    print("Start workers ...")
    results = pool.imap(create_movie_from_path, arg_stack, chunksize = 1)
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
    
    
def rgba_to_rgb(rgba_img, col=(255, 255, 255)):
    """
    Replace all transparent pixels (alpha = 0) wit a solid color.
    see https://stackoverflow.com/q/9166400/13557501

    Keyword Arguments:
    rgba_img -- four-channel np.array, last dim are col channels
    col -- RGB values to fill in (default 255, 255, 255)

    """ 
    r, g, b, a = np.rollaxis(rgba_img, axis=-1)
    r[a == 0] = col[0]
    g[a == 0] = col[1]
    b[a == 0] = col[2] 
    rgb_img = np.stack([r, g, b])
    rgb_img = np.moveaxis(rgb_img,0,3)
    return rgb_img
import os
import pickle
import numpy as np

def create_folders(path,folders):
    """Create all folders if they don't already exist.

    Keyword arguments:
    path -- location of main folder
    folders -- list of all subfolders
    """
    
    for folder in folders:
        dir_name = path + folder
        try:
            os.makedirs(dir_name)
            print("Directory" , dir_name ,  "created ") 
        except FileExistsError:
            print("Directory" , dir_name ,  "already exists")
            
            
def split_stack(tiff_stack,path,nb_cores):
    """Splits a numpy tiff-stack into chunks of a similar size and dumps them. This is to avoid memory leaks when using multiprocessing, as every subprocess inherits the whole memory of the parent process. If the tiffstack in and output is 8GB and 50 CPUs are used, we already have 2*8GB*50 = 800GB.

    Keyword arguments:
    tiff_stack -- the whole stack
    path -- where the substacks will be stored
    nb_cores -- how many chunks are created
    channels -- which channels to save (e.g "0","[0,1,2]")
    """
    paths = [] #list of paths to all chunks
    nb_frames = np.shape(tiff_stack)[0]-1 #last frame index
    borders = np.round(np.linspace(0,nb_frames+1,nb_cores+1))
    
    print("Splitting of dataset: ")
    for c in range(nb_cores):
        #stack_indices.append((int(borders[c]),int(borders[c+1]-1)))
        start,end = ((int(borders[c]),int(borders[c+1])))
        if np.ndim(tiff_stack) == 3: #for one channel tiffs (e.g just H2B channel)
            this_chunk = tiff_stack[start:end,:,:].copy()
        elif np.ndim(tiff_stack) == 4: #for multichannel tiffs
            this_chunk = tiff_stack[start:end,:,:,:].copy()
        else:
            raise Exception("Unsupported number of dimensions on input stack.") 
        print("Core "+str(c+1)+"/"+str(nb_cores)+": "+str(end-start)+" frames ["+str(start)+"-"+str(end-1)+"].")
        fname = f'chunk_{c:03}.npy'     
        np.save(path+fname,this_chunk)
        paths.append(path+fname)
    return paths
                      
             
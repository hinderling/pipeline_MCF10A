# pipeline_MCF10A

## How to use the code
The code can be run interactively trough jupyter notebooks. There are three notebooks to be used in the order:
* `01_train_classifier.ipynb` Train an illastik-like classifier on a single image. Store the classifier, so it can then be used in the pipeline.
* `02_pipeline.ipynb` apply semantic and instance segmentation on tiff stacks. Track the cells, extract and calculate features like nuclear area or nuclear to cytosolic intensity.
* `03_create_movies.ipynb` Create movies colored by some feature, e.g. nuclear to cytosolic intensity.

### Data structure
The pipeline uses this folder structure, where all (intermediate) results are stored:
```
my_project/ stacks_raw/             -- raw tiff stacks
            classifier/             -- trained classifier
            stacks_segmented/       -- probability map of nuclei
            stacks_labeled/         -- labeled instance segmentation of nuclei
            stacks_labeled_rings/   -- labeled corresponding nuclear rings
            table_untracked/        -- table with features of all nuclei
            table_tracked/          -- same table but nuclei are linked over time
            movies/                 -- output of colored movies
            tmp/                    -- temporary files, used for multiprocessing
```
The name of the results file is always the same name as the input file.

![Overview of all output files](/readme_figures/overview.png)

### Table columns
- `x` and `y`: position of detected particles
- `size`: nb pixels belonging to nucleus
- `frame`: timestep of timelapse movie
- `mean_nuc_c1` / `_c2` and `mean_ring_c1` / `_c2`: mean pixel intensities of extracted nuclei and cytosolic rings. Used to calculate ratios. `c1` and `c2` stand for the channel numbers.
- `ratio_c1 / c2` the  ratio of `mean_nuc` over `mean_ring`
- `particle`: collumn gets added by the tracking library `trackpy` and is the global particle label assigned by trackpy. Continuous between frames if nucleus is linked.
- `label_frame`: the label initially given to each nucleus after a first segmentation. not continuous between frames!
- `p_nucleus`: mean class confidence of all pixels belonging to the nucleus

The pipeline is modular, steps can replaced (e.g instead of using the classifier from the pipeline, import nuclei probability maps created with illastik)



### Setup the environment and install all dependencies: Detailed instructions for Pertzlab
Connect to the cluster via ssh:

    ssh -X username@izblisbon.unibe.ch

Move to the main storage partition so you can access the scripts from all nodes:

    cd myimaging/

Create a new folder where all the code and python libraries will be stored.
    
    mkdir pipeline_projects
    cd pipeline_projects

Download the sourcode from this repo:

    git clone https://github.com/hinderling/pipeline_MCF10A.git

Create a virtual environment (venv) and activate it:

    python3 -m venv pipeline_env
    source pipeline_env/bin/activate   
    
Note: The commandline will now show `(pipeline_env)` in front of your username to indicate that you're running from the venv. To leave the environement after you're done with the work, run `deactivate`.

Upgrade pip and install all the dependencies from the `requirements.txt` file:
    
    pip install --upgrade pip
    pip install -r pipeline_MCF10A/requirements.txt
    
Congratulations, you are done! You can now start the jupyter server by typing:

    jupyter notebook

To connect to it refer to the next chapter or the [Lab Wiki](http://pertzlab.unibe.ch/doku.php?id=wiki:other_software#jupyter_notebooks_on_the_server).
Be aware that you only can access the packages you installed if you started the jupyter server from inside of the venv. OPTIONAL: If you want to be able to start the server from anywhere, you can add the venv to your main jupyter installation.

    pip install --user ipykernel 
    python3 -m ipykernel install --user --name=pipeline_env

This will output the following:

    Installed kernelspec pipeline_env in /home/username/.local/share/jupyter/kernels/pipeline_env

If everything worked the folder in the output will contain a `kernel.json` file, which contains the path to this venv. You can now select the kernel from your venv at runtime in the jupyter server interface in your browser!
For more info, refer to his [blogpost](https://janakiev.com/blog/jupyter-virtual-envs/).


If you want to update the code at a later date, navigate to the repository folder and fetch the changes from github:
    
    cd pipeline_MCF10A
    git fetch --all
    git reset --hard origin/main

### Run a jupyter notebook on the cluster
To start your own jupyter notebook server, first allocate ressources trough slurm: 

    `salloc --mem 250GB -w izbdelhi --time 12:00:00 --cpus-per-task=50`
    
Specifying the node makes it easier to use the scratch drive and forward the right ports. 
Activate the venv:

    source myimaging/pipeline_projects/pipeline_env/bin/activate
    
Next, start a jupyter notebook server by running:

    jupyter notebook

In the output will be an adress and a port number e.g. `localhost:8888`. To connect to the server from your machine, open a new terminal and forward the port:

    ssh -X -L 18888:127.0.0.1:8888 username@izbdelhi.unibe.ch

Here the port 8888 is forwarded from ther server to 18888 on your machine.
Next copy the link into your browser, change the port number and you will have access to the jupyter server.
Make sure that you copy the whole link (including the token), and that the nodes match between `salloc` and the port forward.

A more complete explanation can be found on the [lab wiki](http://pertzlab.unibe.ch/doku.php?id=wiki:other_software). 


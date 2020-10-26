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

The pipeline is modular, steps can replaced (e.g instead of using the classifier from the pipeline, import nuclei probability maps created with illastik)

### How to run a jupyter notebook on the cluster
To start your own jupyter notebook server, first allocate ressources trough slurm:

    `salloc --mem 250GB -w izbdelhi --time 12:00:00 --cpus-per-task=50`
Specifying the node makes it easier to use the scratch drive and forward the right ports. 
Next, start a jupyter notebook server by running:

    `jupyter notebook`

In the output will be an adress and a port number e.g. `localhost:8888`. To connect to the server from your machine, open a new terminal and forward the port:

    `ssh -X -L 18888:127.0.0.1:8888 username@izbdelhi.unibe.ch`

Here the port 8888 is forwarded to 18888.
Next copy the link into your browser, change the port number and you will have access to the jupyter server.
Make sure that you copy the whole link (including the token), and that the nodes match between `salloc` and the port forward.

A more complete explanation can be found on the [lab wiki](http://pertzlab.unibe.ch/doku.php?id=wiki:other_software). 

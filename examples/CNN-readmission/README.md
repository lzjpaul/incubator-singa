# Train CNN for Re-Admission

A CNN is trained over the EMR data for re-admission prediction.

## SINGA version

Note that all examples should clearly specify the SINGA version against which the scripts are tested. The format is `Apache SINGA-<VERSION>-<COMMITID>`. For example,
All scripts have been tested against [Apache SINGA-v1.0.0-fac3af9](https://github.com/apache/incubator-singa/tree/4826d40a1a6b67dd322fec5d3f6a5af1a17dc73d).

## Folder layout

The folder structure for an example is as follows where README.md is required and other files are optional.

* README.md. Every example **should have** a README.md file for the model description, SINGA version and running instructions.
* train.py. The training script. Users should be able to run it directly by `python train.py`. It is optional if the model is shared only for prediction or serving tasks.
* model.py. It has the functions for creating the neural net. It could be merged into train.py and serve.py, hence are optional.

Some models may have other files and scripts. Typically, it is not recommended to put large files (e.g. >10MB) into this folder as it would be slow to clone the gist repo.

## Instructions

### Local mode
To run the scripts on your local computer, you need to install SINGA.
Please refer to the [installation page](http://singa.apache.org/en/docs/installation.html) for detailed instructions.


#### Training
The training program could be started by

        python train.py

By default, the training is conducted on a GPU card, to use CPU for training (very slow), run

        python train.py --use_cpu

The model parameters would be dumped periodically, into `model-<epoch ID>` and the last one is `model`.


### Cloud mode

To run the scripts on the Rafiki platform, you don't need to install SINGA. But you need to add the dependent libs introduced by your code into the requirement.txt file.

#### Adding model

The Rafiki front-end provides a web page for users to import gist repos directly.
Users just specify the HTTPS (NOT the git web URL) clone link and click `load` to import a repo.

#### Training

The Rafiki font-end has a Job view for adding a new training job. Users need to configure the job type as 'training',
select the model (i.e. the repo added in the above step) and its version.
With these fields configured, the job could be started by clicking the `start` button.
Afterwards, the users would be redirected to the monitoring view.
Note that it may take sometime to download the data for the first time.
The Rafiki backend would run `python train.py` in the backend.

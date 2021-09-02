# To-Tust-A-Liar
by Mark Mifsud, Colin Layfield, Joel Azzopardi, John Abela (Dept of Computer Information Systems, Faculty of ICT-University of Malta -Msida Malta)


The whole project is mainly separated based on the datasets used. There is a separate main directory for trials on the LIAR dataset, the Shuffled-LIAR dataset, the Cleaned-LIAR Dataset, Reputation Scores and for the SST-5 dataset, on which Sentiment Analysis was performed.

Note that since the resulting transformer models are each many gigabytes in size they are not being delivered with the code.  The results produced are, however, visible in the saved output of the code being run (within the Jupyter notebook) or in respective output files.  Should you wish to ask for any trained model produced in this study, please contact the first author on mark.mifsud.16@um.edu.mt . 

Following is an overview of the contents:

_ **The main directories** _

| **Main Directory** | **Contents** |
| --- | --- |
| Cleaned LIAR | Dataset and Experiments on Cleaned LIAR |
| LIAR Unchanged | Dataset and Experiments on LIAR |
| Shuffled LIAR | Dataset and Experiments on the Shuffled-LIAR |
| Reputation Only | Dataset and FcNN code that utilise Reputation scores only |
| Sentiment | The SST-5 dataset and experiments run on it |
| Supplementary Material | Extra code that was used and other resources |


Sub-directories and files common to more than one directory:

| **Sub-directory or file** | **Description** |
| --- | --- |
| [Transformer Name]\_Classification\_Vector with Reputation.ipynb | The code that uses the transformer [Transformer Name], used to do classification with said transformer including and excluding the FcNN.  There should be a variant of this file for each transformer used. |
| [Transformer Name]\_Classification\_Vector with Reputation-Evolving.ipynb | The Code segment that trains a FcNN using a Evolving Reputation and the output from a transformer. |
| Ill\_Conditioned\_test -[Transformer Name].ipynb | The code used to discern if the Fine Grained, Fake-News classification of short statements is ill-conditioned. |
|   |
| train[description].xls | The training part of the dataset, where [description] indicates the main dataset being used. |
| valid[description].xls | The validation set |
| test[description].xls | The testing set |
| train-Reputation.xls | The reputation vector for the training set |
| valid-Reputation.xls | The reputation vector for the validation set |
| test-Reputation.xls | The reputation vector for the testing set |
| /TunedModels/ | The fine-tuned transformers are saved here together with their output and the FcNN that takes their output for classification. |
| /TunedModels/[transformer class]/[transformer version]/ | There should be a variant of this directory for each transformer used.  The weights of the fine- tuned transformer are stored here. |
| /TunedModels/[transformer class]/[transformer version]/cache | The cache produced when fine tuning the transformer is saved every few steps. In case the process encounters an issue and needs to be stopped, it can continue from the saved position. |
| /TunedModels/[transformer class]/[transformer version]/Saves | The classification vector, output from the transformer for each entry in the dataset is store here to be used by the FcNN. |
| /TunedModels/[transformer class]/[transformer version]/NNetwork | Trained FcNNs were saved here and the code allows one to load them again. |

On the datasets which were used for the ill-conditioned test (SS-5,Cleaned-LIAR and LIAR), these subfolders will be present.

| /folds/ | The code used to investigate for testing the ill-conditioning of the problem saves the trained transformers and all relevant data in this directory. |
| /folds/train\_fold[fold number].xls | There are 5 of these. They are the different training sets used. |
| /folds/valid[fold number].xls | There are 5 of these. They are the different validation sets used. |
| /folds/[transformer class]\_results[date] [time].xls | The results obtained by running the transformer [transformer class] |
| /folds/fold[fold number]/ [transformer class]/[transformer version]/ | There are 5 of these directories and they contain the transformers fine-tuned with a different training set (with results) |

_ **Sub-directories and files common to only one directory:** _

In /LIAR Unchanged/

| Vector for User Input string.ipynb | This notebook is useful to simply see the output vector from BERT.  This is the notebook that was used to test if BERT is modelling veracity for section 4.4 of the paper.  It is possible to check the categorisation of any arbitrary statement with it too.|
| --- | --- |

**In /Reputation Only/**

| **Sub-directory or file** | **Description** |
| --- | --- |
| Reputation Based Classifier.ipynb | Code for the FcNN used to classify statements using only the reputation scores. |
| Sigmoid\_linearMSE\_adam\_43.pth | The saved weights of this FcNN |
| TrainReputationLowNoise.xlsx | The data that the FcNN was trained on. |
| /Reputation Bias Test | This directory contains code for the test to check how the FcNN treats statements with a level of veracity which doesn&#39;t correspond to the speaker&#39;s usual level of honesty. |
| /Reputation Bias Test/RepBias.ipynb | The code for this test |
| /Reputation Bias Test/test\_reputationExceptions.xlsx | The data.  These are lies by mostly honest people, and truth said by frequent liars. |

**In /Sentiment/**

| /SST\_data | The directory containing the SST-5 Dataset |
| --- | --- |

**In /Supplementary Material/**

| **Sub-directory or file** | **Description** |
| --- | --- |
| /Compute truth column/ | Code used to count the number of True statements for each speaker. Used for LIAR. |
| /computing reputation for Cleaned LIAR/ | Since a number of entries were omitted from Clean\_LIAR, this is the code used to recount the statements for each speaker in each of the 6 categories.There is also the code used to compute the evolving reputation score. |
| /LIAR unprocessed | The LIAR dataset as first downloaded, without any modification. The one used for transformers was adapted for that task. |
| Google Colab specific code.ipynb | The commented code, containing the lines that need to be added to any notebook, if it is to be run on Google Colab. |
| torch-transformers.yaml | The Conda Environment used for the project |

**Hardware and Software used**

Transformers are computationally very demanding. All the code was produced, tested and run on a PC having a GPU with 24Gb of VRAM.  The code is meant to run on a CUDA Enabled GPU with 11GB (like Nvidia RTX 2080 Ti) or more.  If this is not available there are other options mentioned below.

_ **Software pre-requisites:** _

The following need to be installed:
Latest GPU Drivers
Nvidia CUDA 10 or later
Nvidia Apex
Python 3
Conda

_ **Programming Language, IDE &amp; Environment** _

Python was the language of choice since it is well suited for Machine Learning.  Anaconda with Jupyter Notebook was used as the development environment, since it is free to use for anyone, and the code can also be easily opened in Google Colab, Floydhub or similar cloud-based environments, making it more accessible.

Anaconda allows multiple environments, each with different toolchains and libraries to be available on the same machine.   The same libraries that were used to run the code need to be installed.  The quickest way to install all necessary Python libraries in Conda is to use the &quot;torch-transformers.yaml&quot; environment supplied int the Supplementary Material directory.  A spreadsheet program was also used to inspect the dataset and make modifications necessary for tests.

_ **Libraries used** _

Libraries used include Sklearn, Pandas, Pytorch for the FcNN, HuggingFace Transformers for the BERT variants and Thilina Rajapakse&#39;s Simple Transformers to make the Transformer code simpler and more readable.
https://huggingface.co/models
https://simpletransformers.ai/about/

_ **Troubleshooting known artefact issues** _

This sudy&#39;s code was composed on Windows 10 and also verified to run well on Linux Mint (an Ubuntu variant).

Nvidia Apex is not guaranteed to work properly on Windows.  In case of issues with CUDA drivers and nVidia Apex, any Ubuntu based Linux distro is recommended since it is known that Apex is commonly targeted for this OS variant.

_ **In case a high end GPU is not available** _

**Option 1:**   running the code on Google Colab with files on Google Drive.

Google, through Colab provide free GPU usage in the cloud, but they may enforce which GPU is used or remove GPU access entirely if one uses them abundantly.

A few lines of code will need to be added to the beginning of each notebook to adapt the Colab environment to run the code and also for Colab to access files on Google Drive.  The necessary Google Colab specific code is provided with the delivered artefact and includes comments for explanation and guidelines.  It is in the Supplementary Material directory in the &quot;Google Colab specific code.ipynb&quot; file.

**Option 2** :  A GPU with less RAM, using smaller training batches

A GPU with 8GB of RAM may be enough to run BERT-base.  Roberta-large and Albert-Large-V2 will require more RAM and in that case an Out of Memory error will be reported.  If Out of Memory Errors occur, during fine-tuning, when trying to replicate the results, lower the training-batch size to 32 or 16.

**Option 3:**   Running on CPU

If no adequate CUDA GPU is found, the CPU will be used automatically, however, running Transformers on CPU will take a lot of time so it is not recommended.  BERT-base may take about 20 minutes just to train for one epoch on LIAR on a CPU with 16-threads.  Smaller batches can help if system RAM is scarce, and the smaller the batches the slower the program will run.  Results may also be affected (probably not by much).

_ **Known crashing issues and how to bypass them** _

In the tests where a transformer is followed by a FcNN, the FcNN may crash.  This is due to some variables used by the Simple Transformers library that are not being cleared from memory.

To overcome this issue, the Jupiter Notebook clearly marks the section between the transformer and the FcNN and saves the results of any part of the training to be read from disk at a later step.  Simply halt the program if there are crashes are start running the notebook from the start of the FcNN section.   All necessary libraries and data will be loaded from disk and proceed as intended.

If performing this bypass on the Google Colab service, the setup code that is specific to Colab has to be run prior to continuing from the FcNN section of the notebook.

_ **Cautionary note about Disk Space** _

After every epoch the transformer adds files in the cache directory and the weights are saved after a number of steps which the user can set.  These are very large in size and saving at frequent steps will quickly consume a lot of disk space needlessly.   Ideally only save states after every epoch, and set the cache to overwrite, in the transformer&#39;s variables.


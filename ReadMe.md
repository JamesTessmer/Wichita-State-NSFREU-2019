NetCPS REU 2019 - James Tessmer
Securing Microbiome Classification via Obfuscation
There are two main parts in this research: The GeNet model and the Obfuscation model. Both have their own ReadMe detailing installation instructions and any required packages/libraries, so I will cover the new things I've added and any additional resources required along with any other clarifying or important information.

GeNet
Source Code - https://github.com/mrojascarulla/GeNet
Directories:  
GeNet/ : contains the ReadMe along withe environment and install requirements files.
    Code/ : holds the required code. The only file that should require execution is genet_train.py which can be done through CDing to code then executing 'python genet_train.py -dir_path=../data'
    Data/ : Holds the csv files used to create the taxonomic tree which allows for classification and organization of genomes

The network utilizes genomes from the NCBI database. If not present, they will be downloaded when executing genet_train.py and placed in a 'genomes' folder located in the data directory, this folder is roughly 30gb in size.

I made very few changes to GeNet, mostly minor things to allow it to run on my machine which shouldn't impact the function.
There are a few notable changes; however,
	- The TaxoTree.py file assumes a user will utilize all the genome data. It tracks the number of genomes to load using a 'num_labels' variable. On line 129 I hardcoded a value here to manage the size of the dataset, this line can be removed if you want to use all the data or the value can be changed to the desired number.
	- In genet_train.py in line 124 I've added code (a work in progress, so it may not be finished) that works to use the GeneDataset (discuessed below) to allow GeNet to take in image data instead of a genome, which will hopefully allow it to take the image of an obfuscated genome.

important notes:
	-The genome files are all the data I've used for this research, so GeNet should be run first to make sure this data is present and usable for the rest of the code.
	-I had issues with compatibility when trying to initially install GeNet, I solved this by removing version numbers from the environment file and using that to install the required libraries/packages. I haven't had version issues since then, but as things change issues may arise so a copy of the original environment file with version numbers is included for reference.


Obfuscation Network
Source Code - https://github.com/kuangliu/pytorch-cifar.
The readme included in the obfuscation directory is comprehensive enough in regards to what's contained in each directory and how to run the original files so I will only include my additions here.

In Obfuscation-Net/ I've added some of the GeNet files as they're needed if a user wants to implement GeNet as the classifier for the obfuscation model.
In this same directory I've also added ResNet18Gene.py. This is an implementation of ResNet18 meant to finetune a pretrained model with the genome data. This code should work with other versions of GeNet and any other model available with TorchVision as it was written using a pytorch tutorial. 
This model can be run by executing:
$cd/<paths>
$python ResNet18Gene.py
I've also added a custom dataset, described below, to obf-training.py but a proper classifier for this data has not yet been implemented.


In Obfuscation-Net/Data I've included TaxoTree.py, which is the inclusion of some code to pull the genus of each genome from the csv files to be used as labels for ResNet18Gene.py
This directory also includes gene_dataset.py which is used in ResNet18Gene.py as a custom dataset. It converts genomes to images and returns these along with the label to the dataloader.

Final Notes:
	-There are some hardcoded paths as my virtual machine didn't have enough disk space to store the genomes in both the GeNet folder and the obfuscation folder. These paths are located in:
		-Obfuscation-Net/ResNet18Gene.py, line 33
		-Obfuscation-Net/obf-training.py, line 174
		-Genet/code/genet_train.py, line 88
	-The results of GeNet training can be visualized in TensorBoard, but the other networks are not set up to use this
	-This work has been done on a VirtualBox Ubuntu virtual machine, so work done on another system may require unforseen changes


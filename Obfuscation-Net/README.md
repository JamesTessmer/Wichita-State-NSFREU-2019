Quick README for the NETCPS REU Summer 2018 program 
Privacy-Aware Offloading Dependency To Mutable Networks
 - Nathaniel Hoefer

This should be all materials needed to build off of the research thus far. I apologize for the lack of organization in some files and directories, but overall, it should be fairly evident of how things are structured. 

Directories:
/Obfuscation-Net: Contains all of the data files
    /data: Location of the CIFAR-10 images
    /evaluations: Contains the data from the evaluations from the experiments
    /models: Location of the classification, obfuscation, and deobfuscation models used for the experiments
    /pytorch_cifar: Contains code to generate the ResNet18 classification model. Files originally pulled from https://github.com/kuangliu/pytorch-cifar.
    
Execution:

Both the classification and obfuscation files utilize the PyTorch framework, so naturally, you must have PyTorch installed, as well as a few other libraries which should be clearly indicated when executing if you don't have them installed. 

In order to train a classification model, you must execute pytorch_cifar/main.py. This file is fairly straight-forward as there are no arguments to be passed in, however, you may need to tweak some parameters which is done via the constants found in the main.py file (all upper-case). 

Most of my work was done in the Obfuscation-Net/obf-training.py file, which also acts as the executable for training and evaluating the obfuscation model. There are various sections from helper functions, parameters, data preparation, building the model, and training the model. It should make sense from a PyTorch perspective, but due to time constraints towards the end, I had to do a few hacks such as comment out lines and change string literals to switch between things like training and evaluation. Most changes should be accomplished via the parameters section, but there may need to have other changes as well.

Executing obf file:
$ cd <paths>/Obfuscation-Net/
$ python obf-training.py

Executing classification file:
$ cd <paths>/Obfuscation-Net/pytorch_cifar/
$ python main.py


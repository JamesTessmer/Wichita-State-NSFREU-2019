
"""


"""

import os
import re
import time
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from data import gene_dataset
from pytorch_cifar.utils import progress_bar
from PIL import Image
import taxotree

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy

# Establish paths for storing images
image_dir = os.path.join(os.getcwd(), "images")
training_image_dir = os.path.join(image_dir, time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime()))
if not os.path.exists(image_dir):
    os.mkdir(image_dir)
if not os.path.exists(training_image_dir):
    os.mkdir(training_image_dir)


def plot_images(images, epoch=0, batch=0, save2file=None):
    """Plots images for viewing or saving."""

    plt.figure(figsize=(10,10))
    for i in range(len(images)):
        plt.subplot(1, 3, i+1)
        fmt_images = images[i].detach().numpy()
        plt.imshow(np.transpose(fmt_images, (1, 2, 0)))
        plt.axis('off')
    plt.tight_layout()
    if save2file:
        plt.savefig(save2file)
        plt.close('all')
    else:
        plt.show()


def classify_cifar10(y_list):
    """Returns string representing class from binary matrix representation of CIFAR-10

    :param list y_list: a binary matrix representing the CIFAR-10 classes
    """
    classes = ["airplane",
               "automobile",
               'bird',
               "cat",
               "deer",
               "dog",
               "frog",
               "horse",
               "ship",
               "truck"]
    index = torch.argmax(y_list).item()
    return classes[index]


def print_size(label, tensor, expected=[3, 32, 32]):
    """Prints the size of the given tensor and its expected size."""
    size = [tensor.size()[1], tensor.size()[2], tensor.size()[3]]
    print(label, " - Count:", tensor.size()[0], " - Expected:", expected, " - Actual:", size)


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """

    def tryint(s):
        try:
            return int(s)
        except:
            return s

    return [ tryint(c) for c in re.split('([0-9]+)', s) ]


#########################################################################################################
# Parameters
#########################################################################################################

# Dataset parameters
USE_SUBSET = False
TRAIN_START_IDX = 0
TRAIN_CNT = 20000
TRAIN_BATCH_SZ = 32
TEST_START_IDX = 0
TEST_CNT = 5000
TEST_BATCH_SZ = 100

# Model Parameters
CLASS_MODEL_DIR = 'models/class-models/Final_Class_2018-07-06_18-35-33'
CLASS_MODEL_NAME = 'class-model_resnet18_ep-55_ac-78.750.t7'
# OBF_MODEL_DIR = "models/obf-models/70-Class_2018-07-09_15-58-54"
# OBF_MODEL_NAME = "obf-model_resnet18_ep-100_ac-40.680.t7"
# DEOBF_MODEL_NAME = "deobf-model_resnet18_ep-100_ac-40.680.t7"
OBF_MODEL_DIR = "models/obf-models/Base_Obf_2018-07-05_17-48-04"
OBF_MODEL_NAME = "obf-model_resnet18_ep-100_ac-40.680.t7"
DEOBF_MODEL_NAME = "deobf-model_resnet18_ep-100_ac-40.680.t7"
DATETIME = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())

# Evaluation Parameters
EVALUATE = False #change later
EVALUATION_DIR = os.path.join("evaluations", "BaseObf_TrainedClass_" + DATETIME)
if EVALUATE and not os.path.isdir(EVALUATION_DIR):
    os.mkdir(EVALUATION_DIR)
IMAGES_DIR = os.path.join(EVALUATION_DIR, "images")
if EVALUATE and not os.path.isdir(IMAGES_DIR):
    os.mkdir((IMAGES_DIR))
IMAGES_PRED_FILEPATH = os.path.join(IMAGES_DIR, "predictions.txt")
EVAL_INFO_FILEPATH = os.path.join(EVALUATION_DIR, "info.txt")
EVAL_INFO_NOTE = "Evaluation of the Base Obfuscation Network (~40% accuracy) with the trained Classification network starting with epoch 55(~78% accuracy) to epoch 100(~81% accuracy) using CIFAR-10 testing set."

# Training Parameters
RESUME = False
START_EPOCH = 0  # start from epoch 0 or last checkpoint epoch
EPOCH_COUNT = 100
MODEL_FILEPATH = os.path.join("models", "obf-models", "70-Class_" + DATETIME)
if not EVALUATE and not os.path.isdir(MODEL_FILEPATH):
    os.mkdir(MODEL_FILEPATH)
TRAIN_INFO_FILEPATH = os.path.join(MODEL_FILEPATH, "info.txt")
TRAIN_INFO_NOTE = "Updated Obfuscation training with full CIFAR-10 dataset using classification nn trained for an additional 20 epochs and full CIFAR-10 dataset."


#########################################################################################################
# Data Preparation
#########################################################################################################


def datasubset(loader, start, count, batch_size):
    """Return a subset of the dataloader from the start batch index to the count specified."""
    # Note: start is the start index of batch, not image
    smaller_dataset = []
    end_idx = count / batch_size
    for batch_idx, (orig_images, labels) in enumerate(loader):
        if start <= batch_idx < end_idx:
            smaller_dataset.append((orig_images, labels))
        if batch_idx > end_idx:
            break
    return smaller_dataset


print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])
# utilizes gene-dataset to load data
dir_path =  str(os.getcwd()) + "/data"
dataset_list = 'ncbi_ids.csv'

path_to_taxo = os.path.join(dir_path, 'nodes.dmp')
path_to_dataset = os.path.join(dir_path, dataset_list)
path_to_genomes = os.path.join('Hardcoded path', 'genomes')
	
#creating tree that contains genome data, labels, etc
taxo_tree =taxotree.TaxoTree(path_to_taxo)
taxo_tree.trim_to_dataset(path_to_dataset)
taxo_tree.load_genomes(path_to_genomes)
#read length needs to be the same as the genet train read length

#creating a list of pre shuffled genomes as reference so that the index of label i equals the index of gene i
pre_shuffle_list = []
for gene in taxo_tree.genomes:
    pre_shuffle_list.append(gene)
shuffle(taxo_tree.genomes)

labels = [None]*len(taxo_tree.genomes)
for i in range(0,len(pre_shuffle_list)):
    for p in range(0, len(taxo_tree.genomes)):
        if np.array_equal(taxo_tree.genomes[i],pre_shuffle_list[p]):
            labels[p]=(taxo_tree.genome_to_genus[i])
#print(len(labels))
#print(len(taxo_tree.genome_to_genus))
classes = []
for i in range(0,len(taxo_tree.genomes)):
    if taxo_tree.genome_to_genus[i] not in classes:
        classes.append(taxo_tree.genome_to_genus[i])



trainset = gene_dataset.gene_dataset(taxo_tree,labels, classes,training=True, transform = transform_train)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SZ, shuffle=True, num_workers=2)

testset = gene_dataset.gene_dataset(taxo_tree, labels, classes,transform = transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SZ, shuffle=False, num_workers=2)

if USE_SUBSET:
    trainloader = datasubset(trainloader, TRAIN_START_IDX, TRAIN_CNT, TRAIN_BATCH_SZ)
    testloader = datasubset(testloader, TEST_START_IDX, TEST_CNT, TEST_BATCH_SZ)


# Store info to file
if EVALUATE:
    info_path = EVAL_INFO_FILEPATH
    info_note = EVAL_INFO_NOTE
else:
    info_path = TRAIN_INFO_FILEPATH
    info_note = TRAIN_INFO_NOTE
with open(info_path, "w") as f:
    print(info_note, file=f)
    print("\nDatetime:", DATETIME, file=f)
    print("Number of training images:", len(trainloader) * TRAIN_BATCH_SZ, file=f)
    print("Number of testing images:", len(testloader) * TEST_BATCH_SZ, file=f)
    print("Starting Classification NN:", CLASS_MODEL_NAME, file=f)
    if RESUME:
        print("Initial Checkpoint:", OBF_MODEL_NAME, file=f)
    elif EVALUATE:
        print("Obfuscation NN:", OBF_MODEL_NAME, file=f)
    print("\n", file=f)

#########################################################################################################
# Building Model
#########################################################################################################
print('==> Building model..')

"""
Network Architecture:
    Input       Module          Output Channels     Stride

    3x32x32     Conv2D                  32          1
    32x32x32    Bottleneck              64          2
    64x16x16    Bottleneck              128         2
    128x8x8     Bottleneck              128         2
    128x4x4     Upsample Bottleneck     64          1
    64x8x8      Upsample Bottleneck     32          1
    32x16x16    Upsample Bottleneck     3           1
"""


class BottleNeck(nn.Module):
    """Pointwise Conv + Depthwise Conv + Pointwise Conv"""
    def __init__(self, in_channels=32, out_channels=64, stride=1):
        super(BottleNeck, self).__init__()
        # Pointwise Conv
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        # Depthwise Conv
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                               stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        # Pointwise Conv
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = F.leaky_relu(self.bn2(self.conv2(out)))
        out = F.leaky_relu(self.bn3(self.conv3(out)))
        return out


class UpBottleNeck(nn.Module):
    """Upsample + Pointwise Conv + Depthwise Conv + Pointwise Conv"""
    def __init__(self, in_channels=32, out_channels=64, stride=1):
        super(UpBottleNeck, self).__init__()
        # Upsample
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.bn1 = nn.BatchNorm2d(in_channels)
        # Pointwise Conv
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        # Depthwise Conv
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                               stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        # Pointwise Conv
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.up1(x)))
        out = F.leaky_relu(self.bn2(self.conv1(out)))
        out = F.leaky_relu(self.bn3(self.conv2(out)))
        out = F.sigmoid(self.bn4(self.conv3(out)))
        return out


class ObfNet(nn.Module):

    SUMMARY = False

    def __init__(self):
        super(ObfNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=2, stride=1, padding=0, bias=False)
        self.batchn1 = nn.BatchNorm2d(32)
        self.bn1 = BottleNeck(32, 32, 2)
        self.bn2 = BottleNeck(32, 64, 2)
        self.bn3 = BottleNeck(64, 128, 2)
        self.ubn1 = UpBottleNeck(128, 64)
        self.ubn2 = UpBottleNeck(64, 32)
        self.ubn3 = UpBottleNeck(32, 3)

    def forward(self, x):
        if self.SUMMARY: print_size("X -> Conv1", x, [3, 32, 32])
        out = F.leaky_relu(self.batchn1(self.conv1(x)))
        if self.SUMMARY: print_size("Conv1 -> BN1", out, [32, 32, 32])
        out = self.bn1(out)
        if self.SUMMARY: print_size("BN1 -> BN2", out, [32, 16, 16])
        out = self.bn2(out)
        if self.SUMMARY: print_size("BN2 -> BN3", out, [64, 8, 8])
        out = self.bn3(out)
        if self.SUMMARY: print_size("BN3 -> UBN1", out, [128, 4, 4])
        out = self.ubn1(out)
        if self.SUMMARY: print_size("UBN1 -> UBN2", out, [64, 8, 8])
        out = self.ubn2(out)
        if self.SUMMARY: print_size("UBN2 -> UBN3", out, [32, 16, 16])
        out = self.ubn3(out)
        if self.SUMMARY: print_size("UBN3 -> Out", out, [3, 32, 32])
        return out

    def set_summary(self, value):
        """If true, show summary during forward propagation."""
        self.SUMMARY = value


#########################################################################################################
# Training Model
#########################################################################################################

# Load Classification network
from pytorch_cifar.models import ResNet18
classification = ResNet18()
classification = classification.to(device)
assert os.path.isdir(CLASS_MODEL_DIR), 'Error: no checkpoint directory found!'
checkpoint = torch.load(os.path.join(CLASS_MODEL_DIR, CLASS_MODEL_NAME))
classification.load_state_dict(checkpoint['net'])

from torch.nn.modules.loss import _Loss
from torch.autograd import no_grad
class Euc_Dist(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super(Euc_Dist, self).__init__(size_average, reduce)

    def forward(self, input, target):
        with torch.no_grad():
            return torch.sqrt(torch.sum((target - input) ** 2))

obf = ObfNet()
deobf = ObfNet()

if RESUME:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(OBF_MODEL_DIR), 'Error: no checkpoint directory found!'
    obf_checkpoint = torch.load(os.path.join(OBF_MODEL_DIR, OBF_MODEL_NAME))
    obf.load_state_dict(obf_checkpoint['net'])
    deobf_checkpoint = torch.load(os.path.join(OBF_MODEL_DIR, DEOBF_MODEL_NAME))
    deobf.load_state_dict(deobf_checkpoint['net'])
    best_acc = obf_checkpoint['acc']
    # start_epoch = obf_checkpoint['epoch'] + 1

obf_crit = nn.CrossEntropyLoss()
deobf_crit = nn.MSELoss()

# Optimizers
optimizer_O = optim.Adam(obf.parameters(), lr=0.001)
optimizer_D = optim.Adam(obf.parameters(), lr=0.001)


def train(epoch):
    obf.train()  # Prepares certain modules for training
    deobf.train()
    o_train_loss = 0
    d_train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (orig_images, labels) in enumerate(trainloader):

        # Prep for batch
        orig_images, labels = orig_images.to(device), labels.to(device)
        optimizer_O.zero_grad()
        optimizer_D.zero_grad()

        # 1. Generate obfuscated images from original images
        obf_images = obf(orig_images)

        # 2. Reconstruct original images from obfuscated images
        rec_images = deobf(obf_images)

        # 3. Compare reconstructed image and real image for euclidian distance loss
        d_loss = deobf_crit(rec_images, orig_images)

        # 4. Back-propagate the d-model with euclidian distance loss
        d_loss.backward(retain_graph=True)
        optimizer_D.step()

        # 5. Pass obfuscated images through classification and compute cross entropy loss
        prediction = classification(obf_images)
        # print("Prediction:", prediction)

        c_loss = obf_crit(prediction, labels)

        # 6. Back-propagate the o-model with euclidian distance loss and cross-entropy loss
        o_loss = c_loss - d_loss  # Minimize classification loss, Maximize reconstruction loss
        o_loss.backward()
        optimizer_O.step()

        # Calculate losses for output
        d_train_loss += d_loss.item()
        o_train_loss += o_loss.item()
        _, predicted = prediction.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar(batch_idx, len(trainloader), 'D-Loss: %.3f | O-Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (d_train_loss/(batch_idx+1), o_train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # if batch_idx % 10 == 0:
        #     # print("Actual:", classes[labels[0].item()], "Predict:", classify_cifar10(prediction[0]))
        #     print("")
        #     images = torch.stack((orig_images[0], obf_images[0], rec_images[0]))
        #     plot_images(images, epoch=epoch, batch=batch_idx, save2file=True)


def test(epoch):
    global best_acc
    obf.eval()
    deobf.eval()
    o_test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (orig_images, labels) in enumerate(testloader):

            orig_images, labels = orig_images.to(device), labels.to(device)
            obf_images = obf(orig_images)
            ''' 
            #checking formatting of the obf images
            im_transform = transforms.ToPILImage()
            for i in range(0,len(obf_images)):
                im = im_transform(obf_images[i])
                im.show()
                pixels = list(im.getdata())
                #print(pixels)
           
            #return
            '''

            rec_images = deobf(obf_images)

            d_loss = deobf_crit(rec_images, orig_images)
            prediction = classification(obf_images)
            c_loss = obf_crit(prediction, labels)
            o_loss = c_loss - d_loss

            o_test_loss += o_loss.item()
            _, predicted = prediction.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (o_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    with open(TRAIN_INFO_FILEPATH, "a") as f:
        print("Epoch, %d, Accuracy, %.3f" % (epoch, acc), file=f)
    if epoch % 5 == 0 or epoch == EPOCH_COUNT:
        print('Saving..')
        obf_state = {
            'net': obf.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        deobf_state = {
            'net': deobf.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        obf_filename = "obf-model_resnet18_ep-%d_ac-%.3f.t7" % (epoch, acc)
        deobf_filename = "deobf-model_resnet18_ep-%d_ac-%.3f.t7" % (epoch, acc)
        torch.save(obf_state, os.path.join(MODEL_FILEPATH, obf_filename))
        torch.save(deobf_state, os.path.join(MODEL_FILEPATH, deobf_filename))
        best_acc = acc


def eval_delta_obf():
    """Evaluates the accuracy of the NN by iterating through the obf nn with a fixed class nn."""

    obf_files = [file for file in os.listdir(OBF_MODEL_DIR) if file.startswith(OBF_MODEL_NAME)]
    obf_files.sort(key=alphanum_key)
    deobf_files = [file for file in os.listdir(OBF_MODEL_DIR) if file.startswith(DEOBF_MODEL_NAME)]
    deobf_files.sort(key=alphanum_key)

    for obf_file, deobf_file in zip(obf_files, deobf_files):
        epoch = re.search("ep-(\d+)", obf_file).group(1)
        print("Evaluating Epoch %s..." % epoch)

        # 1. Load initial obfuscation and deobfuscation network
        obf_state = torch.load(os.path.join(OBF_MODEL_DIR, obf_file))
        obf.load_state_dict(obf_state['net'])
        deobf_state = torch.load(os.path.join(OBF_MODEL_DIR, deobf_file))
        deobf.load_state_dict(deobf_state['net'])

        # 2. Test the obfuscation accuracy using the test images and classification network
        obf.eval()
        deobf.eval()
        o_test_loss = 0
        correct = 0
        total = 0
        acc = 0
        with torch.no_grad():
            for batch_idx, (orig_images, labels) in enumerate(testloader):
                orig_images, labels = orig_images.to(device), labels.to(device)
                obf_images = obf(orig_images)
                rec_images = deobf(obf_images)

                d_loss = deobf_crit(rec_images, orig_images)
                prediction = classification(obf_images)
                c_loss = obf_crit(prediction, labels)
                o_loss = c_loss - d_loss

                o_test_loss += o_loss.item()
                _, predicted = prediction.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (o_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                acc = 100. * correct / total

                # Store random image (original, obfuscation, and deobfuscation state)
                if batch_idx == 0:
                    idx = random.randint(0, len(orig_images) - 1)
                    label_class = classes[labels[idx]]
                    pred_class = classes[predicted[idx]]
                    with open(IMAGES_PRED_FILEPATH, "a") as f:
                        print("Epoch: %s Label: %s, Prediction %s" % (epoch, label_class, pred_class), file=f)
                    images = [orig_images[idx], obf_images[idx], rec_images[idx]]
                    img_file = os.path.join(IMAGES_DIR, "epoch-%s_%s" % (epoch, label_class))
                    plot_images(images, epoch, save2file=img_file)

        # Store results in CSV format
        with open(EVAL_INFO_FILEPATH, "a") as f:
            print("Epoch, %s, Accuracy, %.3f" % (epoch, acc), file=f)


def eval_delta_class():
    """Evaluates the accuracy of the NN by iterating through the class nn with a fixed obf nn."""

    class_files = [file for file in os.listdir(CLASS_MODEL_DIR) if file.startswith('class-model')]
    class_files.sort(key=alphanum_key)

    # Load base obfuscation and deobfuscation networks
    obf_state = torch.load(os.path.join(OBF_MODEL_DIR, OBF_MODEL_NAME))
    obf.load_state_dict(obf_state['net'])
    deobf_state = torch.load(os.path.join(OBF_MODEL_DIR, DEOBF_MODEL_NAME))
    deobf.load_state_dict(deobf_state['net'])

    for class_file in class_files:
        epoch = re.search("ep-(\d+)", class_file).group(1)
        print("Evaluating Epoch %s..." % epoch)

        # 1. Load classification network
        class_state = torch.load(os.path.join(CLASS_MODEL_DIR, class_file))
        classification.load_state_dict(class_state['net'])

        # 2. Test the obfuscation accuracy using the test images and classification network
        obf.eval()
        deobf.eval()
        o_test_loss = 0
        correct = 0
        total = 0
        acc = 0
        with torch.no_grad():
            for batch_idx, (orig_images, labels) in enumerate(testloader):
                orig_images, labels = orig_images.to(device), labels.to(device)
                obf_images = obf(orig_images)
                rec_images = deobf(obf_images)

                d_loss = deobf_crit(rec_images, orig_images)
                prediction = classification(obf_images)
                c_loss = obf_crit(prediction, labels)
                o_loss = c_loss - d_loss

                o_test_loss += o_loss.item()
                _, predicted = prediction.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (o_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                acc = 100. * correct / total

                # Store random image (original, obfuscation, and deobfuscation state)
                if batch_idx == 0:
                    idx = random.randint(0, len(orig_images) - 1)
                    label_class = classes[labels[idx]]
                    pred_class = classes[predicted[idx]]
                    with open(IMAGES_PRED_FILEPATH, "a") as f:
                        print("Epoch: %s Label: %s, Prediction %s" % (epoch, label_class, pred_class), file=f)
                    images = [orig_images[idx], obf_images[idx], rec_images[idx]]
                    img_file = os.path.join(IMAGES_DIR, "epoch-%s_%s" % (epoch, label_class))
                    plot_images(images, epoch, save2file=img_file)

        # Store results in CSV format
        with open(EVAL_INFO_FILEPATH, "a") as f:
            print("Epoch, %s, Accuracy, %.3f" % (epoch, acc), file=f)

#########################################################################################################
# Executing Model
#########################################################################################################


if __name__ == '__main__':

    if EVALUATE:
        # eval_delta_obf()
        eval_delta_class()
    else:
        for epoch in range(START_EPOCH, START_EPOCH + EPOCH_COUNT):
            print('\nEpoch: %d' % epoch)
            print("Training..")
            train(epoch)
            print("Evaluating..")
            test(epoch)




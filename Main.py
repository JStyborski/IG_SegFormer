import numpy as np
import cv2
import os
from collections import OrderedDict

import torch

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

from Integrated_Gradients import run_baseline_integ_gradients
from Visualization import visualize, generate_entire_images
from Define_Class_Dicts import define_class_dicts

###############
# User Inputs #
###############

print('Starting Code')

imgDir = 'sample_images/img'
imgName = '0016E5_01350.png'
modelCkpt = 'checkpoints/b0_epoch=81-val_mean_iou=0.53.ckpt'
useCuda = True
nRandBaselines = 5 # Number of random baselines images, set as None to use zero baseline
nSteps = 25 # Number of integration steps between baseline and target

# Target pixel location
tgtPxH1 = 60
tgtPxW1 = 200
tgtPxH2 = tgtPxH1
tgtPxW2 = tgtPxW1

# Choose 'rank' (e.g., best prediction) or 'idx' (pure class label index) for class types
# Can also choose 'same' for class2Type, which just copies the same target label index as class1 - ignores class2Label
class1Type = 'rank'
class1Label = 0
class2Type = 'rank'
class2Label = 10

##############
# Misc Setup #
##############

# Output matrix dimensions are 1/4 the size of the input image
tgtPxH1 = int(tgtPxH1 / 4)
tgtPxW1 = int(tgtPxW1 / 4)
tgtPxH2 = int(tgtPxH2 / 4)
tgtPxW2 = int(tgtPxW2 / 4)

# Function that imports segmentation labels and colors
id2label, id2color, label2id = define_class_dicts('class_dict.csv')

# Set processing device: cuda or cpu
device = torch.device('cuda' if torch.cuda.is_available() and useCuda else 'cpu')

# Check if results directory and model subfolder exists, if not create it
if not os.path.exists('results/'):
    os.mkdir('results/')

#################
# Image Reading #
#################

print('Reading Image')

# Read the image as a HxWxRGB numpy array
imgArr = cv2.imread(os.path.join(imgDir, imgName))

##################
# Model Creation #
##################

print('Creating Model')

# Instantiate model with label dictionaries
model = SegformerForSemanticSegmentation.from_pretrained('nvidia/mit-b0', num_labels=32, id2label=id2label, label2id=label2id)

# Load the checkpoint weights
# The given checkpoint is actually made in PyTorch Lightning, such that it contains much more than the state_dict
# Downselect to just the state_dict key, then rename all the keys in the state_dict to remove the 'net.' prefix
cpSD = torch.load(modelCkpt)['state_dict']
cpSD2 = OrderedDict([(k[4:], v) for k, v in cpSD.items()])
model.load_state_dict(cpSD2)

# Set the model in eval mode and push to cuda (if desired)
model.eval()
model.to(device)

#####################
# Get Target Labels #
#####################

print('Getting Target Labels')

# Preprocess (normalize and resize) the input image array and get a PyTorch tensor
featureExtractor = SegformerFeatureExtractor()
imgTens = featureExtractor(imgArr, return_tensors='pt')['pixel_values'].to(device)
imgTens.requires_grad = True

# For the given input image tensor, get the prediction output tensor and convert to numpy
outputTens = model(pixel_values=imgTens)
outputTens = outputTens.logits
outputTens = outputTens.detach().cpu().numpy()


def get_target_label_idx(classType, classLabel, tgtPxH, tgtPxW):
    if classType == 'rank':
        tgtLabelIdx = np.argsort(outputTens[0, :, tgtPxH, tgtPxW])[::-1][classLabel]
    elif classType == 'idx':
        tgtLabelIdx = classLabel


    return tgtLabelIdx


tgtLabelIdx1 = get_target_label_idx(class1Type, class1Label, tgtPxH1, tgtPxW1)
print('  {} {} Image Label at ({},{}): {} | {}'.format(class1Label, class1Type.upper(), tgtPxH1*4, tgtPxW1*4,
                                                       tgtLabelIdx1, id2label[tgtLabelIdx1]))

if class2Type == 'same':
    tgtLabelIdx2 = tgtLabelIdx1
else:
    tgtLabelIdx2 = get_target_label_idx(class2Type, class2Label, tgtPxH2, tgtPxW2)
print('  {} {} Image Label at ({},{}): {} | {}'.format(class2Label, class2Type.upper(), tgtPxH2*4, tgtPxW2*4,
                                                       tgtLabelIdx2, id2label[tgtLabelIdx2]))

####################################
# Integrated Gradients Calculation #
####################################

print('Calculating Label 1 Integrated Gradients')

# Calculate the integrated gradients
attributions1 = run_baseline_integ_gradients(imgArr, model, tgtLabelIdx1, tgtPxH1, tgtPxW1,
                                             steps=nSteps, nRandBaselines=nRandBaselines, device=device)

print('Calculating Label 2 Integrated Gradients')

# Calculate the integrated gradients
attributions2 = run_baseline_integ_gradients(imgArr, model, tgtLabelIdx2, tgtPxH2, tgtPxW2,
                                             steps=nSteps, nRandBaselines=nRandBaselines, device=device)

#########################
# Generate Output Image #
#########################

print('Writing Output Images')

attributions1 = np.transpose(attributions1, (1, 2, 0))
attributions2 = np.transpose(attributions2, (1, 2, 0))
imgArrRS = cv2.resize(imgArr, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

# Generate the integrated gradient images while clipping above/below certain percentiles
imgIntegGradOverlay1 = visualize(attributions1, imgArrRS, clip_above_percentile=99, clip_below_percentile=0, overlay=True)
imgIntegGrad1 = visualize(attributions1, imgArrRS, clip_above_percentile=99, clip_below_percentile=0, overlay=False)

# Generate the integrated gradient images while clipping above/below certain percentiles
imgIntegGradOverlay2 = visualize(attributions2, imgArrRS, clip_above_percentile=99, clip_below_percentile=0, overlay=True)
imgIntegGrad2 = visualize(attributions2, imgArrRS, clip_above_percentile=99, clip_below_percentile=0, overlay=False)

# Combines the original image plus the generated gradients image into one
outputImg = generate_entire_images(imgArrRS, tgtPxH1*4, tgtPxW1*4, tgtPxH2*4, tgtPxW2*4,
                                   imgIntegGrad1, imgIntegGradOverlay1, imgIntegGrad2, imgIntegGradOverlay2,
                                   id2label[tgtLabelIdx1], id2label[tgtLabelIdx2])
cv2.imwrite('results/asdf2' + imgName, np.uint8(outputImg))

print('Done 8]')

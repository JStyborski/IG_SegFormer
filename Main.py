import numpy as np
from PIL import Image
import os
from collections import OrderedDict

import torch

from transformers import SegformerForSemanticSegmentation

from Integrated_Gradients import calculate_outputs, run_baseline_integ_gradients
from Visualization import visualize, generate_entire_images
from Define_Class_Dicts import define_class_dicts

###############
# User Inputs #
###############

print('Starting Code')

imgDir = 'sample_images/img'
imgName = '0006R0_f03480.png'
labelDir = 'sample_images/labels'
labelName = '0006R0_f03480_L.png'

modelCkpt = 'checkpoints/b5_epoch=90-val_mean_iou=0.66.ckpt'
useCuda = True

# Number of random baselines images and the number of steps between baseline and target
# Set nRandBaselines as 'zero' to use the black baseline, set as 'grad' to use scaled gradients (not integ gradients)
nRandBaselines = 5
nSteps = 20

# Target pixel location (relative to the resized 512x512 image)
tgtPxH1 = 276
tgtPxW1 = 120
tgtPxH2 = tgtPxH1
tgtPxW2 = tgtPxW1

# Truth and predicted
# 200: RoadShoulder and Road
# 201: LaneMkgsNonDriv and Road
# 202 and 203: Void and Road
# 204: Road and Road

# Choose 'rank' (e.g., best prediction) or 'idx' (pure class label index) for class types
# Can also choose 'same' for class2Type, which just copies the same target label index as class1 - ignores class2Label
# 'true' to select get the truth label index from the ground truth segmentation image is implemented, but doesn't
# work due to a bug in the opencv2 interpolation method. Even using exact nearest neighbor, there is pixel interpolation
# such that new colors are created
class1Type = 'true'
class1Label = 0
class2Type = 'rank'
class2Label = 0

##############
# Misc Setup #
##############

if class1Type == 'same' or class1Type == 'true':
    class1Label = '-'
if class2Type == 'same' or class2Type == 'true':
    class2Label = '-'

# Function that imports segmentation labels and colors
id2label, label2id, id2color, color2id = define_class_dicts('class_dict.csv')

# Set processing device: cuda or cpu
device = torch.device('cuda' if torch.cuda.is_available() and useCuda else 'cpu')

# Check if results directory and model subfolder exists, if not create it
if not os.path.exists('results/'):
    os.mkdir('results/')

#################
# Image Reading #
#################

print('Reading Image')

# Read the image as PIL then convert to a HxWxRGB numpy array and resize to 512x512
imgPIL = Image.open(os.path.join(imgDir, imgName))
imgArr = np.array(imgPIL)
imgArrRS = np.array(imgPIL.resize(size=(512, 512), resample=Image.Resampling.BILINEAR))

# Read the segmentation annotation image as PIL, resize to 512x512, then convert to a HxWxRGB numpy array
labelArrRS = np.array(Image.open(os.path.join(labelDir, labelName)).resize(size=(512, 512),
                                                                           resample=Image.Resampling.NEAREST))

##################
# Model Creation #
##################

print('Creating Model')

# Instantiate model with label dictionaries
model = SegformerForSemanticSegmentation.from_pretrained('nvidia/mit-b5', num_labels=32, id2label=id2label, label2id=label2id)

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


def get_target_label_idx(classType, classLabel, tgtPxH, tgtPxW):

    # Look at output and get class index by rank
    if classType == 'rank':

        # Get image input tensor and resulting output tensor
        _, outputTens = calculate_outputs(imgArr, model, device)
        outputTens = outputTens.detach().cpu().numpy()

        # Calculate the target label index by looking at the classes of a certain pixel
        tgtLabelIdx = np.argsort(outputTens[0, :, int(tgtPxH / 4), int(tgtPxW / 4)])[::-1][classLabel]

    # Just get the label index from input
    elif classType == 'idx':
        tgtLabelIdx = classLabel

    # Load the truth image and get the corresponding value from there
    elif classType == 'true':
        tgtLabelIdx = color2id[tuple(labelArrRS[tgtPxH, tgtPxW, :])]

    return tgtLabelIdx


tgtLabelIdx1 = get_target_label_idx(class1Type, class1Label, tgtPxH1, tgtPxW1)
print('  {} {} Image Label at ({},{}): {} | {}'.format(class1Label, class1Type.upper(), tgtPxH1, tgtPxW1,
                                                       tgtLabelIdx1, id2label[tgtLabelIdx1]))

if class2Type == 'same':
    tgtLabelIdx2 = tgtLabelIdx1
else:
    tgtLabelIdx2 = get_target_label_idx(class2Type, class2Label, tgtPxH2, tgtPxW2)
print('  {} {} Image Label at ({},{}): {} | {}'.format(class2Label, class2Type.upper(), tgtPxH2, tgtPxW2,
                                                       tgtLabelIdx2, id2label[tgtLabelIdx2]))

####################################
# Integrated Gradients Calculation #
####################################

print('Calculating Label 1 Integrated Gradients')

# Calculate the integrated gradients
attributions1 = run_baseline_integ_gradients(imgArr, model, tgtLabelIdx1, int(tgtPxH1 / 4), int(tgtPxW1 / 4),
                                             steps=nSteps, nRandBaselines=nRandBaselines, device=device)

print('Calculating Label 2 Integrated Gradients')

# Calculate the integrated gradients
attributions2 = run_baseline_integ_gradients(imgArr, model, tgtLabelIdx2, int(tgtPxH2 / 4), int(tgtPxW2 / 4),
                                             steps=nSteps, nRandBaselines=nRandBaselines, device=device)

#########################
# Generate Output Image #
#########################

print('Writing Output Images')

# Swap 3xHxW to HxWx3
attributions1 = np.transpose(attributions1, (1, 2, 0))
attributions2 = np.transpose(attributions2, (1, 2, 0))

# Generate the integrated gradient images while clipping above/below certain percentiles
imgIntegGradOverlay1 = visualize(attributions1, imgArrRS, clip_above_percentile=99, clip_below_percentile=0, overlay=True)
imgIntegGrad1 = visualize(attributions1, imgArrRS, clip_above_percentile=99, clip_below_percentile=0, overlay=False)

# Generate the integrated gradient images while clipping above/below certain percentiles
imgIntegGradOverlay2 = visualize(attributions2, imgArrRS, clip_above_percentile=99, clip_below_percentile=0, overlay=True)
imgIntegGrad2 = visualize(attributions2, imgArrRS, clip_above_percentile=99, clip_below_percentile=0, overlay=False)

# Combines the original image plus the generated gradients image into one
outputImgArr = generate_entire_images(imgArrRS, tgtPxH1, tgtPxW1, tgtPxH2, tgtPxW2,
                                      imgIntegGrad1, imgIntegGradOverlay1, imgIntegGrad2, imgIntegGradOverlay2,
                                      id2label[tgtLabelIdx1], id2label[tgtLabelIdx2])

outputImgPIL = Image.fromarray(outputImgArr.astype(np.uint8))
outputImgPIL.save('results/' + imgName.split('.')[0] + '_' + str(nRandBaselines) + '_' + id2label[tgtLabelIdx1]
                  + '_' + id2label[tgtLabelIdx2] + '_.' + imgName.split('.')[1])

print('Done 8]')

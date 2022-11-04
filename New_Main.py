import numpy as np
import cv2
import os

from torchvision import models

from New_Integrated_Gradients import calculate_outputs, run_baseline_integ_gradients
from Visualization import visualize, generate_entire_images
from ImageNet_Labels import define_imagenet_labels

###############
# User Inputs #
###############

print('Starting Code')

imgDir = 'examples'
imgName = '02.jpg'
modelName = 'resnet18'  # inception, resnet152, resnet18, vgg19
useCuda = True
nRandBaselines = 10 # Number of random baselines images, set as None to use zero baseline
nSteps = 50 # Number of integration steps between baseline and target

# Choose 'rank' (e.g., best prediction) or 'idx' (pure class label index) for class types
class1Type = 'rank'
class1Label = 0
class2Type = 'rank'
class2Label = 200

# Function that imports the ImageNet1k labels as a dict
imageNetLabels = define_imagenet_labels()

##################
# Model Creation #
##################

print('Creating Model')

# Check if results directory and model subfolder exists, if not create it
if not os.path.exists('results/'):
    os.mkdir('results/')
if not os.path.exists('results/' + modelName):
    os.mkdir('results/' + modelName)

# Import model
if modelName == 'inception':
    model = models.inception_v3(weights='DEFAULT')
elif modelName == 'resnet152':
    model = models.resnet152(weights='DEFAULT')
elif modelName == 'resnet18':
    model = models.resnet18(weights='DEFAULT')
elif modelName == 'vgg19':
    model = models.vgg19_bn(weights='DEFAULT')

# Set the model in eval mode and push to cuda (if desired)
model.eval()
if useCuda:
    model.cuda()

#################
# Image Reading #
#################

print('Reading Image')

# Read the image
imgPIL = cv2.imread(os.path.join(imgDir, imgName))

# InceptionNet expects a different image input size
if modelName == 'inception':
    imgPIL = cv2.resize(imgPIL, (299, 299))

# Convert image type and reorder channels
imgArr = imgPIL.astype(np.float32)
imgArr = imgArr[:, :, (2, 1, 0)]

#####################
# Get Target Labels #
#####################

print('Getting Target Labels')

# For the given input image, get the prediction output tensor and convert to numpy
_, outputTens = calculate_outputs(imgArr, model, cuda=useCuda)
outputTens = outputTens.detach().cpu().numpy()


def get_target_label_idx(classType, classLabel):
    if classType == 'rank':
        targetLabelIdx = np.argsort(outputTens[0])[::-1][classLabel]
    elif classType == 'idx':
        targetLabelIdx = classLabel
    print('  {} {} Image Label: {} | {}'.format(classLabel, classType.upper(), targetLabelIdx, imageNetLabels[targetLabelIdx]))

    return targetLabelIdx


targetLabelIdx1 = get_target_label_idx(class1Type, class1Label)
targetLabelIdx2 = get_target_label_idx(class2Type, class2Label)

####################################
# Integrated Gradients Calculation #
####################################

print('Calculating Label 1 Integrated Gradients')

# Calculate the integrated gradients
attributions1 = run_baseline_integ_gradients(imgArr, model, targetLabelIdx1, steps=nSteps,
                                             nRandBaselines=nRandBaselines, cuda=useCuda)

print('Calculating Label 2 Integrated Gradients')

# Calculate the integrated gradients
attributions2 = run_baseline_integ_gradients(imgArr, model, targetLabelIdx2, steps=nSteps,
                                             nRandBaselines=nRandBaselines, cuda=useCuda)
#########################
# Generate Output Image #
#########################

print('Writing Output Images')

# Generate the integrated gradient images while clipping above/below certain percentiles
imgIntegGradOverlay1 = visualize(attributions1, imgArr, clip_above_percentile=99, clip_below_percentile=0, overlay=True)
imgIntegGrad1 = visualize(attributions1, imgArr, clip_above_percentile=99, clip_below_percentile=0, overlay=False)

# Generate the integrated gradient images while clipping above/below certain percentiles
imgIntegGradOverlay2 = visualize(attributions2, imgArr, clip_above_percentile=99, clip_below_percentile=0, overlay=True)
imgIntegGrad2 = visualize(attributions2, imgArr, clip_above_percentile=99, clip_below_percentile=0, overlay=False)

# Combines the original image plus the generated gradients image into one
outputImg = generate_entire_images(imgArr, imgIntegGrad1, imgIntegGradOverlay1, imgIntegGrad2, imgIntegGradOverlay2,
                                   imageNetLabels[targetLabelIdx1], imageNetLabels[targetLabelIdx2])
cv2.imwrite('results/' + modelName + '/' + imgName, np.uint8(outputImg))

print('Done 8]')

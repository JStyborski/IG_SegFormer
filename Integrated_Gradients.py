import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerFeatureExtractor

# Instantiate Segformer image preprocessor
# Given an uint8 input image numpy array, it will return a resized and normalized PyTorch tensor (1x3x512x512)
# reduce_labels is not included (such that it defaults to False) - Unlike ADE20k, we don't have a background label
featureExtractor = SegformerFeatureExtractor()


# Between a baseline image and the target image, calculate the intermediate gradient values and discretely integrate
def integ_gradients(imgArr, model, targetLabelIdx, targetPxH, targetPxW, baseline, steps, device):

    # imgArr is a  numpy array (224x224x3) representing a PIL image
    # model is the pytorch model
    # targetLabelIdx is the index from which to calculate gradients
    # targetPxH and targetPxW are the target pixel height and width indices
    # baseline is the baseline image numpy array (224x224x3)
    # steps is the number of linearly interpolated images to calculate between baseline and imgArr
    # device is the torch.device

    # This is a list of the image arrays representing the steps along a linear path between baseline and input images
    imgDelta = imgArr - baseline
    scaledInpList = [baseline + float(i) / steps * imgDelta for i in range(0, steps + 1)]

    # Compute the backprop gradients of the output classes wrt to the given scaled input images
    # Make gradsArr as a steps+1x3x224x224 numpy array
    gradsArrList = []
    for inp in scaledInpList:

        # Get the tensor (1x3x512x512) output from the SegFormer preprocessor
        imgTens = featureExtractor(imgArr, return_tensors='pt')['pixel_values'].to(device)
        imgTens.requires_grad = True

        # Evaluate the model and get classifier outputs
        outputTens = model(pixel_values=imgTens)
        outputTens = outputTens.logits

        # Gathers the result at the given targetLabelIdx and pixel location
        outputTens = outputTens[0, targetLabelIdx, targetPxH, targetPxW]

        # Backprop the model to get the gradient of the output class prediction wrt the input image
        model.zero_grad()
        outputTens.backward()
        gradsArr = imgTens.grad.detach().cpu().numpy().squeeze()

        gradsArrList.append(gradsArr)

    gradsArrList = np.array(gradsArrList)

    # Explanation: We have calculated our gradients at 51 (or other) points from baseline to input. This is like f(x)
    # Now we seek to integrate f(x) from baseline to input, so discretely we do a Riemann sum
    # Since we have a constant interval (imgDelta), we can just do avg(f(x)) * (image-baseline)
    # The line below gives the avg(f(x)) using a central Riemann sum.
    # e.g., for 5 steps 0 to 5: avg(f(x)) ~ ((f(0) + f(1)) / 2 + (f(1) + f(2)) / 2 + ... + (f(4) + f(5)) / 2) / 5
    # Equivalently, as in the format below: (f(1) + f(2) + f(3) + f(4) + (f(0) + f(5)) / 2) / 5
    avgGrads = (np.sum(gradsArrList[1:-1], axis=0) + (gradsArrList[0] + gradsArrList[-1]) / 2) \
               / (gradsArrList.shape[0] - 1)

    # deltaX is the difference between normalized imgArr and baseline images on 0-1 scale
    # This is the total difference between the images (normalized)
    prepImgArr = featureExtractor(imgArr, return_tensors='np')['pixel_values'].squeeze()
    prepBaseline = featureExtractor(baseline, return_tensors='np')['pixel_values'].squeeze()
    deltaX = prepImgArr - prepBaseline

    # Final multiplication of the central Riemann sum
    integGrad = deltaX * avgGrads

    return integGrad


# Generate a number of random baselines and calculate integrated_gradients for each, then average the results
def run_baseline_integ_gradients(imgArr, model, targetLabelIdx, targetPxH, targetPxW, steps, nRandBaselines, device):

    # imgArr is a  numpy array (224x224x3) representing a PIL image
    # model is the pytorch model
    # targetLabelIdx is the index from which to calculate gradients
    # targetPxH and targetPxW are the target pixel height and width indices
    # steps is the number of linearly interpolated images to calculate between baseline and imgArr
    # nRandBaselines is the number of randomized baseline images to use and average over. If None, use 0s baseline
    # device is the torch.device

    # Used to collect the integrated gradient results
    allIntegGradsList = []

    # Run integrated gradients with a random baseline image and accumulate the results in allIntegGradsList
    # If no baseline specified, use 0's array as baseline, else run with n baselines
    if nRandBaselines is None:
        print('  Zero Baseline')
        integGrad = integ_gradients(imgArr, model, targetLabelIdx, targetPxH, targetPxW,
                                    baseline=0.0 * imgArr, steps=steps, device=device)
        allIntegGradsList.append(integGrad)
    else:
        for i in range(nRandBaselines):
            print('  Random Baseline {}'.format(i))
            integGrad = integ_gradients(imgArr, model, targetLabelIdx, targetPxH, targetPxW,
                                        baseline=255.0 * np.random.random(imgArr.shape), steps=steps, device=device)
            allIntegGradsList.append(integGrad)

    # Average the integrated gradients across all random baselines
    avgIntegGrads = np.average(np.array(allIntegGradsList), axis=0)

    return avgIntegGrads

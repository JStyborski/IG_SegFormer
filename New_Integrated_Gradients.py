import numpy as np
import torch
import torch.nn.functional as F


# Function to normalize an input image array
def preprocessing(imgArr):

    # imgArr is a single numpy array (224x224x3) representing a PIL image

    # mean and stddev for normalizing images
    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
    std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])

    # Rescale uint8 to 0-1 and normalize
    imgArr = imgArr / 255
    imgArr = (imgArr - mean) / std

    return imgArr


# Converts an input image to a PyTorch tensor and pushes it through the model to get an output tensor
def calculate_outputs(imgArr, model, cuda=False):

    # imgArr is a  numpy array (224x224x3) representing a PIL image
    # model is the pytorch model
    # cuda is a Boolean of using GPU or not

    # Normalize image and return pytorch tensor
    imgArr = preprocessing(imgArr)

    # Reshape 224x224x3 -> 3x224x224 -> 1x3x224x224 to get in pytorch tensor format
    imgArr = np.transpose(imgArr, (2, 0, 1))
    imgArr = np.expand_dims(imgArr, 0)

    # Push image array to cuda or cpu as torch tensor
    if cuda:
        torch_device = torch.device('cuda:0')
    else:
        torch_device = torch.device('cpu')
    inputTens = torch.tensor(imgArr, dtype=torch.float32, device=torch_device, requires_grad=True)

    # Evaluate the model and classifier
    outputTens = model(inputTens)
    outputTens = F.softmax(outputTens, dim=1)

    return inputTens, outputTens


# Calculates the gradient of the output label wrt the input image
def calculate_gradients(inputTens, outputTens, model, targetLabelIdx, cuda=False):

    # inputTens and outputTens are the torch image input and torch softmax output tensors
    # model is the pytorch model
    # targetLabelIdx is the index from which to calculate gradients
    # cuda is a Boolean of using GPU or not

    # Creates a tensor of the target label indices and then gather the softmax result there
    # JS Note: I think output is always a 1x1 tensor, there is no image batching here, outputTens is always 1x1000
    # so index is a 1x1 numpy array and outputTens is gathered to a 1x1 tensor
    index = np.ones((outputTens.size()[0], 1)) * targetLabelIdx
    index = torch.tensor(index, dtype=torch.int64)
    if cuda:
        index = index.cuda()
    outputTens = outputTens.gather(1, index)
    # I think the above lines are equivalent to: outputTens = outputTens[0, targetLabelIdx] and then outputTens.backward()

    # Backprop the model to get the gradient of the output class prediction wrt the input image
    model.zero_grad()
    outputTens.backward()
    gradients = inputTens.grad.detach().cpu().numpy()

    # 1x3x224x224 -> 224x224x3
    gradients = np.transpose(gradients[0], (1, 2, 0))

    return gradients


# Between a baseline image and the target image, calculate the intermediate gradient values and discretely integrate
def integ_gradients(imgArr, model, targetLabelIdx, baseline, steps=50, cuda=False):

    # imgArr is a  numpy array (224x224x3) representing a PIL image
    # model is the pytorch model
    # targetLabelIdx is the index from which to calculate gradients
    # baseline is the baseline image numpy array (224x224x3)
    # steps is the number of linearly interpolated images to calculate between baseline and imgArr
    # cuda is a Boolean of using GPU or not

    # This is a list of the image arrays representing the steps along a linear path between baseline and input images
    imgDelta = imgArr - baseline
    scaledInpList = [baseline + float(i) / steps * imgDelta for i in range(0, steps + 1)]

    # Compute the backprop gradients of the output classes wrt to the given scaled input images
    # Make gradsArr as a steps+1x224x224x3 numpy array
    gradsArr = []
    for inp in scaledInpList:
        inputTens, outputTens = calculate_outputs(inp, model, cuda)
        grads = calculate_gradients(inputTens, outputTens, model, targetLabelIdx, cuda)
        gradsArr.append(grads)
    gradsArr = np.array(gradsArr)

    # Explanation: We have calculated our gradients at 51 (or other) points from baseline to input. This is like f(x)
    # Now we seek to integrate f(x) from baseline to input, so discretely we do a Riemann sum
    # Since we have a constant interval (imgDelta), we can just do avg(f(x)) * (image-baseline)
    # The line below gives the avg(f(x)) using a central Riemann sum.
    # e.g., for 5 steps 0 to 5: avg(f(x)) ~ ((f(0) + f(1)) / 2 + (f(1) + f(2)) / 2 + ... + (f(4) + f(5)) / 2) / 5
    # Equivalently, as in the format below: (f(1) + f(2) + f(3) + f(4) + (f(0) + f(5)) / 2) / 5
    avgGrads = (np.sum(gradsArr[1:-1], axis=0) + (gradsArr[0] + gradsArr[-1]) / 2) / (gradsArr.shape[0] - 1)

    # deltaX is the difference between normalized imgArr and baseline images on 0-1 scale
    # This is the total difference between the images (normalized)
    deltaX = preprocessing(imgArr) - preprocessing(baseline)

    # Final multiplication of the central Riemann sum
    integGrad = deltaX * avgGrads

    return integGrad


# Generate a number of random baselines and calculate integrated_gradients for each, then average the results
def run_baseline_integ_gradients(imgArr, model, targetLabelIdx, steps, nRandBaselines, cuda):

    # imgArr is a  numpy array (224x224x3) representing a PIL image
    # model is the pytorch model
    # targetLabelIdx is the index from which to calculate gradients
    # steps is the number of linearly interpolated images to calculate between baseline and imgArr
    # nRandBaselines is the number of randomized baseline images to use and average over. If None, use 0s baseline
    # cuda is a Boolean of using GPU or not

    # Used to collect the integrated gradient results
    allIntegGradsList = []

    # Run integrated gradients with a random baseline image and accumulate the results in allIntegGradsList
    # If no baseline specified, use 0's array as baseline, else run with n baselines
    if nRandBaselines is None:
        print('  Zero Baseline')
        integGrad = integ_gradients(imgArr, model, targetLabelIdx, baseline=0.0 * imgArr, steps=steps, cuda=cuda)
        allIntegGradsList.append(integGrad)
    else:
        for i in range(nRandBaselines):
            print('  Random Baseline {}'.format(i))
            integGrad = integ_gradients(imgArr, model, targetLabelIdx,
                                        baseline=255.0 * np.random.random(imgArr.shape), steps=steps, cuda=cuda)
            allIntegGradsList.append(integGrad)

    # Average the integrated gradients across all random baselines
    avgIntegGrads = np.average(np.array(allIntegGradsList), axis=0)

    return avgIntegGrads

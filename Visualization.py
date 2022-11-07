import numpy as np
import cv2


# Converting to grayscale is the same as averaging across the image's color channels
def convert_to_gray_scale(attributions):
    return np.average(attributions, axis=2)


# Given an image, rescale it such that the values at the specified percentiles are 0 and 1, clip all others to 0, 1
def linear_transform(attributions, clip_above_percentile=99.9, clip_below_percentile=70.0):

    # JS Note: There was originally an input variable called "low" that shifted the transformed array
    # low was always set equal 0.0, which meant it did nothing, so I removed it

    # Get the min/max threshold intensity values given the percentiles
    m = compute_threshold_by_top_percentage(attributions, percentile=clip_above_percentile)
    e = compute_threshold_by_top_percentage(attributions, percentile=clip_below_percentile)

    # Rescale the intensities such that m=1 and e=0
    transformed = (np.abs(attributions) - e) / (m - e)

    # Reapply the attribution signs to the scaled values and clip any resulting values below 0 or above 1
    transformed *= np.sign(attributions)
    transformed = np.clip(transformed, 0.0, 1.0)
    return transformed


# Given a HxW array, find the intensity value at the given percentile
def compute_threshold_by_top_percentage(attributions, percentile=60):

    # attributions is the HxW numpy array of grayscaled gradients
    # percentile is some value between 0-100

    # Handle percentile input issues
    if percentile < 0 or percentile > 100:
        raise ValueError('percentage must be in [0, 100]')
    if percentile == 0:
        return np.min(attributions)
    if percentile == 100:
        return np.max(attributions)

    # Flatten the numpy image array, sort it (asc order), and cumsum to find the intensity value at the given percentile
    flatAttributions = attributions.flatten()
    sortedAttributions = np.sort(np.abs(flatAttributions))
    cumSum = 100.0 * np.cumsum(sortedAttributions) / np.sum(flatAttributions)
    thresholdIdx = np.where(cumSum >= percentile)[0][0]
    threshold = sortedAttributions[thresholdIdx]

    return threshold


# This function clips the gradient array inputs and outputs them as plottable images
def visualize(attributions, imgArr, clip_above_percentile=99.9, clip_below_percentile=0, overlay=True):

    # attributions are the gradients, a HxWxRGB numpy array
    # imgArr is the original image array, also a HxWxRGB numpy array
    # clip_above/below_percentile are used to clamp negative and high pixels

    # JS Note: To simplify, I removed 9 unused variables
    # positive_channel and polarity were implemented but no scripts called this function with those values specified
    # negative_channel, structure, and outlines_component_percentage were never even called
    # plot_distribution, morphological_cleanup, and outlines were removed because their loops were never implemented
    # mask_mode because it was redundant with the overlay variable

    # Clip gradients to 0-1
    attributions = np.clip(attributions, 0, 1)
    
    # Convert the attributions to the gray scale (HxWxRGB -> HxW) and clip values above and below the specified
    # percentile values (linear_transform will rescale to 0,1) and expand back to HxWx1
    attributions = convert_to_gray_scale(attributions)
    attributions = linear_transform(attributions, clip_above_percentile, clip_below_percentile)
    attributions = np.expand_dims(attributions, 2)

    # Method for rescaling the image back to 0-255: use to scale original image or just output on green channel
    if overlay:
        attributions = np.clip(attributions * imgArr, 0, 255)
    else:
        attributions = attributions * [0, 255, 0]

    return attributions

def draw_square(imgArr, centerCoordH, centerCoordW):

    # Note that the red is applied in BGR

    for i in range(-4, 5):
        for j in range(-4, 5):
            imgArr[centerCoordH + i, centerCoordW + j, :] = [0, 0, 255]

    return imgArr

# This function just combines 5 numpy images into a single image - a heuristic way of plotting 5 images as 1
def generate_entire_images(imgOrigArr, tgtPxH1, tgtPxW1, tgtPxH2, tgtPxW2, integGrad1, integGrad1Overlay, integGrad2, integGrad2Overlay, imageNetLabel1, imageNetLabel2):

    # The rest of the script (Main.py and this one) work with PIL or numpy arrays that are in RGB format, as in the
    # original SegFormer code. The original IG code works with cv2, which uses BGR. I converted most of the code to
    # use PIL, arrays, and RGB, but due to a lot of specific code here I just use cv2 and swap to BGR temporarily.
    imgOrigArr = imgOrigArr[:, :, (2, 1, 0)]
    integGrad1 = integGrad1[:, :, (2, 1, 0)]
    integGrad1Overlay = integGrad1Overlay[:, :, (2, 1, 0)]
    integGrad2 = integGrad2[:, :, (2, 1, 0)]
    integGrad2Overlay = integGrad2Overlay[:, :, (2, 1, 0)]

    # Vertical and horizontal white spaces between images
    blank = np.ones((integGrad1.shape[0], 10, 3), dtype=np.uint8) * 128
    blank_hor = np.ones((10, 20 + integGrad1.shape[0] * 3, 3), dtype=np.uint8) * 128

    # Rows are concatenations of images and white spaces
    upper = np.concatenate([draw_square(imgOrigArr, tgtPxH1, tgtPxW1), blank, integGrad1Overlay, blank, integGrad1], 1)
    middle = np.concatenate([draw_square(imgOrigArr, tgtPxH2, tgtPxW2), blank, integGrad2Overlay, blank, integGrad2], 1)
    lower = np.concatenate([imgOrigArr, blank, np.abs(integGrad1Overlay - integGrad2Overlay), blank,
                           np.abs(integGrad1 - integGrad2)], 1)
    total = np.concatenate([upper, blank_hor, middle, blank_hor, lower], 0)
    total = cv2.resize(total, (1000, 1000))

    # Add text
    cv2.putText(total, 'Class: ' + imageNetLabel1, (5, 25), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(0,0,255))
    cv2.putText(total, 'Class: ' + imageNetLabel2, (5, 360), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(0,0,255))
    cv2.putText(total, 'Overlay IG', (340, 25), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(0,0,255))
    cv2.putText(total, 'Pure IG', (675, 25), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(0,0,255))
    cv2.putText(total, 'Overlay IG Diff', (340, 700), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(0,0,255))
    cv2.putText(total, 'Pure IG Diff', (675, 700), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(0,0,255))

    # Convert output from BGR to RGB
    total = total[:, :, (2, 1, 0)]

    return total

from __future__ import print_function
from PIL import Image
from glob import glob

import numpy as np
import torch
from subprocess import check_output
import os
import psutil
import math

import matplotlib as mpl
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn

from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes


colors = torch.tensor([[  0,   0,   0], # Black
                       [255,   0,   0], # Red
                       [  0, 255,   0], # Green
                       [  0,   0, 255], # Blue
                       [  0, 255, 255], # Cyan
                       [255,   0, 255], # Magenta
                       [255, 255,   0], # Yellow
                       [139,  69,  19], # Brown (saddlebrown)
                       [128,   0, 128], # Purple
                       [255, 140,   0], # Orange
                       [255, 255, 255]], dtype=torch.uint8) # White






def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def get_starting_iteration(opt):
    model_list = glob(os.path.join(opt.checkpoints_dir,'*.pth'))
    if not len(model_list) == 0:
        model_list.sort()
        start_iter = int(model_list[-1][-13])
        opt.logger.info("Loading former Network at Iteration " + str(start_iter))
    else:
        opt.logger.info("Starting new network")
        return 1
    return start_iter


def visualizeForwardsNoGrad(model, iteration, tbWriter, phase='Train'):
    with torch.no_grad():
        for label, image in model.get_current_visuals().items():
            if image is not None:
                if label == 'segm_B' or label == 'rec_B_Segm' or label == 'idt_B_Segm':
                    labelMap = torch.argmax(image[0], 0).byte() * (255 / 7)
                    tbWriter.add_image(phase + '/' + label, torch.stack([labelMap, labelMap, labelMap], dim=0), iteration)
                else:
                    tbWriter.add_image(phase + '/' + label, torch.round((image[0] + 1.) / 2.0 * 255.0).byte(), iteration)

def printDict(dict):
    s = ""
    sum = 0
    for label, key in dict.items():
        s += label + ": {},\t".format(round(key,4))
        sum += key
    return s + "Mean: {}".format(round(sum/len(dict),4))


def generate_ball(radius):
    structure = np.zeros((3, 3), dtype=np.int)
    structure[1, :] = 1
    structure[:, 1] = 1

    ball = np.zeros((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    ball[radius, radius] = 1
    for i in range(radius):
        ball = binary_dilation(ball, structure=structure)
    return np.asarray(ball, dtype=np.int)


def saveIHCTranslation_PredictionOverlay(stain_img, translated_img, network_output, instance_pred, instance_gt, savePath, tubuliInstanceID_StartsWith, saveImageResultsSeparately, figHeight, alpha=0.4):
    colorMap = np.array([[0, 0, 0],  # Black
                           [255, 0, 0],  # Red
                           [0, 255, 0],  # Green
                           [0, 0, 255],  # Blue
                           [0, 255, 255],  # Cyan
                           [255, 0, 255],  # Magenta
                           [255, 255, 0],  # Yellow
                           [139, 69, 19],  # Brown (saddlebrown)
                           [128, 0, 128],  # Purple
                           [255, 140, 0],  # Orange
                           [255, 255, 255]], dtype=np.uint8)  # White

    newRandomColors = np.random.randint(low=0, high=256, dtype=np.uint8, size=(max(instance_pred.max(), instance_gt.max()), 3))
    colorMap = np.concatenate((colorMap, newRandomColors))
    colorMap = colorMap / 255.
    customColorMap = mpl.colors.ListedColormap(colorMap)
    max_number_of_labels = len(colorMap)

    instance_pred_Mask = np.ma.masked_where(instance_pred == 0, instance_pred)

    pixelShiftX = (translated_img.shape[0]-instance_pred_Mask.shape[0])//2 + 1
    pixelShiftY = (translated_img.shape[1]-instance_pred_Mask.shape[1])//2 + 1

    plt.figure(figsize=(figHeight*2.5, figHeight))
    plt.subplot(251)
    plt.imshow(stain_img)
    plt.axis('off')
    plt.subplot(252)
    plt.imshow(translated_img)
    plt.axis('off')
    plt.subplot(253)
    plt.imshow(stain_img[pixelShiftX:pixelShiftX+instance_pred_Mask.shape[0], pixelShiftY:pixelShiftY+instance_pred_Mask.shape[1],:3])
    plt.axis('off')
    plt.subplot(254)
    plt.imshow(translated_img[pixelShiftX:pixelShiftX+instance_pred_Mask.shape[0], pixelShiftY:pixelShiftY+instance_pred_Mask.shape[1],:3])
    plt.axis('off')
    plt.subplot(255)
    plt.imshow(network_output, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1)
    plt.axis('off')
    plt.subplot(256)
    finalPredictionNonInstance = instance_pred.copy()
    finalPredictionNonInstance[finalPredictionNonInstance >= tubuliInstanceID_StartsWith] = 1
    plt.imshow(finalPredictionNonInstance, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1)
    plt.axis('off')
    plt.subplot(257)
    finalPredictionNonInstance_Mask = np.ma.masked_where(finalPredictionNonInstance == 0, finalPredictionNonInstance)
    plt.imshow(translated_img[pixelShiftX:pixelShiftX+instance_pred_Mask.shape[0], pixelShiftY:pixelShiftY+instance_pred_Mask.shape[1],:3])
    plt.imshow(finalPredictionNonInstance_Mask, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1, alpha=alpha)
    plt.axis('off')
    plt.subplot(258)
    plt.imshow(instance_pred, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1)
    plt.axis('off')
    plt.subplot(259)
    plt.imshow(translated_img[pixelShiftX:pixelShiftX+instance_pred_Mask.shape[0], pixelShiftY:pixelShiftY+instance_pred_Mask.shape[1],:3])
    plt.imshow(instance_pred_Mask, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1, alpha=alpha)
    plt.axis('off')
    plt.subplot(2,5,10)
    plt.imshow(instance_gt, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1)
    plt.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(savePath)
    plt.close()

    if saveImageResultsSeparately:
        fileName = savePath.split('/')[-1][:-4]
        savePathName = "/".join(savePath.split('/')[:-1])+"/separateImages"
        if not os.path.exists(savePathName):
            os.makedirs(savePathName)

        plt.figure(figsize=(10, 10))
        plt.imshow(stain_img)
        plt.axis('off')
        plt.savefig(savePathName + '/' + fileName + '_1.png', bbox_inches = 'tight', pad_inches = 0)

        plt.imshow(translated_img)
        plt.axis('off')
        plt.savefig(savePathName + '/' + fileName + '_2.png', bbox_inches = 'tight', pad_inches = 0)

        plt.imshow(stain_img[pixelShiftX:pixelShiftX+instance_pred_Mask.shape[0], pixelShiftY:pixelShiftY+instance_pred_Mask.shape[1],:3])
        plt.axis('off')
        plt.savefig(savePathName + '/' + fileName + '_3.png', bbox_inches = 'tight', pad_inches = 0)

        plt.imshow(translated_img[pixelShiftX:pixelShiftX+instance_pred_Mask.shape[0], pixelShiftY:pixelShiftY+instance_pred_Mask.shape[1],:3])
        plt.axis('off')
        plt.savefig(savePathName + '/' + fileName + '_4.png', bbox_inches = 'tight', pad_inches = 0)

        plt.imshow(network_output, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1)
        plt.axis('off')
        plt.savefig(savePathName + '/' + fileName + '_5.png', bbox_inches = 'tight', pad_inches = 0)

        finalPredictionNonInstance = instance_pred.copy()
        finalPredictionNonInstance[finalPredictionNonInstance >= tubuliInstanceID_StartsWith] = 1
        plt.imshow(finalPredictionNonInstance, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1)
        plt.axis('off')
        plt.savefig(savePathName + '/' + fileName + '_6.png', bbox_inches = 'tight', pad_inches = 0)

        finalPredictionNonInstance_Mask = np.ma.masked_where(finalPredictionNonInstance == 0, finalPredictionNonInstance)
        plt.imshow(translated_img[pixelShiftX:pixelShiftX+instance_pred_Mask.shape[0], pixelShiftY:pixelShiftY+instance_pred_Mask.shape[1],:3])
        plt.imshow(finalPredictionNonInstance_Mask, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1, alpha=alpha)
        plt.axis('off')
        plt.savefig(savePathName + '/' + fileName + '_7.png', bbox_inches = 'tight', pad_inches = 0)

        plt.imshow(instance_pred, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1)
        plt.axis('off')
        plt.savefig(savePathName + '/' + fileName + '_8.png', bbox_inches = 'tight', pad_inches = 0)

        plt.imshow(translated_img[pixelShiftX:pixelShiftX+instance_pred_Mask.shape[0], pixelShiftY:pixelShiftY+instance_pred_Mask.shape[1],:3])
        plt.imshow(instance_pred_Mask, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1, alpha=alpha)
        plt.axis('off')
        plt.savefig(savePathName + '/' + fileName + '_9.png', bbox_inches = 'tight', pad_inches = 0)

        plt.imshow(instance_gt, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1)
        plt.axis('off')
        plt.savefig(savePathName + '/' + fileName + '_10.png', bbox_inches = 'tight', pad_inches = 0)

        instance_gt_copy = instance_gt.copy()
        instance_gt_copy[instance_gt_copy >= tubuliInstanceID_StartsWith] = 1
        plt.imshow(instance_gt_copy, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1)
        plt.axis('off')
        plt.savefig(savePathName + '/' + fileName + '_11.png', bbox_inches = 'tight', pad_inches = 0)

        plt.close()



def convert_labelmap_to_rgb(labelmap):
    """
    Method used to generate rgb label maps for tensorboard visualization
    :param labelmap: HxW label map tensor containing values from 0 to n_classes
    :return: 3xHxW RGB label map containing colors in the following order: Black (background), Red, Green, Blue, Cyan, Magenta, Yellow, Brown, Orange, Purple
    """
    n_classes = labelmap.max()

    result = torch.zeros(size=(labelmap.size()[0], labelmap.size()[1], 3), dtype=torch.uint8)
    for i in range(1, n_classes+1):
        result[labelmap == i] = colors[i]

    return result.permute(2, 0, 1)

def convert_labelmap_to_rgb_with_instance_first_class(labelmap, structure):
    """
    Method used to generate rgb label maps for tensorboard visualization
    :param labelmap: HxW label map tensor containing values from 0 to n_classes
    :return: 3xHxW RGB label map containing colors in the following order: Black (background), Red, Green, Blue, Cyan, Magenta, Yellow, Brown, Orange, Purple
    """
    n_classes = labelmap.max()

    result = np.zeros(shape=(labelmap.shape[0], labelmap.shape[1], 3), dtype=np.uint8)
    for i in range(2, n_classes+1):
        result[labelmap == i] = colors[i].numpy()

    structure = np.ones((3, 3), dtype=np.int)

    labeledTubuli, numberTubuli = label(np.asarray(labelmap == 1, np.uint8), structure)  # datatype of 'labeledTubuli': int32
    for i in range(1, numberTubuli + 1):
        result[binary_dilation(binary_dilation(binary_dilation(labeledTubuli == i)))] = np.random.randint(low=0, high=256, size=3, dtype=np.uint8)  # assign random colors to tubuli

    return result

def convert_labelmap_to_rgb_except_first_class(labelmap):
    """
    Method used to generate rgb label maps for tensorboard visualization
    :param labelmap: HxW label map tensor containing values from 0 to n_classes
    :return: 3xHxW RGB label map containing colors in the following order: Black (background), Red, Green, Blue, Cyan, Magenta, Yellow, Brown, Orange, Purple
    """
    n_classes = labelmap.max()

    result = torch.zeros(size=(labelmap.size()[0], labelmap.size()[1], 3), dtype=torch.uint8)
    for i in range(2, n_classes+1):
        result[labelmap == i] = colors[i]

    return result.permute(2, 0, 1)


def getColorMapForLabelMap():
    return ['black', 'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'brown', 'purple', 'orange', 'white']

def saveFigureResults(img, outputPrediction, postprocessedPrediction, finalPredictionRGB, GT, preprocessedGT, preprocessedGTrgb, fullResultPath, alpha=0.4):
    customColors = getColorMapForLabelMap()
    max_number_of_labels = len(customColors)
    assert outputPrediction.max() < max_number_of_labels, 'Too many labels -> Not enough colors available in custom colormap! Add some colors!'
    customColorMap = mpl.colors.ListedColormap(getColorMapForLabelMap())

    # avoid brown color (border visualization) in output for final GT and prediction
    postprocessedPrediction[postprocessedPrediction==7] = 0
    preprocessedGT[preprocessedGT==7] = 0

    # also dilate tubuli here
    postprocessedPrediction[binary_dilation(postprocessedPrediction==1)] = 1

    predictionMask = np.ma.masked_where(postprocessedPrediction == 0, postprocessedPrediction)

    plt.figure(figsize=(16, 8.1))
    plt.subplot(241)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(outputPrediction, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1)
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(postprocessedPrediction, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1)
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(finalPredictionRGB)
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(img[(img.shape[0]-outputPrediction.shape[0])//2:(img.shape[0]-outputPrediction.shape[0])//2+outputPrediction.shape[0],(img.shape[1]-outputPrediction.shape[1])//2:(img.shape[1]-outputPrediction.shape[1])//2+outputPrediction.shape[1],:])
    plt.imshow(predictionMask, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1, alpha = alpha)
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(GT, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1)
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(preprocessedGT, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1)
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(preprocessedGTrgb)
    plt.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(fullResultPath)
    plt.close()

def savePredictionResults(predictionWithoutTubuliDilation, fullResultPath, figSize):
    prediction = predictionWithoutTubuliDilation.copy()
    prediction[binary_dilation(binary_dilation(binary_dilation(binary_dilation(prediction == 1))))] = 1

    customColors = getColorMapForLabelMap()
    max_number_of_labels = len(customColors)
    assert prediction.max() < max_number_of_labels, 'Too many labels -> Not enough colors available in custom colormap! Add some colors!'
    customColorMap = mpl.colors.ListedColormap(getColorMapForLabelMap())

    fig = plt.figure(figsize=figSize)
    ax = plt.Axes(fig, [0., 0., 1., 1., ])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(prediction, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1)
    plt.savefig(fullResultPath)
    plt.close()

def savePredictionResultsWithoutDilation(prediction, fullResultPath, figSize):
    customColors = getColorMapForLabelMap()
    max_number_of_labels = len(customColors)
    assert prediction.max() < max_number_of_labels, 'Too many labels -> Not enough colors available in custom colormap! Add some colors!'
    customColorMap = mpl.colors.ListedColormap(getColorMapForLabelMap())

    fig = plt.figure(figsize=figSize)
    ax = plt.Axes(fig, [0., 0., 1., 1., ])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(prediction, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1)
    plt.savefig(fullResultPath)
    plt.close()

def savePredictionOverlayResults(img, prediction, fullResultPath, figSize, alpha=0.4):
    predictionMask = np.ma.masked_where(prediction == 0, prediction)

    colorMap = np.array([[0, 0, 0],  # Black
                           [255, 0, 0],  # Red
                           [0, 255, 0],  # Green
                           [0, 0, 255],  # Blue
                           [0, 255, 255],  # Cyan
                           [255, 0, 255],  # Magenta
                           [255, 255, 0],  # Yellow
                           [139, 69, 19],  # Brown (saddlebrown)
                           [128, 0, 128],  # Purple
                           [255, 140, 0],  # Orange
                           [255, 255, 255]], dtype=np.uint8)  # White

    newRandomColors = np.random.randint(low=0, high=256, dtype=np.uint8, size=(prediction.max(), 3))
    colorMap = np.concatenate((colorMap, newRandomColors))
    colorMap = colorMap / 255.

    max_number_of_labels = len(colorMap)
    assert prediction.max() < max_number_of_labels, 'Too many labels -> Not enough colors available in custom colormap! Add some colors!'
    customColorMap = mpl.colors.ListedColormap(colorMap)

    fig = plt.figure(figsize=figSize)
    ax = plt.Axes(fig, [0., 0., 1., 1., ])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img)
    ax.imshow(predictionMask, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1, alpha = alpha)
    plt.savefig(fullResultPath)
    plt.close()

def saveOverlayResults(img, seg, fullResultPath, figHeight, alpha=0.4):
    segMask = np.ma.masked_where(seg == 0, seg)

    customColors = getColorMapForLabelMap()
    max_number_of_labels = len(customColors)
    assert seg.max() < max_number_of_labels, 'Too many labels -> Not enough colors available in custom colormap! Add some colors!'
    customColorMap = mpl.colors.ListedColormap(getColorMapForLabelMap())

    fig = plt.figure(figsize=(figHeight*seg.shape[1]/seg.shape[0], figHeight))
    ax = plt.Axes(fig, [0., 0., 1., 1., ])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img)
    ax.imshow(segMask, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1, alpha = alpha)
    plt.savefig(fullResultPath)
    plt.close()

def saveRGBPredictionOverlayResults(img, prediction, fullResultPath, figSize, alpha=0.4):
    predictionMask = prediction.sum(2)==0
    predictionCopy = prediction.copy()
    predictionCopy[predictionMask] = img[predictionMask]
    fig = plt.figure(figsize=figSize)
    ax = plt.Axes(fig, [0., 0., 1., 1., ])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(np.asarray(np.round(predictionCopy*alpha+(1-alpha)*img), np.uint8))
    plt.savefig(fullResultPath)
    plt.close()

def saveImage(img, fullResultPath, figSize):
    fig = plt.figure(figsize=figSize)
    ax = plt.Axes(fig, [0., 0., 1., 1., ])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img)
    plt.savefig(fullResultPath)
    plt.close()



def getCrossValSplits(dataIDX, amountFolds, foldNo, setting):
    """
    Cross-Validation-Split of indices according to fold number and setting
    Usage:
        dataIDX = np.arange(dataset.__len__())
        # np.random.shuffle(dataIDX)
        for i in range(amountFolds):
            train_idx, val_idx, test_idx = getCrossFoldSplits(dataIDX=dataIDX, amountFolds=amountFolds, foldNo=i+1, setting=setting)
    :param dataIDX: All data indices stored in numpy array
    :param amountFolds: Total amount of folds
    :param foldNo: Fold number, # CARE: Fold numbers start with 1 and go up to amountFolds ! #
    :param setting: Train / Train+Test / Train+Val / Train+Test+Val
    :return: tuple consisting of 3 numpy arrays (trainIDX, valIDX, testIDX) containing indices according to split
    """
    assert (setting in ['train_val_test', 'train_test', 'train_val', 'train']), 'Given setting >'+setting+'< is incorrect!'

    num_total_data = dataIDX.__len__()

    if setting == 'train':
        return dataIDX, None, None

    elif setting == 'train_val':
        valIDX = dataIDX[num_total_data * (foldNo - 1) // amountFolds: num_total_data * foldNo // amountFolds]
        trainIDX = np.setxor1d(dataIDX, valIDX)
        return trainIDX, valIDX, None

    elif setting == 'train_test':
        testIDX = dataIDX[num_total_data * (foldNo - 1) // amountFolds: num_total_data * foldNo // amountFolds]
        trainIDX = np.setxor1d(dataIDX, testIDX)
        return trainIDX, None, testIDX

    elif setting == 'train_val_test':
        testIDX = dataIDX[num_total_data * (foldNo - 1) // amountFolds: num_total_data * foldNo // amountFolds]
        if foldNo != amountFolds:
            valIDX = dataIDX[num_total_data * foldNo // amountFolds: num_total_data * (foldNo+1) // amountFolds]
        else:
            valIDX = dataIDX[0 : num_total_data // amountFolds]
        trainIDX = np.setxor1d(np.setxor1d(dataIDX, testIDX), valIDX)
        return trainIDX, valIDX, testIDX

    else:
        raise ValueError('Given setting >'+str(setting)+'< is invalid!')


def parse_nvidia_smi(unit=0):
    result = check_output(["nvidia-smi", "-i", str(unit),]).decode('utf-8').split('\n')
    return 'Current GPU usage: ' + result[0] + '\r\n' + result[5] + '\r\n' + result[8]


def parse_RAM_info():
    return 'Current RAM usage: '+str(round(psutil.Process(os.getpid()).memory_info().rss / 1E6, 2))+' MB'


def countParam(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def getOneHotEncoding(imgBatch, labelBatch):
    """
    :param imgBatch: image minibatch (FloatTensor) to extract shape and device info for output
    :param labelBatch: label minibatch (LongTensor) to be converted to one-hot encoding
    :return: One-hot encoded label minibatch with equal size as imgBatch and stored on same device
    """
    if imgBatch.size()[1] != 1: # Multi-label segmentation otherwise binary segmentation
        labelBatch = labelBatch.unsqueeze(1)
        onehotEncoding = torch.zeros_like(imgBatch)
        onehotEncoding.scatter_(1, labelBatch, 1)
        return onehotEncoding
    return labelBatch


def getWeightsForCEloss(dataset, train_idx, areLabelsOnehotEncoded, device, logger):
    # Choice 1) Manually set custom weights
    weights = torch.tensor([1,2,4,6,2,3], dtype=torch.float32, device=device)
    weights = weights / weights.sum()

    # Choice 2) Compute weights as "np.mean(histogram) / histogram"
    dataloader = DataLoader(dataset=dataset, batch_size=6, sampler=SubsetRandomSampler(train_idx), num_workers=6)

    if areLabelsOnehotEncoded:
        histograms = 0
        for batch in dataloader:
            imgBatch, segBatch = batch
            amountLabels = segBatch.size()[1]
            if amountLabels == 1: # binary segmentation
                histograms = histograms + torch.tensor([(segBatch==0).sum(),(segBatch==1).sum()])
            else: # multi-label segmentation
                if imgBatch.dim() == 4: #2D data
                    histograms = histograms + segBatch.sum(3).sum(2).sum(0)
                else: #3D data
                    histograms = histograms + segBatch.sum(4).sum(3).sum(2).sum(0)

        histograms = histograms.numpy()
    else:
        histograms = np.array([0])
        for batch in dataloader:
            _, segBatch = batch

            segHistogram = np.histogram(segBatch.numpy(), segBatch.numpy().max()+1)[0]

            if len(histograms) >= len(segHistogram): #(segHistogram could have different size than histograms)
                histograms[:len(segHistogram)] += segHistogram
            else:
                segHistogram[:len(histograms)] += histograms
                histograms = segHistogram

    weights = np.mean(histograms) / histograms
    weights = torch.from_numpy(weights).float().to(device)

    logger.info('=> Weights for CE-loss: '+str(weights))

    return weights



def getMeanDiceScores(diceScores, logger):
    """
    Compute mean label dice scores of numpy dice score array (2d) (and its mean)
    :return: mean label dice scores with '-1' representing totally missing label (meanLabelDiceScores), mean overall dice score (meanOverallDice)
    """
    meanLabelDiceScores = np.ma.masked_where(diceScores == -1, diceScores).mean(0).data
    label_GT_occurrences = (diceScores != -1).sum(0)
    if (label_GT_occurrences == 0).any():
        logger.info('[# WARNING #] Label(s): ' + str(np.argwhere(label_GT_occurrences == 0).flatten() + 1) + ' not present at all in current dataset split!')
        meanLabelDiceScores[label_GT_occurrences == 0] = -1
    meanOverallDice = meanLabelDiceScores[meanLabelDiceScores != -1].mean()

    return meanLabelDiceScores, meanOverallDice


def getDiceScores(prediction, segBatch):
    """
    Compute mean dice scores of predicted foreground labels.
    NOTE: Dice scores of missing gt labels will be excluded and are thus represented by -1 value entries in returned dice score matrix!
    NOTE: Method changes prediction to 0/1 values in the binary case!
    :param prediction: BxCxHxW (if 2D) or BxCxHxWxD (if 3D) FloatTensor (care: prediction has not undergone any final activation!) (note: C=1 for binary segmentation task)
    :param segBatch: BxCxHxW (if 2D) or BxCxHxWxD (if 3D) FloatTensor (Onehot-Encoding) or Bx1xHxW (if 2D) or Bx1xHxWxD (if 3D) LongTensor
    :return: Numpy array containing BxC-1 (background excluded) dice scores
    """
    batchSize, amountClasses = prediction.size()[0], prediction.size()[1]

    if amountClasses == 1: # binary segmentation task => simulate sigmoid to get label results
        prediction[prediction >= 0] = 1
        prediction[prediction < 0] = 0
        prediction = prediction.squeeze(1)
        segBatch = segBatch.squeeze(1)
        amountClasses += 1
    else: # multi-label segmentation task
        prediction = prediction.argmax(1) # LongTensor without C-channel
        if segBatch.dtype == torch.float32:  # segBatch is onehot-encoded
            segBatch = segBatch.argmax(1)
        else:
            segBatch = segBatch.squeeze(1)

    prediction = prediction.view(batchSize, -1)
    segBatch = segBatch.view(batchSize, -1)

    labelDiceScores = np.zeros((batchSize, amountClasses-1), dtype=np.float32) - 1 #ignore background class!
    for b in range(batchSize):
        currPred = prediction[b,:]
        currGT = segBatch[b,:]

        for c in range(1,amountClasses):
            classPred = (currPred == c).float()
            classGT = (currGT == c).float()

            if classGT.sum() != 0: # only evaluate label prediction when is also present in ground-truth
                labelDiceScores[b, c-1] = ((2. * (classPred * classGT).sum()) / (classGT.sum() + classPred.sum())).item()

    return labelDiceScores


def printResultsForDiseaseModel(evaluatorID, allClassEvaluators, logger):
    logger.info('########## NOW: Detection (average precision) and segmentation accuracies (object-level dice): ##########')
    precisionsTub, avg_precisionTub, avg_dice_scoreTub, std_dice_scoreTub, min_dice_scoreTub, max_dice_scoreTub = allClassEvaluators[evaluatorID][0].score()  # tubuliresults
    precisionsGlom, avg_precisionGlom, avg_dice_scoreGlom, std_dice_scoreGlom, min_dice_scoreGlom, max_dice_scoreGlom = allClassEvaluators[evaluatorID][1].score()  # tubuliresults
    precisionsTuft, avg_precisionTuft, avg_dice_scoreTuft, std_dice_scoreTuft, min_dice_scoreTuft, max_dice_scoreTuft = allClassEvaluators[evaluatorID][2].score()  # tubuliresults
    precisionsVeins, avg_precisionVeins, avg_dice_scoreVeins, std_dice_scoreVeins, min_dice_scoreVeins, max_dice_scoreVeins = allClassEvaluators[evaluatorID][3].score()  # tubuliresults
    precisionsArtery, avg_precisionArtery, avg_dice_scoreArtery, std_dice_scoreArtery, min_dice_scoreArtery, max_dice_scoreArtery = allClassEvaluators[evaluatorID][4].score()  # tubuliresults
    precisionsLumen, avg_precisionLumen, avg_dice_scoreLumen, std_dice_scoreLumen, min_dice_scoreLumen, max_dice_scoreLumen = allClassEvaluators[evaluatorID][5].score()  # tubuliresults
    logger.info('DETECTION RESULTS MEASURED BY AVERAGE PRECISION:')
    logger.info('0.5    0.55    0.6    0.65    0.7    0.75    0.8    0.85    0.9 <- Thresholds')
    logger.info(str(np.round(precisionsTub, 4)) + ', Mean: ' + str(np.round(avg_precisionTub, 4)) + '  <-- Tubuli')
    logger.info(str(np.round(precisionsGlom, 4)) + ', Mean: ' + str(np.round(avg_precisionGlom, 4)) + '  <-- Glomeruli (incl. tuft)')
    logger.info(str(np.round(precisionsTuft, 4)) + ', Mean: ' + str(np.round(avg_precisionTuft, 4)) + '  <-- Tuft')
    logger.info(str(np.round(precisionsVeins, 4)) + ', Mean: ' + str(np.round(avg_precisionVeins, 4)) + '  <-- Veins')
    logger.info(str(np.round(precisionsArtery, 4)) + ', Mean: ' + str(np.round(avg_precisionArtery, 4)) + '  <-- Artery (incl. lumen)')
    logger.info(str(np.round(precisionsLumen, 4)) + ', Mean: ' + str(np.round(avg_precisionLumen, 4)) + '  <-- Artery lumen')
    logger.info('SEGMENTATION RESULTS MEASURED BY OBJECT-LEVEL DICE SCORES:')
    logger.info('Mean: ' + str(np.round(avg_dice_scoreTub, 4)) + ', Std: ' + str(np.round(std_dice_scoreTub, 4)) + ', Min: ' + str(np.round(min_dice_scoreTub, 4)) + ', Max: ' + str(np.round(max_dice_scoreTub, 4)) + '  <-- Tubuli')
    logger.info('Mean: ' + str(np.round(avg_dice_scoreGlom, 4)) + ', Std: ' + str(np.round(std_dice_scoreGlom, 4)) + ', Min: ' + str(np.round(min_dice_scoreGlom, 4)) + ', Max: ' + str(np.round(max_dice_scoreGlom, 4)) + '  <-- Glomeruli (incl. tuft)')
    logger.info('Mean: ' + str(np.round(avg_dice_scoreTuft, 4)) + ', Std: ' + str(np.round(std_dice_scoreTuft, 4)) + ', Min: ' + str(np.round(min_dice_scoreTuft, 4)) + ', Max: ' + str(np.round(max_dice_scoreTuft, 4)) + '  <-- Tuft')
    logger.info('Mean: ' + str(np.round(avg_dice_scoreVeins, 4)) + ', Std: ' + str(np.round(std_dice_scoreVeins, 4)) + ', Min: ' + str(np.round(min_dice_scoreVeins, 4)) + ', Max: ' + str(np.round(max_dice_scoreVeins, 4)) + '  <-- Veins')
    logger.info('Mean: ' + str(np.round(avg_dice_scoreArtery, 4)) + ', Std: ' + str(np.round(std_dice_scoreArtery, 4)) + ', Min: ' + str(np.round(min_dice_scoreArtery, 4)) + ', Max: ' + str(np.round(max_dice_scoreArtery, 4)) + '  <-- Artery (incl. lumen)')
    logger.info('Mean: ' + str(np.round(avg_dice_scoreLumen, 4)) + ', Std: ' + str(np.round(std_dice_scoreLumen, 4)) + ', Min: ' + str(np.round(min_dice_scoreLumen, 4)) + ', Max: ' + str(np.round(max_dice_scoreLumen, 4)) + '  <-- Artery lumen')



import numpy as np
from skimage.util import view_as_windows
from itertools import product
from typing import Tuple

def patchify(patches: np.ndarray, patch_size: Tuple[int, int], step: int = 1):
    return view_as_windows(patches, patch_size, step)

def unpatchify(patches: np.ndarray, imsize: Tuple[int, int]):

    assert len(patches.shape) == 4

    i_h, i_w = imsize
    image = np.zeros(imsize, dtype=patches.dtype)
    divisor = np.zeros(imsize, dtype=patches.dtype)

    n_h, n_w, p_h, p_w = patches.shape

    # Calculat the overlap size in each axis
    o_w = (n_w * p_w - i_w) / (n_w - 1)
    o_h = (n_h * p_h - i_h) / (n_h - 1)

    # The overlap should be integer, otherwise the patches are unable to reconstruct into a image with given shape
    assert int(o_w) == o_w
    assert int(o_h) == o_h

    o_w = int(o_w)
    o_h = int(o_h)

    s_w = p_w - o_w
    s_h = p_h - o_h

    for i, j in product(range(n_h), range(n_w)):
        patch = patches[i,j]
        image[(i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w] += patch
        divisor[(i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w] += 1

    return image / divisor

# Examplary use:
# # # # # # # # #
# import numpy as np
# from patchify import patchify, unpatchify
#
# image = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
#
# patches = patchify(image, (2,2), step=1) # split image into 2*3 small 2*2 patches.
#
# assert patches.shape == (2, 3, 2, 2)
# reconstructed_image = unpatchify(patches, image.shape)
#
# assert (reconstructed_image == image).all()



def getChannelSmootingConvLayer(channels, kernel_size=5, sigma=1.5):

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          (-torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)).float()
                      )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False, padding=2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


if __name__ == '__main__':
    print()
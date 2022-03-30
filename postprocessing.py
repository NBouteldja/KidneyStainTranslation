import numpy as np
import torch
import torch.nn as nn
import cv2
import math
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes
from skimage.morphology import remove_small_objects

from utils import getChannelSmootingConvLayer
from utils import generate_ball


def postprocessPredictionAndGT(prediction, GTpre, device, tubuliInstanceID_StartsWith, predictionsmoothing=False, holefilling=True):
    ################# PREDICTION SMOOTHING ################
    if predictionsmoothing:
        smoothingKernel = getChannelSmootingConvLayer(8).to(device)
        prediction = smoothingKernel(prediction)

    netOutputPrediction = torch.argmax(prediction, dim=1).squeeze(0).to("cpu").numpy() # Label 0/1/2/3/4/5/6/7: Background/tubuli/glom_full/glom_tuft/veins/artery_full/artery_lumen/border
    labelMap = netOutputPrediction.copy()

    ################# HOLE FILLING ################
    if holefilling:
        labelMap[binary_fill_holes(labelMap==1)] = 1 #tubuli
        labelMap[binary_fill_holes(labelMap==4)] = 4 #veins
        tempTuftMask = binary_fill_holes(labelMap==3) #tuft
        labelMap[binary_fill_holes(np.logical_or(labelMap==3, labelMap==2))] = 2 #glom
        labelMap[tempTuftMask] = 3 #tuft
        tempArteryLumenMask = binary_fill_holes(labelMap == 6)  #artery_lumen
        labelMap[binary_fill_holes(np.logical_or(labelMap == 5, labelMap == 6))] = 5  #full_artery
        labelMap[tempArteryLumenMask] = 6  #artery_lumen

    ###### REMOVING TOO SMALL CONNECTED REGIONS ######
    TUBULI_MIN_SIZE = 400
    GLOM_MIN_SIZE = 1500
    TUFT_MIN_SIZE = 500
    VEIN_MIN_SIZE = 3000
    ARTERY_MIN_SIZE = 400
    LUMEN_MIN_SIZE = 20

    temp, _ = label(labelMap == 1)
    finalResults_Instance = np.asarray(remove_small_objects(temp, min_size=TUBULI_MIN_SIZE) > 0, np.uint16)
    temp, _ = label(np.logical_or(labelMap == 2, labelMap == 3))
    finalResults_Instance[remove_small_objects(temp, min_size=GLOM_MIN_SIZE) > 0] = 2
    temp, _ = label(labelMap == 3)
    finalResults_Instance[remove_small_objects(temp, min_size=TUFT_MIN_SIZE) > 0] = 3
    temp, _ = label(labelMap == 4)
    finalResults_Instance[remove_small_objects(temp, min_size=VEIN_MIN_SIZE) > 0] = 4
    temp, _ = label(np.logical_or(labelMap == 5, labelMap == 6))
    finalResults_Instance[remove_small_objects(temp, min_size=ARTERY_MIN_SIZE) > 0] = 5
    temp, _ = label(labelMap == 6)
    finalResults_Instance[remove_small_objects(temp, min_size=LUMEN_MIN_SIZE) > 0] = 6

    ############ PERFORM TUBULE DILATION ############
    temp, numberTubuli = label(finalResults_Instance == 1)
    assert numberTubuli < 65500, print('ERROR: TOO MANY TUBULI DETECTED - MAX ARE 2^16=65k COZ OF UINT16 !')
    temp[temp > 0] += (tubuliInstanceID_StartsWith-1)
    temp = cv2.dilate(np.asarray(temp, np.uint16), kernel=np.asarray(generate_ball(2), np.uint8), iterations=1)
    mask = temp > 0
    finalResults_Instance[mask] = temp[mask]




    GT = np.asarray(GTpre, np.uint16)

    ############ PERFORM TUBULE DILATION, Note: In order to separate touching instances in Ground-truth,
    # tubules have been eroded by 1 pixel before exporting from QuPath (all providing label 1). Thus, for evaluation,
    # we here need to dilate them back. ############
    temp, numberTubuli = label(GT == 1)
    assert numberTubuli < 65500, print('ERROR: TOO MANY TUBULI DETECTED - MAX ARE 2^16=65k COZ OF UINT16 !')
    temp[temp > 0] += (tubuliInstanceID_StartsWith-1)
    temp = cv2.dilate(np.asarray(temp, np.uint16), kernel=np.asarray(generate_ball(1), np.uint8), iterations=1)
    mask = temp > 0
    GT[mask] = temp[mask]


    return finalResults_Instance, netOutputPrediction, GT




def extractInstanceChannels(postprocessedPredInstance, postprocessedGTInstance, tubuliInstanceID_StartsWith):
    labeledGlom, _ = label(np.logical_or(postprocessedPredInstance == 2, postprocessedPredInstance == 3))
    labeledTuft, _ = label(postprocessedPredInstance == 3)
    labeledVeins, _ = label(postprocessedPredInstance == 4)
    labeledArtery, _ = label(np.logical_or(postprocessedPredInstance == 5, postprocessedPredInstance == 6))
    labeledArteryLumen, _ = label(postprocessedPredInstance == 6)

    labeledGlomGT, _ = label(np.logical_or(postprocessedGTInstance == 2, postprocessedGTInstance == 3))
    labeledTuftGT, _ = label(postprocessedGTInstance == 3)
    labeledVeinsGT, _ = label(postprocessedGTInstance == 4)
    labeledArteryGT, _ = label(np.logical_or(postprocessedGTInstance == 5, postprocessedGTInstance == 6))
    labeledArteryLumenGT, _ = label(postprocessedGTInstance == 6)

    tubuliInstanceChannel = postprocessedPredInstance.copy()
    tubuliInstanceChannel[tubuliInstanceChannel < tubuliInstanceID_StartsWith] = 0
    tubuliInstanceChannel[tubuliInstanceChannel>0] -= (tubuliInstanceID_StartsWith-1)

    tubuliInstanceChannelGT = postprocessedGTInstance.copy()
    tubuliInstanceChannelGT[tubuliInstanceChannelGT < tubuliInstanceID_StartsWith] = 0
    tubuliInstanceChannelGT[tubuliInstanceChannelGT>0] -= (tubuliInstanceID_StartsWith-1)

    return [tubuliInstanceChannel, labeledGlom, labeledTuft, labeledVeins, labeledArtery, labeledArteryLumen], [tubuliInstanceChannelGT, labeledGlomGT, labeledTuftGT, labeledVeinsGT, labeledArteryGT, labeledArteryLumenGT]




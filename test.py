import os
import math
import time
import sys
import logging as log
from PIL import Image
import cv2
import numpy as np
import torch
from options import Options
from models import create_model
from postprocessing import postprocessPredictionAndGT, extractInstanceChannels
from evaluation import ClassEvaluator
from utils import saveIHCTranslation_PredictionOverlay, printResultsForDiseaseModel
from models.utils import split_6_ch_to_3

from models.model import Custom



chosenModelIterationForAllStains = 300000
stainsToValidate = ['aSMA', 'NGAL', 'CD31', 'ColIII']

saveImageResults = True
saveImageResultsSeparately = False
onlyApplyTestTimeAugmentation = False
saveInstanceDiceScoresForTTA = True

GPUno = 0
device = torch.device("cuda:" + str(GPUno) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(GPUno)

opt = Options().parseTestStain(stain = stainsToValidate[0])

# Set up logger
log.basicConfig(
    level=log.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%Y/%m/%d %I:%M:%S %p',
    handlers=[
        log.FileHandler(opt.testResultsPath + '/testPerformance_model' + str(chosenModelIterationForAllStains) + '.log', 'w'),
        log.StreamHandler(sys.stdout)
    ])
logger = log.getLogger()
logger.info('Script call arguments are:\n\n' + ' '.join(sys.argv[1:]) + '\n\n')

segModel = Custom(input_ch=3, output_ch=8, modelDim=2)
segModel.load_state_dict(torch.load('<path-to-segmentation-model.pt>', map_location=lambda storage, loc: storage))
segModel.eval()
segModel = segModel.to(device)


numberClassesToEvaluate = 6
classEvaluators = []
classEvaluatorsTTA = []
for i in range(len(stainsToValidate)):
    classEvaluators.append([ClassEvaluator() for _ in range(numberClassesToEvaluate)])
    classEvaluatorsTTA.append([ClassEvaluator() for _ in range(numberClassesToEvaluate)])

tubuliInstanceID_StartsWith = 10


for s, stain in enumerate(stainsToValidate):

    opt = Options().parseTestStain(stain = stain)  # get options => Use same training options args
    opt.logger = logger
    opt.gpu_ids = [GPUno]
    opt.phase = 'test'
    opt.load_iter = chosenModelIterationForAllStains

    model = create_model(opt)      # create cyclegan given opt.model and init and push to GPU
    model.setup(opt)               # load networks
    model.setSavedNetsEval()

    image_dir = os.path.join(opt.dataroot, stain, 'test')
    files = sorted(list(filter(lambda x: ').png' in x, os.listdir(image_dir))))

    logger.info('Loading '+stain+' dataset with size: ' + str(len(files)))
    for k, fname in enumerate(files):
        imgOrig = np.array(Image.open(os.path.join(image_dir, fname)))[:,:,:3]
        lbl = np.array(Image.open(os.path.join(image_dir, fname.replace('.png', '-labels.png'))))

        assert imgOrig.shape == (640, 640, 3) and lbl.shape == (516, 516), 'Image size {} or {} unsupported.'.format(imgOrig.shape, lbl.shape)

        img = (np.asarray(imgOrig, np.float32) / 255.0 - 0.5) / 0.5

        lbl = np.asarray(lbl, np.long)
        img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)

        with torch.no_grad():
            STAIN_to_PAS_img = model.perform_test_conversion(img)

            if opt.use_MC:
                STAIN_to_PAS_img = split_6_ch_to_3([STAIN_to_PAS_img])

            prediction = segModel(STAIN_to_PAS_img)

            if not onlyApplyTestTimeAugmentation:
                postprocessedPredInstance, outputPrediction, postprocessedGTInstance = postprocessPredictionAndGT(prediction, lbl, device, tubuliInstanceID_StartsWith, predictionsmoothing=False, holefilling=True)
                classInstancePredictionList, classInstanceGTList = extractInstanceChannels(postprocessedPredInstance, postprocessedGTInstance, tubuliInstanceID_StartsWith)
                for i in range(numberClassesToEvaluate):
                    classEvaluators[s][i].add_example(classInstancePredictionList[i], classInstanceGTList[i])

            STAIN_to_PAS_img = STAIN_to_PAS_img.flip(2)
            prediction += segModel(STAIN_to_PAS_img).flip(2)
            STAIN_to_PAS_img = STAIN_to_PAS_img.flip(3)
            prediction += segModel(STAIN_to_PAS_img).flip(3).flip(2)
            STAIN_to_PAS_img = STAIN_to_PAS_img.flip(2)
            prediction += segModel(STAIN_to_PAS_img).flip(3)
            STAIN_to_PAS_img = STAIN_to_PAS_img.flip(3)
            prediction /= 4.

            postprocessedPredInstanceTTA, outputPredictionTTA, postprocessedGTInstanceTTA = postprocessPredictionAndGT(prediction, lbl, device, tubuliInstanceID_StartsWith, predictionsmoothing=False, holefilling=True)
            classInstancePredictionListTTA, classInstanceGTListTTA = extractInstanceChannels(postprocessedPredInstanceTTA, postprocessedGTInstanceTTA, tubuliInstanceID_StartsWith)
            for i in range(numberClassesToEvaluate):
                classEvaluatorsTTA[s][i].add_example(classInstancePredictionListTTA[i], classInstanceGTListTTA[i])

            if saveImageResults:
                fname_Result = fname[:-4] if fname.endswith('.svs') else fname[:-5]
                figPath = os.path.join(opt.testResultImagesPath, fname_Result + '.png')
                saveIHCTranslation_PredictionOverlay(stain_img=imgOrig,
                                                     translated_img=np.asarray(np.round((STAIN_to_PAS_img.squeeze(0).to("cpu").numpy().transpose(1, 2, 0)*0.5+0.5)*255.), np.uint8),
                                                     network_output=outputPredictionTTA,
                                                     instance_pred=postprocessedPredInstanceTTA,
                                                     instance_gt=postprocessedGTInstanceTTA,
                                                     savePath=figPath,
                                                     tubuliInstanceID_StartsWith=tubuliInstanceID_StartsWith,
                                                     saveImageResultsSeparately = saveImageResultsSeparately,
                                                     figHeight=12,
                                                     alpha=0.4)



for s, stain in enumerate(stainsToValidate):
    logger.info('############################### RESULTS FOR -> '+ stain +' <- ###############################')
    if not onlyApplyTestTimeAugmentation:
        printResultsForDiseaseModel(evaluatorID=s, allClassEvaluators=classEvaluators, logger=logger)
    logger.info('=> TTA Results <=')
    printResultsForDiseaseModel(evaluatorID=s, allClassEvaluators=classEvaluatorsTTA, logger=logger)

    if saveInstanceDiceScoresForTTA:
        resultsDic = '/'.join(opt.testResultImagesPath.split('/')[:-1])+'/'+stain
        np.save(os.path.join(resultsDic, 'diceScores_Tubule_TTA.npy'), np.array(classEvaluatorsTTA[s][0].diceScores))
        np.save(os.path.join(resultsDic, 'diceScores_Glom_TTA.npy'), np.array(classEvaluatorsTTA[s][1].diceScores))
        np.save(os.path.join(resultsDic, 'diceScores_Tuft_TTA.npy'), np.array(classEvaluatorsTTA[s][2].diceScores))
        np.save(os.path.join(resultsDic, 'diceScores_Vein_TTA.npy'), np.array(classEvaluatorsTTA[s][3].diceScores))
        np.save(os.path.join(resultsDic, 'diceScores_Artery_TTA.npy'), np.array(classEvaluatorsTTA[s][4].diceScores))
        np.save(os.path.join(resultsDic, 'diceScores_Lumen_TTA.npy'), np.array(classEvaluatorsTTA[s][5].diceScores))

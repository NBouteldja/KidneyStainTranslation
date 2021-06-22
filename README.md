# KidneyStainTranslation
This repo provides a framework to train CycleGAN- and U-GAT-IT-based translators for unsupervised stain-to-stain translation in histology.
<br>
| Input aSMA image | Fake PAS translation |
|:--:|:--:|
| <img src="https://github.com/NBouteldja/KidneyStainTranslation/blob/main/exemplaryResults/aSMA_image.png?raw=true" width="400">| <img src="https://github.com/NBouteldja/KidneyStainTranslation/blob/main/exemplaryResults/fakePAS_translation.png?raw=true" width="400"> |
| Prediction | Ground-truth |
| <img src="https://github.com/NBouteldja/KidneyStainTranslation/blob/main/exemplaryResults/prediction.png?raw=true" width="324">| <img src="https://github.com/NBouteldja/KidneyStainTranslation/blob/main/exemplaryResults/groundtruth.png?raw=true" width="324"> |

# Training
To train a CycleGAN translator (e.g. incorporating a prior segmentation model), use the following command:
```
python ./KidneyStainTranslation/train.py --stain aSMA --stainB PAS --dataroot <path-to-data> --resultsPath <path-to-store-results> --netD n_layers --netG unet_7 --ngf 32 --ndf 32 --batch_size 3 --niters_init 0 --lr 0.0001 --preprocess none --niters 300000 --load_size 640 --crop_size 640 --lambda_A 1 --lambda_B 1 --lambda_id 1 --niters_linDecay 100 --saveModelEachNIteration 10000 --validation_freq 1000 --n_layers_D 4 --gpu_ids 0 --update_TB_images_freq 5000 --use_segm_model --lambda_Seg 1
```

# Testing
Use the same arguments to test the trained translator:
```
python ./KidneyStainTranslation/test.py --stain aSMA --stainB PAS --dataroot <path-to-data> --resultsPath <path-to-store-results> --netD n_layers --netG unet_7 --ngf 32 --ndf 32 --batch_size 3 --niters_init 0 --lr 0.0001 --preprocess none --niters 300000 --load_size 640 --crop_size 640 --lambda_A 1 --lambda_B 1 --lambda_id 1 --niters_linDecay 100 --saveModelEachNIteration 10000 --validation_freq 1000 --n_layers_D 4 --gpu_ids 0 --update_TB_images_freq 5000 --use_segm_model --lambda_Seg 1
```

# Contact
Nassim Bouteldja<br>
Institute of Pathology<br>
RWTH Aachen University Hospital<br>
Pauwelsstrasse 30<br>
52074 Aachen, Germany<br>
E-mail: 	nbouteldja@ukaachen.de<br>
<br>

#
    /**************************************************************************
    *                                                                         *
    *   Copyright (C) 2021 by RWTH Aachen University                          *
    *   http://www.rwth-aachen.de                                             *
    *                                                                         *
    *   License:                                                              *
    *                                                                         *
    *   This software is dual-licensed under:                                 *
    *   • Commercial license (please contact: lfb@lfb.rwth-aachen.de)         *
    *   • AGPL (GNU Affero General Public License) open source license        *
    *                                                                         *
    ***************************************************************************/     

# Stageformer
## Note:

Due to the double-blind policy for other papers associated with this account, we may at any time set this project to "private" and make it invisible, in which case, if you need the code, send an email to leewu@mail.sdu.edu.cn and tensorfire@mail.sdu.edu.cn
## Requirements
 - Scikit-learn 1.0
 - Pytorch 1.10.0
 - TensorFlow 2.7.0
 - Numpy 1.21.2 
 - Panda 1.2.3	

## Process MIMIC Dataset
### Pre-process MIMIC
Read https://github.com/AnonymousRes/MIMIC-IV-PROCESSING/blob/main/README.md
### Generate Experimental Data
1. Run Stageformer/DataGenerating/dp_mimic3_processing.py

       python dp_mimic3_processing.py       

3. Run Stageformer/DataGenerating/dp_mimic4_processing.py

       python dp_mimic4_processing.py

### Train Stageformer
1. Run Stageformer/OurModelDecompensationPrediction.py (MIMIC-III)

       python OurModelDecompensationPrediction.py 3
       
2. Run Stageformer/OurModelDecompensationPrediction.py (MIMIC-IV)

       python OurModelDecompensationPrediction.py 4
       
3. Run Ablation Experiment (MIMIC-III)

       python Ablation.py 3
       
3. Run Ablation Experiment (MIMIC-IV)

       python Ablation.py 4

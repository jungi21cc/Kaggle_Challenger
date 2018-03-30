
# The core idea behind all blends is "diversity". 
# By blending some moderate model results (with some weights), we can create a more "diverse" stable results.
# Errors from one model will be covered by others. Same goes for all the models. 
# So, we get more stable results after blendings. 


import pandas as pd
import numpy as np




# tidy = pd.read_csv('../input/tidy-xgboost-glmnet-text2vec-lsa/tidy_xgb_glm.csv') #0.9788
# Conv1D = pd.read_csv('../input/Conv1D/dpcnn_test_preds.csv') #0.9810
# wordbtch = pd.read_csv('../input/wordbatch-fm-ftrl-using-mse-lb-0-9804/lvl0_wordbatch_clean_sub.csv') #0.9813
# textcnn = pd.read_csv('../input/textcnn/submission(2).csv') #0.9820
# rkera = pd.read_csv('../input/why-a-such-low-score-with-r-and-keras/submission.csv') #0.9824
# Pooled_GRU_FastText = pd.read_csv('../input/allofdata/submission(5).csv') #0.9830
best = pd.read_csv('../input/toxic-hight-of-blending/hight_of_blending.csv') #0.9835

bilst = pd.read_csv('../input/bidirectional-gru-with-convolution/submission.csv') #0.9841
Two_RNN_CNN = pd.read_csv('../input/two-rnn-cnn/submission (3).csv') #0.9844
lgbm = pd.read_csv('../input/lgbm-with-words-and-chars-n-gram/lvl0_lgbm_clean_sub.csv') #0.9848
corrbl = pd.read_csv('../input/another-blend-tinkered-by-correlation/corr_blend.csv') #0.9855
sub1 = pd.read_csv('../input/two-more-source/rank_averaged_submission.csv') #0.9857
OOF_stacking_regime = pd.read_csv('../input/allofdata/submission (6).csv') #0.9858
toxic_one_more_blend = pd.read_csv('../input/one-more-blend/one_more_blend.csv') #0.9865
Toxic_Avenger_Spin = pd.read_csv('../input/allofdata/submission (4).csv') #0.9866
blend_it_all = pd.read_csv('../input/allofdata/blend_it_all (1).csv') #0.9867
blend_it_all2 = pd.read_csv('../input/blenditall2/blend_it_all (2).csv') #0.9868






b1 = best.copy()
col = best.columns

col = col.tolist()
col.remove('id')
for i in col:
    b1[i] = (5*sub1[i] + 2*Two_RNN_CNN[i] + 8*Toxic_Avenger_Spin[i] + 6*OOF_stacking_regime[i] + 9*blend_it_all[i] + 7*toxic_one_more_blend[i] + 1*bilst[i]  + 3*lgbm[i] + 4*corrbl[i] + 10*blend_it_all2[i]) /  55
    
b1.to_csv('blend_it_all.csv', index = False)

# 300 model
import numpy as np
import scipy.io
import pandas as pd


# Loop through subs

def DecisionStim(behav_path, mri_path, df, sub_list):

    fName = 'DecisionStim_wMot'
    subNames = []
    
    import numpy as np

    for subNum, subName in df.items():

        if np.isin(subNum,sub_list):
            
            subNames.append(subName)

            #Load mat files as dicts
            trialsPRun = 66
            currData = scipy.io.loadmat(behav_path+'/'+str(subNum)+'/learning_'+str(subNum)+'_session_1.mat')
            predTypes = 3
            predNum = trialsPRun*predTypes
            trialNum = 2*trialsPRun

            #Get reward vec
            rewardOnsets1 = currData['rewardOnset'][0][1:67]-currData['alignTimesScanner'][0][0]
            rewardOnsets2 = currData['rewardOnset'][0][67:134]-currData['alignTimesScanner'][0][1]
            rewards1 = currData['reward'][0:66,0]
            rewards2 = currData['reward'][66:133,0]
            switchRew1 = np.all(currData['pairsValue'] == currData['R'],axis=1)[0:66]
            switchRew2 = np.all(currData['pairsValue'] == currData['R'],axis=1)[66:133]
            rewardOnsets1_pos = rewardOnsets1[rewards1 == 10]
            rewardOnsets1_neg = rewardOnsets1[rewards1 == -10]
            rewardOnsets1_neut = rewardOnsets1[rewards1 == 0]
            rewardOnsets1_swit = rewardOnsets1[switchRew1==False]
            rewardOnsets2_pos = rewardOnsets2[rewards2 == 10]
            rewardOnsets2_neg = rewardOnsets2[rewards2 == -10]
            rewardOnsets2_neut = rewardOnsets2[rewards2 == 0]
            rewardOnsets2_swit = rewardOnsets2[switchRew2==False]

            #Get choice vec for motion
            rectOnsets1 = currData['rectOnset'][0][0:66]-currData['alignTimesScanner'][0][0]
            rectOnsets2 = currData['rectOnset'][0][66:133]-currData['alignTimesScanner'][0][1]

            #Get face value that was chosen
            choices = currData['subjectChoice']
            options = currData['pairsValue']
            trials = np.arange(trialNum).reshape(trialNum,1)
            faceVals = options[trials,choices-1]

            #Get face vec
            faceOnsets1 = currData['stimOnset'][0][0:66]-currData['alignTimesScanner'][0][0]
            faceOnsets2 = currData['stimOnset'][0][66:133]-currData['alignTimesScanner'][0][1]
            faceVals1 = faceVals[0:66,0]
            faceVals2 = faceVals[66:133,0]
            faceOnsets1_pos = faceOnsets1[faceVals1 == 1]
            faceOnsets1_neg = faceOnsets1[faceVals1 == -1]
            faceOnsets1_neut = faceOnsets1[faceVals1 == 0]
            faceOnsets2_pos = faceOnsets2[faceVals2 == 1]
            faceOnsets2_neg = faceOnsets2[faceVals2 == -1]
            faceOnsets2_neut = faceOnsets2[faceVals2 == 0]

            #Create vectors for pandas table - %% 
            onsets = np.zeros(predNum)
            onsets[np.arange(0,predNum,predTypes)] = faceOnsets1
            onsets[np.arange(1,predNum+1,predTypes)] = rectOnsets1
            onsets[np.arange(2,predNum+1,predTypes)] = rewardOnsets1
            durations = np.zeros(predNum)
            durations[np.arange(0,predNum,predTypes)] = 3
            durations[np.arange(1,predNum+1,predTypes)] = 0
            durations[np.arange(2,predNum+1,predTypes)] = 2
            trial_type = np.empty((predNum),dtype='object')
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_pos))] = 'posFace'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neg))] = 'negFace'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neut))] = 'neuFace'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_pos))] = 'posRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neg))] = 'negRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neut))] = 'neuRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_swit))] = 'surpRew'
            trial_type[np.nonzero(np.isin(onsets,rectOnsets1))] = 'rectOns'
            weights = np.ones(predNum,dtype=int)
            pd1 = pd.DataFrame({'onset':onsets,'duration':durations,'weight':weights,'trial_type':trial_type})

            onsets = np.zeros(predNum)
            onsets[np.arange(0,predNum,predTypes)] = faceOnsets2
            onsets[np.arange(1,predNum+1,predTypes)] = rectOnsets2
            onsets[np.arange(2,predNum+1,predTypes)] = rewardOnsets2
            durations = np.zeros(predNum)
            durations[np.arange(0,predNum,predTypes)] = 3
            durations[np.arange(1,predNum+1,predTypes)] = 0
            durations[np.arange(2,predNum+1,predTypes)] = 2
            trial_type = np.empty((predNum),dtype='object')
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_pos))] = 'posFace'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neg))] = 'negFace'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neut))] = 'neuFace'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_pos))] = 'posRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neg))] = 'negRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neut))] = 'neuRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_swit))] = 'surpRew'
            trial_type[np.nonzero(np.isin(onsets,rectOnsets2))] = 'rectOns'
            weights = np.ones(predNum,dtype=int)
            pd2 = pd.DataFrame({'onset':onsets,'duration':durations,'weight':weights,'trial_type':trial_type})

            #Deduct two dummy scans
            pd1['onset'] = pd1['onset']
            pd2['onset'] = pd2['onset']

            pd1.to_csv(mri_path+'/MRI preproc fMRIPrep v2/'+subName+'/Events/task_events1_'+fName+'.tsv')
            pd2.to_csv(mri_path+'/MRI preproc fMRIPrep v2/'+subName+'/Events/task_events2_'+fName+'.tsv')

    return pd1, pd2, fName, subNames

# 200 model

# Loop through subs

def Decision200(behav_path, mri_path, df, sub_list):

    decConstant = 0.2 #200 ms deducted
    
    fName = 'Decision200'
    subNames = []

    for subNum, subName in df.items():

        if np.isin(subNum,sub_list):

            subNames.append(subName)

            #Load mat files as dicts
            trialNum = 132
            currData = scipy.io.loadmat(behav_path+'/'+str(subNum)+'/learning_'+str(subNum)+'_session_1.mat')

            #Get reward vec
            rewardOnsets1 = currData['rewardOnset'][0][1:67]-currData['alignTimesScanner'][0][0]
            rewardOnsets2 = currData['rewardOnset'][0][67:134]-currData['alignTimesScanner'][0][1]
            rewards1 = currData['reward'][0:66,0]
            rewards2 = currData['reward'][66:133,0]
            rewardOnsets1_pos = rewardOnsets1[rewards1 == 10]
            rewardOnsets1_neg = rewardOnsets1[rewards1 == -10]
            rewardOnsets1_neut = rewardOnsets1[rewards1 == 0]
            rewardOnsets2_pos = rewardOnsets2[rewards2 == 10]
            rewardOnsets2_neg = rewardOnsets2[rewards2 == -10]
            rewardOnsets2_neut = rewardOnsets2[rewards2 == 0]

            #Get face value that was chosen
            choices = currData['subjectChoice']
            options = currData['pairsValue']
            trials = np.arange(trialNum).reshape(trialNum,1)
            faceVals = options[trials,choices-1]

            #Get face vec
            faceOnsets1 = currData['rectOnset'][0][0:66]-currData['alignTimesScanner'][0][0]-decConstant
            faceOnsets2 = currData['rectOnset'][0][66:133]-currData['alignTimesScanner'][0][1]-decConstant
            faceVals1 = faceVals[0:66,0]
            faceVals2 = faceVals[66:133,0]
            faceOnsets1_pos = faceOnsets1[faceVals1 == 1]
            faceOnsets1_neg = faceOnsets1[faceVals1 == -1]
            faceOnsets1_neut = faceOnsets1[faceVals1 == 0]
            faceOnsets2_pos = faceOnsets2[faceVals2 == 1]
            faceOnsets2_neg = faceOnsets2[faceVals2 == -1]
            faceOnsets2_neut = faceOnsets2[faceVals2 == 0]

            #Create vectors for pandas table - %% 
            onsets = np.zeros(2*66)
            onsets[np.arange(0,trialNum,2)] = faceOnsets1
            onsets[np.arange(1,trialNum+1,2)] = rewardOnsets1
            durations = np.zeros(2*66)
            durations[np.arange(0,trialNum,2)] = 0
            durations[np.arange(1,trialNum+1,2)] = 2
            trial_type = np.empty((trialNum),dtype='object')
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_pos))] = 'posFace'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neg))] = 'negFace'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neut))] = 'neuFace'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_pos))] = 'posRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neg))] = 'negRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neut))] = 'neuRewd'
            weights = np.ones(trialNum,dtype=int)
            pd1 = pd.DataFrame({'onset':onsets,'duration':durations,'weight':weights,'trial_type':trial_type})

            onsets = np.zeros(2*66)
            onsets[np.arange(0,trialNum,2)] = faceOnsets2
            onsets[np.arange(1,trialNum+1,2)] = rewardOnsets2
            durations = np.zeros(2*66)
            durations[np.arange(0,trialNum,2)] = 0
            durations[np.arange(1,trialNum+1,2)] = 2
            trial_type = np.empty((trialNum),dtype='object')
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_pos))] = 'posFace'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neg))] = 'negFace'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neut))] = 'neuFace'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_pos))] = 'posRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neg))] = 'negRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neut))] = 'neuRewd'
            weights = np.ones(trialNum,dtype=int)
            pd2 = pd.DataFrame({'onset':onsets,'duration':durations,'weight':weights,'trial_type':trial_type})

            #Deduct two dummy scans
            pd1['onset'] = pd1['onset']
            pd2['onset'] = pd2['onset']

            pd1.to_csv(mri_path+'/MRI preproc fMRIPrep/'+subName+'/Events/task_events1_'+fName+'.tsv')
            pd2.to_csv(mri_path+'/MRI preproc fMRIPrep/'+subName+'/Events/task_events2_'+fName+'.tsv')
            
    return pd1, pd2, fName, subNames


# 0 model

# Loop through subs

def Decision0(behav_path, mri_path, df, sub_list):

    decConstant = 0 #0 ms deducted
    
    fName = 'Decision0'
    subNames = []

    for subNum, subName in df.items():

        if np.isin(subNum,sub_list):

            subNames.append(subName)

            #Load mat files as dicts
            trialNum = 132
            currData = scipy.io.loadmat(behav_path+'/'+str(subNum)+'/learning_'+str(subNum)+'_session_1.mat')

            #Get reward vec
            rewardOnsets1 = currData['rewardOnset'][0][1:67]-currData['alignTimesScanner'][0][0]
            rewardOnsets2 = currData['rewardOnset'][0][67:134]-currData['alignTimesScanner'][0][1]
            rewards1 = currData['reward'][0:66,0]
            rewards2 = currData['reward'][66:133,0]
            rewardOnsets1_pos = rewardOnsets1[rewards1 == 10]
            rewardOnsets1_neg = rewardOnsets1[rewards1 == -10]
            rewardOnsets1_neut = rewardOnsets1[rewards1 == 0]
            rewardOnsets2_pos = rewardOnsets2[rewards2 == 10]
            rewardOnsets2_neg = rewardOnsets2[rewards2 == -10]
            rewardOnsets2_neut = rewardOnsets2[rewards2 == 0]

            #Get face value that was chosen
            choices = currData['subjectChoice']
            options = currData['pairsValue']
            trials = np.arange(trialNum).reshape(trialNum,1)
            faceVals = options[trials,choices-1]

            #Get face vec
            faceOnsets1 = currData['rectOnset'][0][0:66]-currData['alignTimesScanner'][0][0]-decConstant
            faceOnsets2 = currData['rectOnset'][0][66:133]-currData['alignTimesScanner'][0][1]-decConstant
            faceVals1 = faceVals[0:66,0]
            faceVals2 = faceVals[66:133,0]
            faceOnsets1_pos = faceOnsets1[faceVals1 == 1]
            faceOnsets1_neg = faceOnsets1[faceVals1 == -1]
            faceOnsets1_neut = faceOnsets1[faceVals1 == 0]
            faceOnsets2_pos = faceOnsets2[faceVals2 == 1]
            faceOnsets2_neg = faceOnsets2[faceVals2 == -1]
            faceOnsets2_neut = faceOnsets2[faceVals2 == 0]

            #Create vectors for pandas table - %% 
            onsets = np.zeros(2*66)
            onsets[np.arange(0,trialNum,2)] = faceOnsets1
            onsets[np.arange(1,trialNum+1,2)] = rewardOnsets1
            durations = np.zeros(2*66)
            durations[np.arange(0,trialNum,2)] = 0
            durations[np.arange(1,trialNum+1,2)] = 2
            trial_type = np.empty((trialNum),dtype='object')
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_pos))] = 'posFace'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neg))] = 'negFace'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neut))] = 'neuFace'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_pos))] = 'posRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neg))] = 'negRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neut))] = 'neuRewd'
            weights = np.ones(trialNum,dtype=int)
            pd1 = pd.DataFrame({'onset':onsets,'duration':durations,'weight':weights,'trial_type':trial_type})

            onsets = np.zeros(2*66)
            onsets[np.arange(0,trialNum,2)] = faceOnsets2
            onsets[np.arange(1,trialNum+1,2)] = rewardOnsets2
            durations = np.zeros(2*66)
            durations[np.arange(0,trialNum,2)] = 0
            durations[np.arange(1,trialNum+1,2)] = 2
            trial_type = np.empty((trialNum),dtype='object')
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_pos))] = 'posFace'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neg))] = 'negFace'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neut))] = 'neuFace'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_pos))] = 'posRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neg))] = 'negRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neut))] = 'neuRewd'
            weights = np.ones(trialNum,dtype=int)
            pd2 = pd.DataFrame({'onset':onsets,'duration':durations,'weight':weights,'trial_type':trial_type})

            #Deduct two dummy scans
            pd1['onset'] = pd1['onset']
            pd2['onset'] = pd2['onset']

            pd1.to_csv(mri_path+'/MRI preproc fMRIPrep/'+subName+'/Events/task_events1_'+fName+'.tsv')
            pd2.to_csv(mri_path+'/MRI preproc fMRIPrep/'+subName+'/Events/task_events2_'+fName+'.tsv')
            
    return pd1, pd2, fName, subNames
            

# to onset model

# Loop through subs

def DecisionDur(behav_path, mri_path, df, sub_list):
    
    fName = 'DecisionDur'
    subNames = []

    for subNum, subName in df.items():

        if np.isin(subNum,sub_list):

            subNames.append(subName)

            #Load mat files as dicts
            trialNum = 132
            currData = scipy.io.loadmat(behav_path+'/'+str(subNum)+'/learning_'+str(subNum)+'_session_1.mat')

            #Get reward vec
            rewardOnsets1 = currData['rewardOnset'][0][1:67]-currData['alignTimesScanner'][0][0]
            rewardOnsets2 = currData['rewardOnset'][0][67:134]-currData['alignTimesScanner'][0][1]
            rewards1 = currData['reward'][0:66,0]
            rewards2 = currData['reward'][66:133,0]
            rewardOnsets1_pos = rewardOnsets1[rewards1 == 10]
            rewardOnsets1_neg = rewardOnsets1[rewards1 == -10]
            rewardOnsets1_neut = rewardOnsets1[rewards1 == 0]
            rewardOnsets2_pos = rewardOnsets2[rewards2 == 10]
            rewardOnsets2_neg = rewardOnsets2[rewards2 == -10]
            rewardOnsets2_neut = rewardOnsets2[rewards2 == 0]

            #Get face value that was chosen
            choices = currData['subjectChoice']
            options = currData['pairsValue']
            trials = np.arange(trialNum).reshape(trialNum,1)
            faceVals = options[trials,choices-1]

            #Get face vec and decision vec
            faceOnsets1 = currData['stimOnset'][0][0:66]-currData['alignTimesScanner'][0][0]
            faceOnsets2 = currData['stimOnset'][0][66:133]-currData['alignTimesScanner'][0][1]
            rectOnsets1 = currData['rectOnset'][0][0:66]-currData['alignTimesScanner'][0][0]
            rectOnsets2 = currData['rectOnset'][0][66:133]-currData['alignTimesScanner'][0][1]
            faceVals1 = faceVals[0:66,0]
            faceVals2 = faceVals[66:133,0]
            faceOnsets1_pos = faceOnsets1[faceVals1 == 1]
            faceOnsets1_neg = faceOnsets1[faceVals1 == -1]
            faceOnsets1_neut = faceOnsets1[faceVals1 == 0]
            faceOnsets2_pos = faceOnsets2[faceVals2 == 1]
            faceOnsets2_neg = faceOnsets2[faceVals2 == -1]
            faceOnsets2_neut = faceOnsets2[faceVals2 == 0]

            #Create vectors for pandas table - %% 
            onsets = np.zeros(2*66)
            onsets[np.arange(0,trialNum,2)] = faceOnsets1
            onsets[np.arange(1,trialNum+1,2)] = rewardOnsets1
            durations = np.zeros(2*66)
            durations[np.arange(0,trialNum,2)] = currData['RT'][0:66]
            durations[np.arange(1,trialNum+1,2)] = 2
            trial_type = np.empty((trialNum),dtype='object')
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_pos))] = 'posFace'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neg))] = 'negFace'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neut))] = 'neuFace'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_pos))] = 'posRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neg))] = 'negRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neut))] = 'neuRewd'
            weights = np.ones(trialNum,dtype=int)
            pd1 = pd.DataFrame({'onset':onsets,'duration':durations,'weight':weights,'trial_type':trial_type})

            onsets = np.zeros(2*66)
            onsets[np.arange(0,trialNum,2)] = faceOnsets2
            onsets[np.arange(1,trialNum+1,2)] = rewardOnsets2
            durations = np.zeros(2*66)
            durations[np.arange(0,trialNum,2)] = currData['RT'][66:133]
            durations[np.arange(1,trialNum+1,2)] = 2
            trial_type = np.empty((trialNum),dtype='object')
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_pos))] = 'posFace'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neg))] = 'negFace'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neut))] = 'neuFace'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_pos))] = 'posRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neg))] = 'negRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neut))] = 'neuRewd'
            weights = np.ones(trialNum,dtype=int)
            pd2 = pd.DataFrame({'onset':onsets,'duration':durations,'weight':weights,'trial_type':trial_type})

            #Deduct two dummy scans
            pd1['onset'] = pd1['onset']
            pd2['onset'] = pd2['onset']

            pd1.to_csv(mri_path+'/MRI preproc fMRIPrep/'+subName+'/Events/task_events1_'+fName+'.tsv')
            pd2.to_csv(mri_path+'/MRI preproc fMRIPrep/'+subName+'/Events/task_events2_'+fName+'.tsv')
            
    return pd1, pd2, fName, subNames


# to onset model

# Loop through subs

def FacePairs(behav_path, mri_path, df, sub_list): #ERROR IN SCRIPT
    
    fName = 'FacePairs'
    subNames = []

    for subNum, subName in df.items():

        if np.isin(subNum,sub_list):

            subNames.append(subName)

            #Load mat files as dicts
            trialNum = 132
            currData = scipy.io.loadmat(behav_path+'/'+str(subNum)+'/learning_'+str(subNum)+'_session_1.mat')

            #Get reward vec
            rewardOnsets1 = currData['rewardOnset'][0][1:67]-currData['alignTimesScanner'][0][0]
            rewardOnsets2 = currData['rewardOnset'][0][67:134]-currData['alignTimesScanner'][0][1]
            rewards1 = currData['reward'][0:66,0]
            rewards2 = currData['reward'][66:133,0]
            rewardOnsets1_pos = rewardOnsets1[rewards1 == 10]
            rewardOnsets1_neg = rewardOnsets1[rewards1 == -10]
            rewardOnsets1_neut = rewardOnsets1[rewards1 == 0]
            rewardOnsets2_pos = rewardOnsets2[rewards2 == 10]
            rewardOnsets2_neg = rewardOnsets2[rewards2 == -10]
            rewardOnsets2_neut = rewardOnsets2[rewards2 == 0]

            #Get face value that was chosen
            choices = currData['subjectChoice']
            options = currData['pairsValue']
            trials = np.arange(trialNum).reshape(trialNum,1)
            faceVals = options[trials,choices-1]

            #Get face vec and decision vec
            faceOnsets1 = currData['stimOnset'][0][0:66]-currData['alignTimesScanner'][0][0]
            faceOnsets2 = currData['stimOnset'][0][66:133]-currData['alignTimesScanner'][0][1]
            faceVals1 = faceVals[0:66,0]
            faceVals2 = faceVals[66:133,0]
            faceOnsets1_posneg = faceOnsets1[np.all(currData['pairsValue'] == [1, -1],1)]
            faceOnsets1_posneut = faceOnsets1[np.all(currData['pairsValue'] == [1, 0],1)]
            faceOnsets1_neutneg = faceOnsets1[np.all(currData['pairsValue'] == [0, -1],1)]
            faceOnsets1_neutneut = faceOnsets1[np.all(currData['pairsValue'] == [0, 0],1)]
            faceOnsets1_negneg = faceOnsets1[np.all(currData['pairsValue'] == [-1, -1],1)]
            faceOnsets2_posneg = faceOnsets2[np.all(currData['pairsValue'] == [1, -1],1)]
            faceOnsets2_posneut = faceOnsets2[np.all(currData['pairsValue'] == [1, 0],1)]
            faceOnsets2_neutneg = faceOnsets2[np.all(currData['pairsValue'] == [0, -1],1)]
            faceOnsets2_neutneut = faceOnsets2[np.all(currData['pairsValue'] == [0, 0],1)]
            faceOnsets2_negneg = faceOnsets2[np.all(currData['pairsValue'] == [-1, -1],1)]

            #Create vectors for pandas table - %% 
            onsets = np.zeros(2*66)
            onsets[np.arange(0,trialNum,2)] = faceOnsets1
            onsets[np.arange(1,trialNum+1,2)] = rewardOnsets1
            durations = np.zeros(2*66)
            durations[np.arange(0,trialNum,2)] = 3
            durations[np.arange(1,trialNum+1,2)] = 2
            trial_type = np.empty((trialNum),dtype='object')
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_posneg))] = 'posnegF'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_posneut))] = 'posneuF'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neutneg))] = 'neunegF'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neutneut))] = 'neuneuF'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_negneg))] = 'negnegF'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_pos))] = 'posRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neg))] = 'negRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neut))] = 'neuRewd'
            weights = np.ones(trialNum,dtype=int)
            pd1 = pd.DataFrame({'onset':onsets,'duration':durations,'weight':weights,'trial_type':trial_type})

            onsets = np.zeros(2*66)
            onsets[np.arange(0,trialNum,2)] = faceOnsets2
            onsets[np.arange(1,trialNum+1,2)] = rewardOnsets2
            durations = np.zeros(2*66)
            durations[np.arange(0,trialNum,2)] = 3
            durations[np.arange(1,trialNum+1,2)] = 2
            trial_type = np.empty((trialNum),dtype='object')
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_posneg))] = 'posnegF'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_posneut))] = 'posneuF'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neutneg))] = 'neunegF'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neutneut))] = 'neuneuF'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_negneg))] = 'negnegF'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_pos))] = 'posRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neg))] = 'negRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neut))] = 'neuRewd'
            weights = np.ones(trialNum,dtype=int)
            pd2 = pd.DataFrame({'onset':onsets,'duration':durations,'weight':weights,'trial_type':trial_type})

            #Deduct two dummy scans
            pd1['onset'] = pd1['onset']
            pd2['onset'] = pd2['onset']

            pd1.to_csv(mri_path+'/MRI preproc fMRIPrep/'+subName+'/Events/task_events1_'+fName+'.tsv')
            pd2.to_csv(mri_path+'/MRI preproc fMRIPrep/'+subName+'/Events/task_events2_'+fName+'.tsv')
            
    return pd1, pd2, fName, subNames

# decisionstim - FC Face

def DecisionStim_FCFace(behav_path, mri_path, df, sub_list):

    fName = 'DecisionStim_FCFace'
    subNames = []
    
    import numpy as np

    for subNum, subName in df.items():

        if np.isin(subNum,sub_list):

            subNames.append(subName)

            #Load mat files as dicts
            trialNum = 132
            currData = scipy.io.loadmat(behav_path+'/'+str(subNum)+'/learning_'+str(subNum)+'_session_1.mat')

            #Get reward vec
            rewardOnsets1 = currData['rewardOnset'][0][1:67]-currData['alignTimesScanner'][0][0]
            rewardOnsets2 = currData['rewardOnset'][0][67:134]-currData['alignTimesScanner'][0][1]
            rewards1 = currData['reward'][0:66,0]
            rewards2 = currData['reward'][66:133,0]
            rewardOnsets1_pos = rewardOnsets1[rewards1 == 10]
            rewardOnsets1_neg = rewardOnsets1[rewards1 == -10]
            rewardOnsets1_neut = rewardOnsets1[rewards1 == 0]
            rewardOnsets2_pos = rewardOnsets2[rewards2 == 10]
            rewardOnsets2_neg = rewardOnsets2[rewards2 == -10]
            rewardOnsets2_neut = rewardOnsets2[rewards2 == 0]

            #Get face value that was chosen
            choices = currData['subjectChoice']
            options = currData['pairsValue']
            trials = np.arange(trialNum).reshape(trialNum,1)
            faceVals = options[trials,choices-1]

            #Get face vec
            faceOnsets1 = currData['stimOnset'][0][0:66]-currData['alignTimesScanner'][0][0]
            faceOnsets2 = currData['stimOnset'][0][66:133]-currData['alignTimesScanner'][0][1]
            faceVals1 = faceVals[0:66,0]
            faceVals2 = faceVals[66:133,0]
            faceOnsets1_pos = faceOnsets1[faceVals1 == 1]
            faceOnsets1_neg = faceOnsets1[faceVals1 == -1]
            faceOnsets1_neut = faceOnsets1[faceVals1 == 0]
            faceOnsets2_pos = faceOnsets2[faceVals2 == 1]
            faceOnsets2_neg = faceOnsets2[faceVals2 == -1]
            faceOnsets2_neut = faceOnsets2[faceVals2 == 0]

            #Create vectors for pandas table - %% 
            onsets = np.zeros(2*66)
            onsets[np.arange(0,trialNum,2)] = faceOnsets1
            onsets[np.arange(1,trialNum+1,2)] = rewardOnsets1
            durations = np.zeros(2*66)
            durations[np.arange(0,trialNum,2)] = 3
            durations[np.arange(1,trialNum+1,2)] = 2
            #New trial type
            trial_type = np.empty((trialNum),dtype='<18U')
            stimNum = np.sum(np.isin(onsets,faceOnsets1_pos))
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_pos))] = np.char.add('posFace',np.arange(1,stimNum+1).astype(str))
            stimNum = np.sum(np.isin(onsets,faceOnsets1_neg))
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neg))] = np.char.add('negFace',np.arange(1,stimNum+1).astype(str))
            stimNum = np.sum(np.isin(onsets,faceOnsets1_neut))
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neut))] = np.char.add('neutFace',np.arange(1,stimNum+1).astype(str))
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_pos))] = 'posRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neg))] = 'negRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neut))] = 'neuRewd'
            #Old trial type
            old_trial_type = np.empty((trialNum),dtype='object')
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets1_pos))] = 'posFace'
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neg))] = 'negFace'
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neut))] = 'neuFace'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_pos))] = 'posRewd'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neg))] = 'negRewd'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neut))] = 'neuRewd'
            #Weights
            weights = np.ones(trialNum,dtype=int)
            pd1 = pd.DataFrame({'onset':onsets,'duration':durations,'weight':weights,'trial_type':trial_type,'old_trial_type':old_trial_type})

            onsets = np.zeros(2*66)
            onsets[np.arange(0,trialNum,2)] = faceOnsets2
            onsets[np.arange(1,trialNum+1,2)] = rewardOnsets2
            durations = np.zeros(2*66)
            durations[np.arange(0,trialNum,2)] = 3
            durations[np.arange(1,trialNum+1,2)] = 2
            #New trial type
            trial_type = np.empty((trialNum),dtype='<18U')
            stimNum = np.sum(np.isin(onsets,faceOnsets2_pos))
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_pos))] = np.char.add('posFace',np.arange(1,stimNum+1).astype(str))
            stimNum = np.sum(np.isin(onsets,faceOnsets2_neg))
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neg))] = np.char.add('negFace',np.arange(1,stimNum+1).astype(str))
            stimNum = np.sum(np.isin(onsets,faceOnsets2_neut))
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neut))] = np.char.add('neutFace',np.arange(1,stimNum+1).astype(str))
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_pos))] = 'posRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neg))] = 'negRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neut))] = 'neuRewd'
            #Old trial type
            old_trial_type = np.empty((trialNum),dtype='object')
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets2_pos))] = 'posFace'
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neg))] = 'negFace'
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neut))] = 'neuFace'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_pos))] = 'posRewd'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neg))] = 'negRewd'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neut))] = 'neuRewd'
            #Weights
            weights = np.ones(trialNum,dtype=int)
            pd2 = pd.DataFrame({'onset':onsets,'duration':durations,'weight':weights,'trial_type':trial_type,'old_trial_type':old_trial_type})

            #Deduct two dummy scans
            pd1['onset'] = pd1['onset']
            pd2['onset'] = pd2['onset']

            pd1.to_csv(mri_path+'/MRI preproc fMRIPrep/'+subName+'/Events/task_events1_'+fName+'.tsv')
            pd2.to_csv(mri_path+'/MRI preproc fMRIPrep/'+subName+'/Events/task_events2_'+fName+'.tsv')
            
    return pd1, pd2, fName, subNames

# decisionstim - FC Reward

def DecisionStim_FCRewd(behav_path, mri_path, df, sub_list):

    fName = 'DecisionStim_FCRewd'
    subNames = []
    
    import numpy as np

    for subNum, subName in df.items():

        if np.isin(subNum,sub_list):

            subNames.append(subName)

            #Load mat files as dicts
            trialNum = 132
            currData = scipy.io.loadmat(behav_path+'/'+str(subNum)+'/learning_'+str(subNum)+'_session_1.mat')

            #Get reward vec
            rewardOnsets1 = currData['rewardOnset'][0][1:67]-currData['alignTimesScanner'][0][0]
            rewardOnsets2 = currData['rewardOnset'][0][67:134]-currData['alignTimesScanner'][0][1]
            rewards1 = currData['reward'][0:66,0]
            rewards2 = currData['reward'][66:133,0]
            rewardOnsets1_pos = rewardOnsets1[rewards1 == 10]
            rewardOnsets1_neg = rewardOnsets1[rewards1 == -10]
            rewardOnsets1_neut = rewardOnsets1[rewards1 == 0]
            rewardOnsets2_pos = rewardOnsets2[rewards2 == 10]
            rewardOnsets2_neg = rewardOnsets2[rewards2 == -10]
            rewardOnsets2_neut = rewardOnsets2[rewards2 == 0]

            #Get face value that was chosen
            choices = currData['subjectChoice']
            options = currData['pairsValue']
            trials = np.arange(trialNum).reshape(trialNum,1)
            faceVals = options[trials,choices-1]

            #Get face vec
            faceOnsets1 = currData['stimOnset'][0][0:66]-currData['alignTimesScanner'][0][0]
            faceOnsets2 = currData['stimOnset'][0][66:133]-currData['alignTimesScanner'][0][1]
            faceVals1 = faceVals[0:66,0]
            faceVals2 = faceVals[66:133,0]
            faceOnsets1_pos = faceOnsets1[faceVals1 == 1]
            faceOnsets1_neg = faceOnsets1[faceVals1 == -1]
            faceOnsets1_neut = faceOnsets1[faceVals1 == 0]
            faceOnsets2_pos = faceOnsets2[faceVals2 == 1]
            faceOnsets2_neg = faceOnsets2[faceVals2 == -1]
            faceOnsets2_neut = faceOnsets2[faceVals2 == 0]

            #Create vectors for pandas table - %% 
            onsets = np.zeros(2*66)
            onsets[np.arange(0,trialNum,2)] = faceOnsets1
            onsets[np.arange(1,trialNum+1,2)] = rewardOnsets1
            durations = np.zeros(2*66)
            durations[np.arange(0,trialNum,2)] = 3
            durations[np.arange(1,trialNum+1,2)] = 2
            #New trial type
            trial_type = np.empty((trialNum),dtype='<18U')
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_pos))] = 'posFace'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neg))] = 'negFace'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neut))] = 'neuFace'
            stimNum = np.sum(np.isin(onsets,rewardOnsets1_pos))
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_pos))] = np.char.add('posRewd',np.arange(1,stimNum+1).astype(str))
            stimNum = np.sum(np.isin(onsets,rewardOnsets1_neg))
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neg))] = np.char.add('negRewd',np.arange(1,stimNum+1).astype(str))
            stimNum = np.sum(np.isin(onsets,rewardOnsets1_neut))
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neut))] = np.char.add('neutRewd',np.arange(1,stimNum+1).astype(str))
            #Old trial type
            old_trial_type = np.empty((trialNum),dtype='object')
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets1_pos))] = 'posFace'
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neg))] = 'negFace'
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neut))] = 'neuFace'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_pos))] = 'posRewd'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neg))] = 'negRewd'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neut))] = 'neuRewd'
            #Weights
            weights = np.ones(trialNum,dtype=int)
            pd1 = pd.DataFrame({'onset':onsets,'duration':durations,'weight':weights,'trial_type':trial_type,'old_trial_type':old_trial_type})

            onsets = np.zeros(2*66)
            onsets[np.arange(0,trialNum,2)] = faceOnsets2
            onsets[np.arange(1,trialNum+1,2)] = rewardOnsets2
            durations = np.zeros(2*66)
            durations[np.arange(0,trialNum,2)] = 3
            durations[np.arange(1,trialNum+1,2)] = 2
            #New trial type
            trial_type = np.empty((trialNum),dtype='<18U')
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_pos))] = 'posFace'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neg))] = 'negFace'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neut))] = 'neuFace'
            stimNum = np.sum(np.isin(onsets,rewardOnsets2_pos))
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_pos))] = np.char.add('posRewd',np.arange(1,stimNum+1).astype(str))
            stimNum = np.sum(np.isin(onsets,rewardOnsets2_neg))
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neg))] = np.char.add('negRewd',np.arange(1,stimNum+1).astype(str))
            stimNum = np.sum(np.isin(onsets,rewardOnsets2_neut))
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neut))] = np.char.add('neutRewd',np.arange(1,stimNum+1).astype(str))
            #Old trial type
            old_trial_type = np.empty((trialNum),dtype='object')
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets2_pos))] = 'posFace'
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neg))] = 'negFace'
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neut))] = 'neuFace'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_pos))] = 'posRewd'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neg))] = 'negRewd'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neut))] = 'neuRewd'
            #Weights
            weights = np.ones(trialNum,dtype=int)
            pd2 = pd.DataFrame({'onset':onsets,'duration':durations,'weight':weights,'trial_type':trial_type,'old_trial_type':old_trial_type})

            #Deduct two dummy scans
            pd1['onset'] = pd1['onset']
            pd2['onset'] = pd2['onset']

            pd1.to_csv(mri_path+'/MRI preproc fMRIPrep/'+subName+'/Events/task_events1_'+fName+'.tsv')
            pd2.to_csv(mri_path+'/MRI preproc fMRIPrep/'+subName+'/Events/task_events2_'+fName+'.tsv')
            
    return pd1, pd2, fName, subNames


# 200 model - FC Face

# Loop through subs

def Decision200_FCFace(behav_path, mri_path, df, sub_list):

    decConstant = 0.2 #200 ms deducted
    
    fName = 'Decision200_FCFace'
    subNames = []

    for subNum, subName in df.items():

        if np.isin(subNum,sub_list):

            subNames.append(subName)

            #Load mat files as dicts
            trialNum = 132
            currData = scipy.io.loadmat(behav_path+'/'+str(subNum)+'/learning_'+str(subNum)+'_session_1.mat')

            #Get reward vec
            rewardOnsets1 = currData['rewardOnset'][0][1:67]-currData['alignTimesScanner'][0][0]
            rewardOnsets2 = currData['rewardOnset'][0][67:134]-currData['alignTimesScanner'][0][1]
            rewards1 = currData['reward'][0:66,0]
            rewards2 = currData['reward'][66:133,0]
            rewardOnsets1_pos = rewardOnsets1[rewards1 == 10]
            rewardOnsets1_neg = rewardOnsets1[rewards1 == -10]
            rewardOnsets1_neut = rewardOnsets1[rewards1 == 0]
            rewardOnsets2_pos = rewardOnsets2[rewards2 == 10]
            rewardOnsets2_neg = rewardOnsets2[rewards2 == -10]
            rewardOnsets2_neut = rewardOnsets2[rewards2 == 0]

            #Get face value that was chosen
            choices = currData['subjectChoice']
            options = currData['pairsValue']
            trials = np.arange(trialNum).reshape(trialNum,1)
            faceVals = options[trials,choices-1]

            #Get face vec
            faceOnsets1 = currData['rectOnset'][0][0:66]-currData['alignTimesScanner'][0][0]-decConstant
            faceOnsets2 = currData['rectOnset'][0][66:133]-currData['alignTimesScanner'][0][1]-decConstant
            faceVals1 = faceVals[0:66,0]
            faceVals2 = faceVals[66:133,0]
            faceOnsets1_pos = faceOnsets1[faceVals1 == 1]
            faceOnsets1_neg = faceOnsets1[faceVals1 == -1]
            faceOnsets1_neut = faceOnsets1[faceVals1 == 0]
            faceOnsets2_pos = faceOnsets2[faceVals2 == 1]
            faceOnsets2_neg = faceOnsets2[faceVals2 == -1]
            faceOnsets2_neut = faceOnsets2[faceVals2 == 0]

            #Create vectors for pandas table - %% 
            onsets = np.zeros(2*66)
            onsets[np.arange(0,trialNum,2)] = faceOnsets1
            onsets[np.arange(1,trialNum+1,2)] = rewardOnsets1
            durations = np.zeros(2*66)
            durations[np.arange(0,trialNum,2)] = 0
            durations[np.arange(1,trialNum+1,2)] = 2
            #New trial type
            trial_type = np.empty((trialNum),dtype='<18U')
            stimNum = np.sum(np.isin(onsets,faceOnsets1_pos))
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_pos))] = np.char.add('posFace',np.arange(1,stimNum+1).astype(str))
            stimNum = np.sum(np.isin(onsets,faceOnsets1_neg))
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neg))] = np.char.add('negFace',np.arange(1,stimNum+1).astype(str))
            stimNum = np.sum(np.isin(onsets,faceOnsets1_neut))
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neut))] = np.char.add('neutFace',np.arange(1,stimNum+1).astype(str))
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_pos))] = 'posRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neg))] = 'negRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neut))] = 'neuRewd'
            #Old trial type
            old_trial_type = np.empty((trialNum),dtype='object')
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets1_pos))] = 'posFace'
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neg))] = 'negFace'
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neut))] = 'neuFace'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_pos))] = 'posRewd'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neg))] = 'negRewd'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neut))] = 'neuRewd'
            #Weights
            weights = np.ones(trialNum,dtype=int)
            pd1 = pd.DataFrame({'onset':onsets,'duration':durations,'weight':weights,'trial_type':trial_type,'old_trial_type':old_trial_type})

            onsets = np.zeros(2*66)
            onsets[np.arange(0,trialNum,2)] = faceOnsets2
            onsets[np.arange(1,trialNum+1,2)] = rewardOnsets2
            durations = np.zeros(2*66)
            durations[np.arange(0,trialNum,2)] = 0
            durations[np.arange(1,trialNum+1,2)] = 2
            #New trial type
            trial_type = np.empty((trialNum),dtype='<18U')
            stimNum = np.sum(np.isin(onsets,faceOnsets2_pos))
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_pos))] = np.char.add('posFace',np.arange(1,stimNum+1).astype(str))
            stimNum = np.sum(np.isin(onsets,faceOnsets2_neg))
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neg))] = np.char.add('negFace',np.arange(1,stimNum+1).astype(str))
            stimNum = np.sum(np.isin(onsets,faceOnsets2_neut))
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neut))] = np.char.add('neutFace',np.arange(1,stimNum+1).astype(str))
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_pos))] = 'posRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neg))] = 'negRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neut))] = 'neuRewd'
            #Old trial type
            old_trial_type = np.empty((trialNum),dtype='object')
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets2_pos))] = 'posFace'
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neg))] = 'negFace'
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neut))] = 'neuFace'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_pos))] = 'posRewd'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neg))] = 'negRewd'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neut))] = 'neuRewd'
            #Weights
            weights = np.ones(trialNum,dtype=int)
            pd2 = pd.DataFrame({'onset':onsets,'duration':durations,'weight':weights,'trial_type':trial_type,'old_trial_type':old_trial_type})

            #Deduct two dummy scans
            pd1['onset'] = pd1['onset']
            pd2['onset'] = pd2['onset']

            pd1.to_csv(mri_path+'/MRI preproc fMRIPrep/'+subName+'/Events/task_events1_'+fName+'.tsv')
            pd2.to_csv(mri_path+'/MRI preproc fMRIPrep/'+subName+'/Events/task_events2_'+fName+'.tsv')
            
    return pd1, pd2, fName, subNames

# 200 model - FC Reward

# Loop through subs

def Decision200_FCRewd(behav_path, mri_path, df, sub_list):

    decConstant = 0.2 #200 ms deducted
    
    fName = 'Decision200_FCRewd'
    subNames = []

    for subNum, subName in df.items():

        if np.isin(subNum,sub_list):

            subNames.append(subName)

            #Load mat files as dicts
            trialNum = 132
            currData = scipy.io.loadmat(behav_path+'/'+str(subNum)+'/learning_'+str(subNum)+'_session_1.mat')

            #Get reward vec
            rewardOnsets1 = currData['rewardOnset'][0][1:67]-currData['alignTimesScanner'][0][0]
            rewardOnsets2 = currData['rewardOnset'][0][67:134]-currData['alignTimesScanner'][0][1]
            rewards1 = currData['reward'][0:66,0]
            rewards2 = currData['reward'][66:133,0]
            rewardOnsets1_pos = rewardOnsets1[rewards1 == 10]
            rewardOnsets1_neg = rewardOnsets1[rewards1 == -10]
            rewardOnsets1_neut = rewardOnsets1[rewards1 == 0]
            rewardOnsets2_pos = rewardOnsets2[rewards2 == 10]
            rewardOnsets2_neg = rewardOnsets2[rewards2 == -10]
            rewardOnsets2_neut = rewardOnsets2[rewards2 == 0]

            #Get face value that was chosen
            choices = currData['subjectChoice']
            options = currData['pairsValue']
            trials = np.arange(trialNum).reshape(trialNum,1)
            faceVals = options[trials,choices-1]

            #Get face vec
            faceOnsets1 = currData['rectOnset'][0][0:66]-currData['alignTimesScanner'][0][0]-decConstant
            faceOnsets2 = currData['rectOnset'][0][66:133]-currData['alignTimesScanner'][0][1]-decConstant
            faceVals1 = faceVals[0:66,0]
            faceVals2 = faceVals[66:133,0]
            faceOnsets1_pos = faceOnsets1[faceVals1 == 1]
            faceOnsets1_neg = faceOnsets1[faceVals1 == -1]
            faceOnsets1_neut = faceOnsets1[faceVals1 == 0]
            faceOnsets2_pos = faceOnsets2[faceVals2 == 1]
            faceOnsets2_neg = faceOnsets2[faceVals2 == -1]
            faceOnsets2_neut = faceOnsets2[faceVals2 == 0]

            #Create vectors for pandas table - %% 
            onsets = np.zeros(2*66)
            onsets[np.arange(0,trialNum,2)] = faceOnsets1
            onsets[np.arange(1,trialNum+1,2)] = rewardOnsets1
            durations = np.zeros(2*66)
            durations[np.arange(0,trialNum,2)] = 0
            durations[np.arange(1,trialNum+1,2)] = 2
            #New trial type
            trial_type = np.empty((trialNum),dtype='<18U')
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_pos))] = 'posFace'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neg))] = 'negFace'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neut))] = 'neuFace'
            stimNum = np.sum(np.isin(onsets,rewardOnsets1_pos))
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_pos))] = np.char.add('posRewd',np.arange(1,stimNum+1).astype(str))
            stimNum = np.sum(np.isin(onsets,rewardOnsets1_neg))
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neg))] = np.char.add('negRewd',np.arange(1,stimNum+1).astype(str))
            stimNum = np.sum(np.isin(onsets,rewardOnsets1_neut))
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neut))] = np.char.add('neutRewd',np.arange(1,stimNum+1).astype(str))
            #Old trial type
            old_trial_type = np.empty((trialNum),dtype='object')
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets1_pos))] = 'posFace'
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neg))] = 'negFace'
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neut))] = 'neuFace'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_pos))] = 'posRewd'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neg))] = 'negRewd'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neut))] = 'neuRewd'
            #Weights
            weights = np.ones(trialNum,dtype=int)
            pd1 = pd.DataFrame({'onset':onsets,'duration':durations,'weight':weights,'trial_type':trial_type,'old_trial_type':old_trial_type})

            onsets = np.zeros(2*66)
            onsets[np.arange(0,trialNum,2)] = faceOnsets2
            onsets[np.arange(1,trialNum+1,2)] = rewardOnsets2
            durations = np.zeros(2*66)
            durations[np.arange(0,trialNum,2)] = 0
            durations[np.arange(1,trialNum+1,2)] = 2
            #New trial type
            trial_type = np.empty((trialNum),dtype='<18U')
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_pos))] = 'posFace'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neg))] = 'negFace'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neut))] = 'neuFace'
            stimNum = np.sum(np.isin(onsets,rewardOnsets2_pos))
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_pos))] = np.char.add('posRewd',np.arange(1,stimNum+1).astype(str))
            stimNum = np.sum(np.isin(onsets,rewardOnsets2_neg))
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neg))] = np.char.add('negRewd',np.arange(1,stimNum+1).astype(str))
            stimNum = np.sum(np.isin(onsets,rewardOnsets2_neut))
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neut))] = np.char.add('neutRewd',np.arange(1,stimNum+1).astype(str))
            #Old trial type
            old_trial_type = np.empty((trialNum),dtype='object')
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets2_pos))] = 'posFace'
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neg))] = 'negFace'
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neut))] = 'neuFace'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_pos))] = 'posRewd'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neg))] = 'negRewd'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neut))] = 'neuRewd'
            #Weights
            weights = np.ones(trialNum,dtype=int)
            pd2 = pd.DataFrame({'onset':onsets,'duration':durations,'weight':weights,'trial_type':trial_type,'old_trial_type':old_trial_type})

            #Deduct two dummy scans
            pd1['onset'] = pd1['onset']
            pd2['onset'] = pd2['onset']

            pd1.to_csv(mri_path+'/MRI preproc fMRIPrep/'+subName+'/Events/task_events1_'+fName+'.tsv')
            pd2.to_csv(mri_path+'/MRI preproc fMRIPrep/'+subName+'/Events/task_events2_'+fName+'.tsv')
            
    return pd1, pd2, fName, subNames


# 0 model - FC Face

# Loop through subs

def Decision0_FCFace(behav_path, mri_path, df, sub_list):

    decConstant = 0 #0 ms deducted
    
    fName = 'Decision0_FCFace'
    subNames = []

    for subNum, subName in df.items():

        if np.isin(subNum,sub_list):

            subNames.append(subName)

            #Load mat files as dicts
            trialNum = 132
            currData = scipy.io.loadmat(behav_path+'/'+str(subNum)+'/learning_'+str(subNum)+'_session_1.mat')

            #Get reward vec
            rewardOnsets1 = currData['rewardOnset'][0][1:67]-currData['alignTimesScanner'][0][0]
            rewardOnsets2 = currData['rewardOnset'][0][67:134]-currData['alignTimesScanner'][0][1]
            rewards1 = currData['reward'][0:66,0]
            rewards2 = currData['reward'][66:133,0]
            rewardOnsets1_pos = rewardOnsets1[rewards1 == 10]
            rewardOnsets1_neg = rewardOnsets1[rewards1 == -10]
            rewardOnsets1_neut = rewardOnsets1[rewards1 == 0]
            rewardOnsets2_pos = rewardOnsets2[rewards2 == 10]
            rewardOnsets2_neg = rewardOnsets2[rewards2 == -10]
            rewardOnsets2_neut = rewardOnsets2[rewards2 == 0]

            #Get face value that was chosen
            choices = currData['subjectChoice']
            options = currData['pairsValue']
            trials = np.arange(trialNum).reshape(trialNum,1)
            faceVals = options[trials,choices-1]

            #Get face vec
            faceOnsets1 = currData['rectOnset'][0][0:66]-currData['alignTimesScanner'][0][0]-decConstant
            faceOnsets2 = currData['rectOnset'][0][66:133]-currData['alignTimesScanner'][0][1]-decConstant
            faceVals1 = faceVals[0:66,0]
            faceVals2 = faceVals[66:133,0]
            faceOnsets1_pos = faceOnsets1[faceVals1 == 1]
            faceOnsets1_neg = faceOnsets1[faceVals1 == -1]
            faceOnsets1_neut = faceOnsets1[faceVals1 == 0]
            faceOnsets2_pos = faceOnsets2[faceVals2 == 1]
            faceOnsets2_neg = faceOnsets2[faceVals2 == -1]
            faceOnsets2_neut = faceOnsets2[faceVals2 == 0]

            #Create vectors for pandas table - %% 
            onsets = np.zeros(2*66)
            onsets[np.arange(0,trialNum,2)] = faceOnsets1
            onsets[np.arange(1,trialNum+1,2)] = rewardOnsets1
            durations = np.zeros(2*66)
            durations[np.arange(0,trialNum,2)] = 0
            durations[np.arange(1,trialNum+1,2)] = 2
            #New trial type
            trial_type = np.empty((trialNum),dtype='<18U')
            stimNum = np.sum(np.isin(onsets,faceOnsets1_pos))
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_pos))] = np.char.add('posFace',np.arange(1,stimNum+1).astype(str))
            stimNum = np.sum(np.isin(onsets,faceOnsets1_neg))
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neg))] = np.char.add('negFace',np.arange(1,stimNum+1).astype(str))
            stimNum = np.sum(np.isin(onsets,faceOnsets1_neut))
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neut))] = np.char.add('neutFace',np.arange(1,stimNum+1).astype(str))
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_pos))] = 'posRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neg))] = 'negRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neut))] = 'neuRewd'
            #Old trial type
            old_trial_type = np.empty((trialNum),dtype='object')
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets1_pos))] = 'posFace'
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neg))] = 'negFace'
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neut))] = 'neuFace'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_pos))] = 'posRewd'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neg))] = 'negRewd'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neut))] = 'neuRewd'
            #Weights
            weights = np.ones(trialNum,dtype=int)
            pd1 = pd.DataFrame({'onset':onsets,'duration':durations,'weight':weights,'trial_type':trial_type,'old_trial_type':old_trial_type})

            onsets = np.zeros(2*66)
            onsets[np.arange(0,trialNum,2)] = faceOnsets2
            onsets[np.arange(1,trialNum+1,2)] = rewardOnsets2
            durations = np.zeros(2*66)
            durations[np.arange(0,trialNum,2)] = 0
            durations[np.arange(1,trialNum+1,2)] = 2
            #New trial type
            trial_type = np.empty((trialNum),dtype='<18U')
            stimNum = np.sum(np.isin(onsets,faceOnsets2_pos))
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_pos))] = np.char.add('posFace',np.arange(1,stimNum+1).astype(str))
            stimNum = np.sum(np.isin(onsets,faceOnsets2_neg))
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neg))] = np.char.add('negFace',np.arange(1,stimNum+1).astype(str))
            stimNum = np.sum(np.isin(onsets,faceOnsets2_neut))
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neut))] = np.char.add('neutFace',np.arange(1,stimNum+1).astype(str))
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_pos))] = 'posRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neg))] = 'negRewd'
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neut))] = 'neuRewd'
            #Old trial type
            old_trial_type = np.empty((trialNum),dtype='object')
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets2_pos))] = 'posFace'
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neg))] = 'negFace'
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neut))] = 'neuFace'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_pos))] = 'posRewd'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neg))] = 'negRewd'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neut))] = 'neuRewd'
            #Weights
            weights = np.ones(trialNum,dtype=int)
            pd2 = pd.DataFrame({'onset':onsets,'duration':durations,'weight':weights,'trial_type':trial_type,'old_trial_type':old_trial_type})

            #Deduct two dummy scans
            pd1['onset'] = pd1['onset']
            pd2['onset'] = pd2['onset']

            pd1.to_csv(mri_path+'/MRI preproc fMRIPrep/'+subName+'/Events/task_events1_'+fName+'.tsv')
            pd2.to_csv(mri_path+'/MRI preproc fMRIPrep/'+subName+'/Events/task_events2_'+fName+'.tsv')
            
    return pd1, pd2, fName, subNames

# 0 model - FC Reward

# Loop through subs

def Decision0_FCRewd(behav_path, mri_path, df, sub_list):

    decConstant = 0 #0 ms deducted
    
    fName = 'Decision0_FCRewd'
    subNames = []

    for subNum, subName in df.items():

        if np.isin(subNum,sub_list):

            subNames.append(subName)

            #Load mat files as dicts
            trialNum = 132
            currData = scipy.io.loadmat(behav_path+'/'+str(subNum)+'/learning_'+str(subNum)+'_session_1.mat')

            #Get reward vec
            rewardOnsets1 = currData['rewardOnset'][0][1:67]-currData['alignTimesScanner'][0][0]
            rewardOnsets2 = currData['rewardOnset'][0][67:134]-currData['alignTimesScanner'][0][1]
            rewards1 = currData['reward'][0:66,0]
            rewards2 = currData['reward'][66:133,0]
            rewardOnsets1_pos = rewardOnsets1[rewards1 == 10]
            rewardOnsets1_neg = rewardOnsets1[rewards1 == -10]
            rewardOnsets1_neut = rewardOnsets1[rewards1 == 0]
            rewardOnsets2_pos = rewardOnsets2[rewards2 == 10]
            rewardOnsets2_neg = rewardOnsets2[rewards2 == -10]
            rewardOnsets2_neut = rewardOnsets2[rewards2 == 0]

            #Get face value that was chosen
            choices = currData['subjectChoice']
            options = currData['pairsValue']
            trials = np.arange(trialNum).reshape(trialNum,1)
            faceVals = options[trials,choices-1]

            #Get face vec
            faceOnsets1 = currData['rectOnset'][0][0:66]-currData['alignTimesScanner'][0][0]-decConstant
            faceOnsets2 = currData['rectOnset'][0][66:133]-currData['alignTimesScanner'][0][1]-decConstant
            faceVals1 = faceVals[0:66,0]
            faceVals2 = faceVals[66:133,0]
            faceOnsets1_pos = faceOnsets1[faceVals1 == 1]
            faceOnsets1_neg = faceOnsets1[faceVals1 == -1]
            faceOnsets1_neut = faceOnsets1[faceVals1 == 0]
            faceOnsets2_pos = faceOnsets2[faceVals2 == 1]
            faceOnsets2_neg = faceOnsets2[faceVals2 == -1]
            faceOnsets2_neut = faceOnsets2[faceVals2 == 0]

            #Create vectors for pandas table - %% 
            onsets = np.zeros(2*66)
            onsets[np.arange(0,trialNum,2)] = faceOnsets1
            onsets[np.arange(1,trialNum+1,2)] = rewardOnsets1
            durations = np.zeros(2*66)
            durations[np.arange(0,trialNum,2)] = 0
            durations[np.arange(1,trialNum+1,2)] = 2
            
            #New trial type
            trial_type = np.empty((trialNum),dtype='<18U')
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_pos))] = 'posFace'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neg))] = 'negFace'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neut))] = 'neuFace'
            stimNum = np.sum(np.isin(onsets,rewardOnsets1_pos))
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_pos))] = np.char.add('posRewd',np.arange(1,stimNum+1).astype(str))
            stimNum = np.sum(np.isin(onsets,rewardOnsets1_neg))
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neg))] = np.char.add('negRewd',np.arange(1,stimNum+1).astype(str))
            stimNum = np.sum(np.isin(onsets,rewardOnsets1_neut))
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neut))] = np.char.add('neutRewd',np.arange(1,stimNum+1).astype(str))
            #Old trial type
            old_trial_type = np.empty((trialNum),dtype='object')
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets1_pos))] = 'posFace'
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neg))] = 'negFace'
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets1_neut))] = 'neuFace'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_pos))] = 'posRewd'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neg))] = 'negRewd'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets1_neut))] = 'neuRewd'
            #Weights
            weights = np.ones(trialNum,dtype=int)
            pd1 = pd.DataFrame({'onset':onsets,'duration':durations,'weight':weights,'trial_type':trial_type,'old_trial_type':old_trial_type})

            onsets = np.zeros(2*66)
            onsets[np.arange(0,trialNum,2)] = faceOnsets2
            onsets[np.arange(1,trialNum+1,2)] = rewardOnsets2
            durations = np.zeros(2*66)
            durations[np.arange(0,trialNum,2)] = 0
            durations[np.arange(1,trialNum+1,2)] = 2
            #New trial type
            trial_type = np.empty((trialNum),dtype='<18U')
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_pos))] = 'posFace'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neg))] = 'negFace'
            trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neut))] = 'neuFace'
            stimNum = np.sum(np.isin(onsets,rewardOnsets2_pos))
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_pos))] = np.char.add('posRewd',np.arange(1,stimNum+1).astype(str))
            stimNum = np.sum(np.isin(onsets,rewardOnsets2_neg))
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neg))] = np.char.add('negRewd',np.arange(1,stimNum+1).astype(str))
            stimNum = np.sum(np.isin(onsets,rewardOnsets2_neut))
            trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neut))] = np.char.add('neutRewd',np.arange(1,stimNum+1).astype(str))
            #Old trial type
            old_trial_type = np.empty((trialNum),dtype='object')
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets2_pos))] = 'posFace'
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neg))] = 'negFace'
            old_trial_type[np.nonzero(np.isin(onsets,faceOnsets2_neut))] = 'neuFace'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_pos))] = 'posRewd'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neg))] = 'negRewd'
            old_trial_type[np.nonzero(np.isin(onsets,rewardOnsets2_neut))] = 'neuRewd'
            #Weights
            weights = np.ones(trialNum,dtype=int)
            pd2 = pd.DataFrame({'onset':onsets,'duration':durations,'weight':weights,'trial_type':trial_type,'old_trial_type':old_trial_type})

            #Deduct two dummy scans
            pd1['onset'] = pd1['onset']
            pd2['onset'] = pd2['onset']

            pd1.to_csv(mri_path+'/MRI preproc fMRIPrep/'+subName+'/Events/task_events1_'+fName+'.tsv')
            pd2.to_csv(mri_path+'/MRI preproc fMRIPrep/'+subName+'/Events/task_events2_'+fName+'.tsv')
            
    return pd1, pd2, fName, subNames


            

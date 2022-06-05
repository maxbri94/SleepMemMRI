
#Define contrasts

import numpy as np
import scipy.io
import nilearn

def create_DesignMatrix_Original(DM_size=6):
    columnAdd = DM_size-6
    contrasts = {
        'averageFace': np.pad(np.array([1/3, 0, 1/3, 0, 1/3, 0]),(0, columnAdd),constant_values=(0, 0)),
        'negFace':   np.pad(np.array([1, 0, 0, 0, 0, 0]),(0, columnAdd),constant_values=(0, 0)),
        'neutFace':   np.pad(np.array([0, 0, 1, 0, 0, 0]),(0, columnAdd),constant_values=(0, 0)),
        'posFace':   np.pad(np.array([0, 0, 0, 0, 1, 0]),(0, columnAdd),constant_values=(0, 0)),
        'averageReward':   np.pad(np.array([0, 1/3, 0, 1/3, 0, 1/3]),(0, columnAdd),constant_values=(0, 0)),
        'negReward':   np.pad(np.array([0, 1, 0, 0, 0, 0]),(0, columnAdd),constant_values=(0, 0)),
        'neutReward':   np.pad(np.array([0, 0, 0, 1, 0, 0]),(0, columnAdd),constant_values=(0, 0)),
        'posReward':   np.pad(np.array([0, 0, 0, 0, 0, 1]),(0, columnAdd),constant_values=(0, 0)),
        'negposReward':   np.pad(np.array([0, -1, 0, 0, 0, 1]),(0, columnAdd),constant_values=(0, 0)),
        'neutnegReward':   np.pad(np.array([0, -1, 0, 1, 0, 0]),(0, columnAdd),constant_values=(0, 0)),
        'posneutReward':   np.pad(np.array([0, 0, 0, -1, 0, 1]),(0, columnAdd),constant_values=(0, 0)),
        'negposFace':   np.pad(np.array([-1, 0, 0, 0, 1, 0]),(0, columnAdd),constant_values=(0, 0)),
        'neutnegFace':   np.pad(np.array([-1, 0, 1, 0, 0, 0]),(0, columnAdd),constant_values=(0, 0)),
        'posneutFace':   np.pad(np.array([0, 0, -1, 0, 1, 0]),(0, columnAdd),constant_values=(0, 0))
    }
    return contrasts

def create_DesignMatrix_Decision(DM_size=6):
    columnAdd = DM_size-6
    contrasts = {
        'negposReward':   np.pad(np.array([0, -1, 0, 0, 0, 1]),(0, columnAdd),constant_values=(0, 0)),
        'neutnegReward':   np.pad(np.array([0, -1, 0, 1, 0, 0]),(0, columnAdd),constant_values=(0, 0)),
        'posneutReward':   np.pad(np.array([0, 0, 0, -1, 0, 1]),(0, columnAdd),constant_values=(0, 0)),
        'negposFace':   np.pad(np.array([-1, 0, 0, 0, 1, 0]),(0, columnAdd),constant_values=(0, 0)),
        'neutnegFace':   np.pad(np.array([-1, 0, 1, 0, 0, 0]),(0, columnAdd),constant_values=(0, 0)),
        'posneutFace':   np.pad(np.array([0, 0, -1, 0, 1, 0]),(0, columnAdd),constant_values=(0, 0))
    }
    return contrasts

def create_DesignMatrix_facePairs(DM_size=8):
    columnAdd = DM_size-8
    contrasts = {
        'Helpless':   np.pad(np.array([1/2, -1/3, 1/2, -1/3, -1/3, 0, 0, 0]),(0, columnAdd),constant_values=(0, 0)),
        'negneuHelp':   np.pad(np.array([1, 0, -1, 0, 0, 0, 0, 0]),(0, columnAdd),constant_values=(0, 0)),
        'posneuFace':   np.pad(np.array([0, 1, 0, -1, 0, 0, 0, 0]),(0, columnAdd),constant_values=(0, 0)),
        'posnegFace':   np.pad(np.array([0, 1, 0, 0, -1, 0, 0, 0]),(0, columnAdd),constant_values=(0, 0)),
        'neunegFace':   np.pad(np.array([0, 0, 0, 1, -1, 0, 0, 0]),(0, columnAdd),constant_values=(0, 0)),
        'negposReward':   np.pad(np.array([0, 0, 0, 0, 0, 1, 0, -1]),(0, columnAdd),constant_values=(0, 0)),
        'neutnegReward':   np.pad(np.array([0, 0, 0, 0, 0, 1, -1, 0]),(0, columnAdd),constant_values=(0, 0)),
        'posneutReward':   np.pad(np.array([0, 0, 0, 0, 0, 0, 1, -1]),(0, columnAdd),constant_values=(0, 0)),
    }
    return contrasts

def create_ROImasks(sampleIMG, brainAreas, atlas):
#Create mask of area of interest
    
    ROImasks = {}
    for area in brainAreas:
        
        #Get area image as logical
        area_logic = np.logical_or(brainAreas[area]['Left'], brainAreas[area]['Right'])

        #Save brain area boolean in image and resample output to brain scan
        img_from_atlas = nilearn.image.new_img_like(atlas.maps,area_logic)
        img_from_atlas_resampled = nilearn.image.resample_to_img(img_from_atlas, sampleIMG, interpolation = 'nearest')
        
        #Store in dict
        ROImasks[area] = img_from_atlas_resampled
        
    return ROImasks
    
    
    
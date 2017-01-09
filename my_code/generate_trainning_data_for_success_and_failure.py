''' Currently this program requires clean data set. Meaning: 3 folders only: Segments, 
Composites, and llBehaviors. Each folder must have only 6 files associated with _Fx, 
_Fy, _Fz, _Mx, _My, and _Mz. There should also be a State.dat file outside the three folders.
If more data is encountered it will not be processed at this time.'''



import os
import shutil

from copy import deepcopy

import util.output_features                     as output_features
import data_parser.data_folder_parser           as data_folder_parser
import feature_extractor.data_feature_extractor as data_feature_extractor

import traceback,sys#,code

# Globals
global DB_PRINT
DB_PRINT=0

def main():
    from inc.config import states
    from inc.config import levels 
    from inc.config import axes 
    # What kind of success_strategy will you analyze
    success_strategy='SIM_HIRO_ONE_SA_SUCCESS'
    failure_strategy="SIM_HIRO_ONE_SA_ERROR_CHARAC_prob"
    strategy=success_strategy # default value. used in hblstates

    # Folder names
    data_folder_names=[]        # Filtered to only take relevant folders
    orig_data_folder_names=[]

    # Dictionary building blocks
    folder_dims={}
    dict_dims={}
    dict_all={}
    allTrialLabels={}

    # Set program paths
    results_dir="/Users/sklaw_mba/Desktop/ex/dr_juan_proj/workshop/data"
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = cur_dir
    os.chdir(base_dir)

    # my training data
    directory='my_training_data'

    # What kind of data should we collect?
    # - Success
    # - Failure
    # + Generate high level states data

    # 1. Get data for success tasks for a given success_strategy

    strategy=success_strategy    
    hlb_dir=strategy
    if not os.path.exists(os.path.join(base_dir, '..', 'my_training_data', hlb_dir)):
        os.makedirs(os.path.join(base_dir, '..', 'my_training_data', hlb_dir))
    
    
    # Get Folder names
    #data_folder_prefix = os.path.join(base_dir, '..', 'my_data', success_strategy)
    data_folder_prefix = os.path.join(results_dir, strategy)
    orig_data_folder_names = os.listdir(data_folder_prefix)
    
    # Remove undesired folders
    for data_folder_name in orig_data_folder_names:
        data_folder_names.append(data_folder_name)
    
    # Create a dictionary structure for all trials, RCBHT levels, and axis.
    for data_folder_name in data_folder_names:
        data_folder_full_path = os.path.join(data_folder_prefix, data_folder_name)
        if DB_PRINT:
            print data_folder_full_path
        
        dict_cooked_from_folder = data_folder_parser.parse_folder(data_folder_full_path)
        if dict_cooked_from_folder == None:
            continue
        else:
            dict_all[data_folder_name]=dict_cooked_from_folder

    if bool(dict_all):
        success_dict_all = dict_all;
    else:
        raise Exception('The success dictionary dict_all is empty') 
    
    # Clear up
    folder_dims={}
    dict_dims={}
    dict_all={}
    allTrialLabels={}
    data_folder_names=[]        
    orig_data_folder_names=[]
#-------------------------------------------------------------------------
##FAILURE ANALYSIS
#------------------------------------------------------------------------_
    
    strategy=failure_strategy
    hlb_dir=strategy
    if not os.path.exists(os.path.join(base_dir, '..', 'my_training_data', hlb_dir)):
        os.makedirs(os.path.join(base_dir, '..', 'my_training_data', hlb_dir))
    
    # Read failure data
    data_folder_prefix = os.path.join(results_dir, failure_strategy)
    orig_data_folder_names = os.listdir(data_folder_prefix)

    # Remove undesired folders
    for data_folder_name in orig_data_folder_names:
        data_folder_names.append(data_folder_name)   
            
    
    # Get full path for each folder name
    for data_folder_name in data_folder_names:
        data_folder_full_path = os.path.join(data_folder_prefix, data_folder_name)
        if DB_PRINT:
            print data_folder_full_path
        
        # Get dictionary cooked from all folders
        dict_cooked_from_folder = data_folder_parser.parse_folder(data_folder_full_path)        
        if dict_cooked_from_folder == None:
            continue
        else:
            dict_all[data_folder_name]=dict_cooked_from_folder

    # Once dict_cooked_from_folder exists, get dimensions of level/axis for each folder
    if bool(dict_all):        
        fail_dict_all = dict_all;
    else:
        raise Exception('The failure dictionary dict_all is empty')

    # Clear up
    folder_dims={}
    dict_dims={}
    dict_all={}
    allTrialLabels={}
    data_folder_names=[]        
    orig_data_folder_names=[]

    for level in levels:
        folder_dims[level] = {}
        for axis in axes:
            folder_dims[level][axis]=0

    #cook folder_dims for both success&fail samples
    for dict_all in [success_dict_all, fail_dict_all]:
        for data_folder_name in dict_all:
            for level in levels:
                for axis in axes: 
                    temp = len(dict_all[data_folder_name][level][axis])
                    if temp > folder_dims[level][axis]:
                        folder_dims[level][axis] = temp

    #output data for success
    dict_all = success_dict_all
    for data_folder_name in dict_all:        
        data_feature_extractor.extract_features(dict_all[data_folder_name],folder_dims)
        allTrialLabels[data_folder_name]=deepcopy(dict_all[data_folder_name])  
    file_for_S_classification = open(os.path.join(base_dir, '..', 'my_training_data', success_strategy, 'training_set_of_success'), 'w')
    output_features.output_sample_one_trial(file_for_S_classification, '1', allTrialLabels, os.path.join(base_dir,'..', 'my_training_data', success_strategy, "img_of_success.png"))

    dict_dims={}
    dict_all={}
    allTrialLabels={}
    data_folder_names=[]        
    orig_data_folder_names=[]

    #output data for fail
    dict_all = fail_dict_all
    for data_folder_name in dict_all:        
        data_feature_extractor.extract_features(dict_all[data_folder_name],folder_dims)
        allTrialLabels[data_folder_name]=deepcopy(dict_all[data_folder_name])  
    file_for_F_classification = open(os.path.join(base_dir, '..', 'my_training_data', failure_strategy, 'training_set_of_fail'), 'w')
    output_features.output_sample_one_trial(file_for_F_classification, '0', allTrialLabels, os.path.join(base_dir,'..', 'my_training_data', failure_strategy, "img_of_fail.png")); 

    # Clear up
    folder_dims={}
    dict_dims={}
    dict_all={}
    allTrialLabels={}
    data_folder_names=[]        
    orig_data_folder_names=[]



main();

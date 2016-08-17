''' Currently this program requires clean data set. Meaning: 3 folders only: Segments, 
Composites, and llBehaviors. Each folder must have only 6 files associated with _Fx, 
_Fy, _Fz, _Mx, _My, and _Mz. There should also be a State.dat file outside the three folders.
If more data is encountered it will not be processed at this time.'''

import os
import shutil

from copy import deepcopy

import data_parser.data_folder_parser as data_folder_parser
import feature_extractor.data_feature_extractor as data_feature_extractor
import util.output_features as output_features

import ipdb
ipdb.set_trace()

# What kind of success_strategy will you analyze
success_strategy='SIM_HIRO_ONE_SA_SUCCESS'
failure_strategy="SIM_HIRO_ONE_SA_FAILURE"

## Flags
output_per_one_trial_flag=0 # if true, output is performed for each trial for all axis. Otherwise all trials for each axis is displayed.

# Set program paths
results_dir='/home/vmrguser/sc/research/AIST/Results/ForceControl/'
cur_dir = os.path.dirname(os.path.realpath(__file__))
base_dir = cur_dir
os.chdir(base_dir)

# my training data
directory='my_training_data'
if not os.path.exists(os.path.join(base_dir, '..', directory)):
    os.makedirs(os.path.join(base_dir, '..', directory))
else:
    shutil.rmtree(os.path.join(base_dir, '..', 'my_training_data'))
    os.makedirs(os.path.join(base_dir, '..', 'my_training_data'))

# allTrials_success
directory='allTrials_success'
if not os.path.exists(os.path.join(base_dir, '..', 'my_training_data',success_strategy,directory)):
    os.makedirs(os.path.join(base_dir, '..', 'my_training_data',success_strategy,directory))    

# What kind of data should we collect?
# - Success
# - Failure
# + Generate high level states data
tasks = ["sucess or fail", "high level states"]

# 1. Get data for success tasks for a given success_strategy
if "sucess or fail" in tasks: 
    file_for_success_set = open(os.path.join(base_dir, '..', 'my_training_data', 'training_set_of_success'), 'w')

    # Get Folder names
    data_folder_names=[]        # Filtered to only take relevant folders
    orig_data_folder_names=[]
    #data_folder_prefix = os.path.join(base_dir, '..', 'my_data', success_strategy)
    data_folder_prefix = os.path.join(results_dir, success_strategy)
    orig_data_folder_names = os.listdir(data_folder_prefix)
    
    # Remove undesired folders
    for data_folder_name in orig_data_folder_names:
        if data_folder_name[:2] == '20':
            data_folder_names.append(data_folder_name)
            

    numTrials=len(data_folder_names)
    # Get max number of iterations for each level/axis for entire experiment sets   
    levels=['primitive', 'composite', 'llbehavior']
    axes=['Fx','Fy','Fz','Mx','My','Mz']
    
    # Dictionary building blocks
    folder_dims={}
    dict_dims={}
    dict_all={}
    
    # Construct dictionary to hold dimensions. Need level and axis sub-spaces.
    for level in levels:
        folder_dims[level] = {}
        for axis in axes:
            folder_dims[level][axis]={}
    
    # Create a dictionary structure for all trials, RCBHT levels, and axis.
    for data_folder_name in data_folder_names:
        data_folder_full_path = os.path.join(data_folder_prefix, data_folder_name)
        print data_folder_full_path
        
        dict_cooked_from_folder = data_folder_parser.parse_folder(data_folder_full_path)
        if dict_cooked_from_folder == None:
            continue
        else:
            dict_all[data_folder_name]=dict_cooked_from_folder

    # Once dict_cooked_from_folder exists, get dimensions of level/axis for each folder
    for data_folder_name in data_folder_names:
        for level in levels:
            for axis in axes: 
                folder_dims[level][axis] = len(dict_all[data_folder_name][level][axis])
        dict_dims[data_folder_name]=deepcopy(folder_dims)
                
    # Only keep the largest dimensions for each level/axis
    for level in levels:
        for axis in axes:                            
            for data_folder_name in data_folder_names:                
                temp=dict_dims[data_folder_name][level][axis] 
                if temp > folder_dims[level][axis]:
                    folder_dims[level][axis]=temp

    # For one trial, take the dictionary and reshape it (change number of iterations) so we have the same NUMBER of labels for all axes. 
    # Then only return the labels.
    # Currently we take the max number of iterations in any given trial/level/axis.
    allTrialLabels={}
    for data_folder_name in data_folder_names:        
        data_feature_extractor.extract_features(dict_all[data_folder_name],folder_dims)
        allTrialLabels[data_folder_name]=deepcopy(dict_all[data_folder_name])
    
    for data_folder_name in data_folder_names:   
        if not os.path.exists(os.path.join(base_dir, '..', 'my_training_data', 'img_of_success')):
            os.makedirs(os.path.join(base_dir, '..', 'my_training_data', 'img_of_success'))
            
       # label 1 indicates vs high_level_states... have a file and a place to put images
        if output_per_one_trial_flag:
            output_features.output_sample_one_trial(file_for_success_set, '1', dict_cooked_from_folder, os.path.join(base_dir, '..', 'my_training_data', 'img_of_success'))
        else:
            output_features.output_sample_all_trial(file_for_success_set, '1', allTrialLabels,data_folder_names, numTrials,os.path.join(base_dir, '..', 'my_training_data',success_strategy))

    #-------------------------------------------------------------------------
    ##FAILURE ANALYSIS
    #-------------------------------------------------------------------------
#    if failureFlag:
#        file_for_fail_set = open(os.path.join(base_dir, '..', 'my_training_data', 'training_set_of_fail'), 'w')
#        
#        data_folder_prefix = os.path.join(base_dir, '..', 'my_data', 'failure_strategy')
#        data_folder_names = os.listdir(data_folder_prefix)
#        
#        for data_folder_name in data_folder_names:
#            data_folder_full_path = os.path.join(data_folder_prefix, data_folder_name)
#            print data_folder_full_path
#            
#        dict_cooked_from_folder = data_folder_parser.parse_folder(data_folder_full_path)        
#        if dict_cooked_from_folder == None:
#            continue
#        
#        data_feature_extractor.extract_features(dict_cooked_from_folder)
#        if not os.path.exists(os.path.join(base_dir, '..', 'my_training_data', 'img_of_fail')):
#             os.makedirs(os.path.join(base_dir, '..', 'my_training_data', 'img_of_fail'))
#             
#        output_features.output_sample(file_for_fail_set, '0', dict_cooked_from_folder, os.path.join(base_dir, '..', 'my_training_data', 'img_of_fail'))


#-------------------------------------------------------------------------
## Parse information by State 
#-------------------------------------------------------------------------
#if "high level states" in tasks:
#    #generate training data from SIM to classify states
#
#    from inc.states import states
#    files_for_states = {}
#
#    for state in states:
#        files_for_states[state] = open(os.path.join(base_dir,'..', 'my_training_data', "training_set_for_"+state), 'w')    
#
#
#    #data_folder_prefix = os.path.join(base_dir, '..', 'my_data', 'data_003_SIM_HIRO_SA_Success-master')
#    data_folder_prefix = os.path.join(base_dir,'..', 'my_data', success_strategy)
#    data_folder_names = os.listdir(data_folder_prefix)
#
#    for data_folder_name in data_folder_names:
#        data_folder_full_path = os.path.join(data_folder_prefix, data_folder_name)
#        print data_folder_full_path
#        
#        dict_cooked_from_folder = data_folder_parser.parse_folder(folder_path=data_folder_full_path, split_by_states=True)
#
#        if dict_cooked_from_folder == None:
#            continue
#
#
#        for state in dict_cooked_from_folder:
#            data_feature_extractor.extract_features(dict_cooked_from_folder[state])
#            if not os.path.exists(os.path.join(base_dir, '..', 'my_training_data', 'img_of_'+state)):
#                os.makedirs(os.path.join(base_dir, '..', 'my_training_data', 'img_of_'+state))
#
#            output_features.output_sample(files_for_states[state], str(states.index(state)), dict_cooked_from_folder[state], os.path.join(base_dir, '..', 'my_training_data', 'img_of_'+state))


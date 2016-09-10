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
    ## Flags
    successAndFailFlag=1
    hlStatesFlag=1
    output_per_one_trial_flag=1 # if true, output is performed for each trial for all axis. Otherwise all trials for each axis is displayed.

    # What kind of success_strategy will you analyze
    success_strategy='SIM_HIRO_ONE_SA_SUCCESS'
    failure_strategy="SIM_HIRO_ONE_SA_ERROR_CHARAC_Prob"
    strategy=success_strategy # default value. used in hblstates

    # lists
    states=['approach','rotation','insertion','mating']
    levels=['primitive', 'composite', 'llbehavior']
    axes=['Fx','Fy','Fz','Mx','My','Mz']

    # Folder names
    data_folder_names=[]        # Filtered to only take relevant folders
    orig_data_folder_names=[]
    failure_data_folder_names=[]

    # Dictionary building blocks
    folder_dims={}
    dict_dims={}
    dict_all={}
    allTrialLabels={}

    # Set program paths
    results_dir="/Users/sklaw_mba/Desktop/ex/dr_juan_proj/data"
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

    # img_directory for successful strategies
    directory='allTrials_success'
    if not os.path.exists(os.path.join(base_dir, '..', 'my_training_data',success_strategy,directory)):
        os.makedirs(os.path.join(base_dir, '..', 'my_training_data',success_strategy,directory))    

    # img directory for failure strategies
    directory='allTrials_failure'
    if not os.path.exists(os.path.join(base_dir, '..', 'my_training_data',failure_strategy,directory)):
        os.makedirs(os.path.join(base_dir, '..', 'my_training_data',failure_strategy,directory))    
        
    # img_directory for higher level state separation
    #directory='allTrials_hlb'
    #if not os.path.exists(os.path.join(base_dir, '..', 'my_training_data',hlb,directory)):
    #    os.makedirs(os.path.join(base_dir, '..', 'my_training_data',hlb,directory))        

    # What kind of data should we collect?
    # - Success
    # - Failure
    # + Generate high level states data

    # 1. Get data for success tasks for a given success_strategy

    if successAndFailFlag:
        strategy=success_strategy    
        file_for_success_set = open(os.path.join(base_dir, '..', 'my_training_data', strategy, 'training_set_of_success'), 'w')
        
        
        # Get Folder names
        #data_folder_prefix = os.path.join(base_dir, '..', 'my_data', success_strategy)
        data_folder_prefix = os.path.join(results_dir, strategy)
        orig_data_folder_names = os.listdir(data_folder_prefix)
        
        # Remove undesired folders
        for data_folder_name in orig_data_folder_names:
            if data_folder_name[:2] == '20':
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
        failure_data_folder_names=[]
    #-------------------------------------------------------------------------
    ##FAILURE ANALYSIS
    #------------------------------------------------------------------------_
        
        strategy=failure_strategy
        file_for_fail_set = open(os.path.join(base_dir, '..', 'my_training_data', strategy, 'training_set_of_fail'), 'w')    
        
        # Read failure data
        data_folder_prefix = os.path.join(results_dir, failure_strategy)
        orig_data_folder_names = os.listdir(data_folder_prefix)

        # Remove undesired folders
        for data_folder_name in orig_data_folder_names:
            if data_folder_name[:2] == 'ex' or data_folder_name[:2] == 'FC':
                failure_data_folder_names.append(data_folder_name)   
                
        
        # Get full path for each folder name
        for data_folder_name in failure_data_folder_names:
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
        failure_data_folder_names=[]

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
        if output_per_one_trial_flag:
            output_features.output_sample_one_trial(file_for_success_set, '1', allTrialLabels)
        else:
            output_features.output_sample_all_trial(file_for_success_set, 's', allTrialLabels, allTrialLabels.keys(), len(allTrialLabels), os.path.join(base_dir, '..', 'my_training_data',success_strategy))

        dict_dims={}
        dict_all={}
        allTrialLabels={}
        data_folder_names=[]        
        orig_data_folder_names=[]
        failure_data_folder_names=[]

        #output data for fail
        dict_all = fail_dict_all
        for data_folder_name in dict_all:        
            data_feature_extractor.extract_features(dict_all[data_folder_name],folder_dims)
            allTrialLabels[data_folder_name]=deepcopy(dict_all[data_folder_name])  
        if output_per_one_trial_flag:
            output_features.output_sample_one_trial(file_for_fail_set, '0', allTrialLabels); 
        else:
            output_features.output_sample_all_trial(file_for_fail_set, 'f', allTrialLabels, allTrialLabels.keys(), len(allTrialLabels), os.path.join(base_dir, '..', 'my_training_data',failure_strategy))

        # Clear up
        folder_dims={}
        dict_dims={}
        dict_all={}
        allTrialLabels={}
        data_folder_names=[]        
        orig_data_folder_names=[]
        failure_data_folder_names=[]



    #-------------------------------------------------------------------------
    ## Parse information by State 
    #-------------------------------------------------------------------------
    if hlStatesFlag:
        strategy=success_strategy    
        #generate training data from SIM to classify automata states

        from inc.states import states
        files_for_states = {}
        
        # Open files for each of the states we will analyze
        for state in states:
            files_for_states[state] = open(os.path.join(base_dir,'..', 'my_training_data', strategy, "training_set_for_"+state), 'w')    

        data_folder_prefix = os.path.join(results_dir, success_strategy)
        orig_data_folder_names = os.listdir(data_folder_prefix)
        
        # Remove undesired folders
        for data_folder_name in orig_data_folder_names:
            if data_folder_name[:2] == '20':
                data_folder_names.append(data_folder_name)  
                
        numTrials=len(data_folder_names)
                
               
        # Create a dictionary structure for all trials, RCBHT levels, and axis.
        for data_folder_name in data_folder_names:
            data_folder_full_path = os.path.join(data_folder_prefix, data_folder_name)
            if DB_PRINT:
                print data_folder_full_path            
            
            dict_cooked_from_folder = data_folder_parser.parse_folder(data_folder_full_path, split_by_states=True)
            if dict_cooked_from_folder == None:
                continue
            else:
                dict_all[data_folder_name]=dict_cooked_from_folder
            
            
        # Generate a dictionary of dimensions for all trials/states/levels/axis
        if bool(dict_all): 
            folder_dims = {}
            for level in levels:
                folder_dims[level] = {}
                for axis in axes:
                    folder_dims[level][axis]=0           
         
            for data_folder_name in dict_all:
                for state in states:
                    for level in levels: 
                        for axis in axes:
                            temp = len(dict_all[data_folder_name][state][level][axis])
                            if temp > folder_dims[level][axis]:
                                folder_dims[level][axis]=temp
                            
                                    
            # For one trial, take the dictionary and reshape it (change number of iterations) so we have the same NUMBER of labels for all axes. 
            # Then only return the labels.
            # Currently we take the max number of iterations in any given trial/level/axis.
            allTrialLabels={}
            for data_folder_name in data_folder_names:
                for state in states:                
                    data_feature_extractor.extract_features(dict_all[data_folder_name][state], folder_dims)                                                                                          
                
            # Create the allTrialsLables structure. It is organized by: state/trials/level/axis. Only contains labels.
            for state in states:
                allTrialLabels[state]={}
                for data_folder_name in data_folder_names:
                    allTrialLabels[state][data_folder_name]={}                        
                    allTrialLabels[state][data_folder_name]=deepcopy(dict_all[data_folder_name][state])                        
                        
            ## Create directories to keep both labels and images in file
            for state in states:
                hlb_dir=strategy + '/' + 'img_of_'+state
                if not os.path.exists(os.path.join(base_dir, '..', 'my_training_data', hlb_dir)):
                    os.makedirs(os.path.join(base_dir, '..', 'my_training_data', hlb_dir))
                
                # Write labels and images to file. 2 choices: individual iterations or all iterations per state.
                # label 1 indicates SUCCESS. Have a file and a place to put images
                if output_per_one_trial_flag:
                    output_features.output_sample_one_trial(files_for_states[state], str(states.index(state)), allTrialLabels[state])
                else:
                    output_features.output_sample_all_trial(files_for_states[state], state, allTrialLabels[state], data_folder_names, numTrials, os.path.join(base_dir, '..', 'my_training_data',hlb_dir))
        else:
                raise Exception('dict_all from hlb states is not available')
        folder_dims={}
        dict_dims={}
        dict_all={}
        allTrialLabels={}
        data_folder_names=[] 
        orig_data_folder_names=[]
        failure_data_folder_names=[]

main();

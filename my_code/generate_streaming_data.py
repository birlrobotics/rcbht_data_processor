
''' Currently this program requires clean data set. Meaning: 3 folders only: Segments, 
Composites, and llBehaviors. Each folder must have only 6 files associated with _Fx, 
_Fy, _Fz, _Mx, _My, and _Mz. There should also be a State.dat file outside the three folders.
If more data is encountered it will not be processed at this time.'''



import os
import shutil

from copy import deepcopy

import util.output_features                     as output_features
import util.output_streaming_experiments        as output_streaming_experiments
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

    # What kind of success_strategy will you analyze
    success_strategy='REAL_HIRO_ONE_SA_SUCCESS'
    failure_strategy="REAL_HIRO_ONE_SA_ERROR_CHARAC"
    strategy=success_strategy # default value. used in hblstates


    import inc.config as config
    # lists
    states = config.states
    levels = config.levels
    axes = config.axes

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
    results_dir="/Users/sklaw_mba/Desktop/ex/dr_juan_proj/workshop/data"
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = cur_dir
    os.chdir(base_dir)

    # my training data
    directory='my_streaming_experiments'
    if not os.path.exists(os.path.join(base_dir, '..', directory)):
        os.makedirs(os.path.join(base_dir, '..', directory))
    else:
        shutil.rmtree(os.path.join(base_dir, '..', directory))
        os.makedirs(os.path.join(base_dir, '..', directory))

    streaming_exp_dir = os.path.join(base_dir, '..', directory)

    if successAndFailFlag:
        strategy=success_strategy    
        
        
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
        
    #-------------------------------------------------------------------------
    ##FAILURE ANALYSIS
    #------------------------------------------------------------------------_
        
        strategy=failure_strategy
        
        # Read failure data
        data_folder_prefix = os.path.join(results_dir, failure_strategy)
        orig_data_folder_names = os.listdir(data_folder_prefix)

        # Remove undesired folders
        for data_folder_name in orig_data_folder_names:
            if data_folder_name[:1] == 'x':
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

        #init folder_dims
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

        print "model feature extraction for S&F will and must use folder_dims:", folder_dims

        for data_folder_name in success_dict_all:
            import experiment_streamer.experiment_streamer as experiment_streamer
            array_of_streaming_dicts = experiment_streamer.stream_one_experiment(success_dict_all[data_folder_name])
            for idx in range(len(array_of_streaming_dicts)):
                data_feature_extractor.extract_features(array_of_streaming_dicts[idx], folder_dims) 
            output_streaming_experiments.output_one_streaming_exp(streaming_exp_dir, "success", data_folder_name, array_of_streaming_dicts)    

        for data_folder_name in fail_dict_all:
            import experiment_streamer.experiment_streamer as experiment_streamer
            array_of_streaming_dicts = experiment_streamer.stream_one_experiment(success_dict_all[data_folder_name])
            for idx in range(len(array_of_streaming_dicts)):
                data_feature_extractor.extract_features(array_of_streaming_dicts[idx], folder_dims) 
            output_streaming_experiments.output_one_streaming_exp(streaming_exp_dir, "failure", data_folder_name, array_of_streaming_dicts)    

    #-------------------------------------------------------------------------
    ## Parse information by State 
    #-------------------------------------------------------------------------
    if hlStatesFlag:
        # Folder names
        data_folder_names=[]        # Filtered to only take relevant folders
        orig_data_folder_names=[]
        failure_data_folder_names=[]

        # Dictionary building blocks
        folder_dims={}
        dict_dims={}
        dict_all={}
        allTrialLabels={}

        strategy=success_strategy    
        #generate training data from SIM to classify automata states

        from inc.states import states
        
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
            pass
        else:
            raise Exception('The states dictionary dict_all is empty')

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
                            
        print "model feature extraction for STATES will and must use folder_dims:", folder_dims

        for data_folder_name in dict_all:
            for state in states:
                import experiment_streamer.experiment_streamer as experiment_streamer
                array_of_streaming_dicts = experiment_streamer.stream_one_experiment(dict_all[data_folder_name][state])
                for idx in range(len(array_of_streaming_dicts)):
                    data_feature_extractor.extract_features(array_of_streaming_dicts[idx], folder_dims) 
                output_streaming_experiments.output_one_streaming_exp(streaming_exp_dir, state, data_folder_name, array_of_streaming_dicts)    
main();

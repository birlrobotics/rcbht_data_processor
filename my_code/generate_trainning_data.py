''' 
TODO: online classification.

Currently this program requires clean data set. Meaning: 3 folders only: Segments, 
Composites, and llBehaviors. Each folder must have only 6 files associated with _Fx, 
_Fy, _Fz, _Mx, _My, and _Mz. There should also be a State.dat file outside the three folders.
If more data is encountered it will not be processed at this time.'''

# debugging
import ipdb, traceback,sys#,code
ipdb.set_trace()

# Performance
#from numba import jit
## Cython
#import pyximport
#pyximport.install()

# system
import os
#import shutil

# General
from copy import deepcopy
import cPickle as pickle

# math
from math import log10,pow,sqrt
import numpy as np 

# time
#import time

# Iterations
import itertools

# parsing
import util.output_features                     as output_features
import data_parser.data_folder_parser           as data_folder_parser
import feature_extractor.data_feature_extractor as data_feature_extractor

# classification
from sklearn import cross_validation

#from datasketch import MinHash

# LCS
import lcs.LCS
import lcs.rcbht_lbl_conversion as lbl

#------------------------------------------------------------------------------

# Globals
global DB_PRINT
DB_PRINT=0

#------------------------------------------------------------------------------

global states
global levels
global axes

#@jit
#def actionGrammar():

# lists
states=['approach','rotation','insertion','mating']
levels=['primitive', 'composite', 'llbehavior']
axes=['Fx','Fy','Fz','Mx','My','Mz']
train_validate=['train','validate']

# Dictionaries
allTrialLabels={}

# lenghts
s_len=len(states)
l_len=len(levels)
a_len=len(axes)
tv_len=len(train_validate)
#------------------------------------------------------------------------------

## Flags
initFlag        =0              # Used for an initialization routine in creating jaccard axis
sliceLabels     =1           # Determines whether we slice labels so that they all axes have equal number of labels
loadFromFileFlag=1      # Determines wehther we load saved structures to the program when they have been computed once.

successFlag     =0           # Find labels for entire task
failureFlag     =0
hlStatesFlag    =0          # Separate labels by state
# For success/failure/hlStates there are 2 types of output: (i) the output per one trial, (ii) the output per all trais
output_per_one_trial_flag=0 
#------------------------------------------------------------------------------
# stream processing
jaccard=0
get_allTrial_Labels=0
#------------------------------------------------------------------------------
lcss=1
#------------------------------------------------------------------------------

# What kind of success_strategy will you analyze
success_strategy='REAL_HIRO_ONE_SA_SUCCESS'
failure_strategy="SIM_HIRO_ONE_SA_ERROR_CHARAC_Prob"
strategy=success_strategy # default value. used in hblstates

#------------------------------------------------------------------------------
# Pickle folders
#------------------------------------------------------------------------------

# jaccard
hlb_pickle                  ='allTrials_hlbStates.pickle'
state_probabilities_pickle  ='state_prob_onelevel.pickle'
results_pickle              ='results.pickle'

#------------------------------------------------------------------------------
# lcss
trial_lcss_mat_pickle         ='trial_lcss_mat.pickle'
sim_state_metric_pickle       ='sim_state_metric.pickle'
lcss_results_pickle           ='lcss_results.pickle'
permutation_matrix_pickle     = 'permutation_matrix.pickle'
classification_matrix_pickle  = 'classification_matrix.pickle'
avg_prob_vector_pickle        = 'avg_prob_vector_pickle'

#------------------------------------------------------------------------------
# Folder names
data_folder_names           =[]        # Filtered to only take relevant folders
orig_data_folder_names      =[]
failure_data_folder_names   =[]

#------------------------------------------------------------------------------

# Set program paths
results_dir ='/home/vmrguser/sc/research/AIST/Results/ForceControl/'
cur_dir     = os.path.dirname(os.path.realpath(__file__))
base_dir    = cur_dir
os.chdir(base_dir)

#------------------------------------------------------------------------------

# my training data
directory='my_training_data'
train_data_dir=os.path.join(base_dir, '..', directory)

if not os.path.exists(train_data_dir):
    os.makedirs(train_data_dir)
    
# If the directory already exists, recursively remove it.     
#else:
#    shutil.rmtree(train_data_dir)
#    os.makedirs(train_data_dir)

# img_directory for successful strategies
directory='allTrials_success'
if not os.path.exists(os.path.join(train_data_dir,success_strategy,directory)):
    os.makedirs(os.path.join(train_data_dir,success_strategy,directory))    

# img directory for failure strategies
directory='allTrials_failure'
if not os.path.exists(os.path.join(train_data_dir,failure_strategy,directory)):
    os.makedirs(os.path.join(train_data_dir,failure_strategy,directory))    
    
# img_directory for higher level state separation
#directory='allTrials_hlb'
#if not os.path.exists(os.path.join(base_dir, '..', 'my_training_data',hlb,directory)):
#    os.makedirs(os.path.join(base_dir, '..', 'my_training_data',hlb,directory))        

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# What kind of data should we collect?
# - Success
# - Failure
# + Generate high level states data

# 1. Get data for success tasks for a given success_strategy
if successFlag:
    
    # Dictionary building blocks
    folder_dims={}
    dict_dims={}
    dict_all={}
    allTrialLabels={}
    
    # Assign the current strategy directory
    strategy             = success_strategy     
    strat_dir            = os.path.join(train_data_dir,strategy,)             
    file_for_success_set = open(os.path.join(strat_dir, 'training_set_of_success'), 'w')
    
    
    # Get Folder names
    #data_folder_prefix = os.path.join(base_dir, '..', 'my_data', success_strategy)
    data_folder_prefix = os.path.join(results_dir, strategy)
    orig_data_folder_names = os.listdir(data_folder_prefix)
    
    # Remove undesired folders
    for data_folder_name in orig_data_folder_names:
        if data_folder_name[:2] == '20':
            data_folder_names.append(data_folder_name)
            
    numTrials=len(data_folder_names)
    
    # Get max number of iterations for each level/axis for entire experiment sets      
    # Construct dictionary to hold dimensions. Need level and axis sub-spaces.
    for level in levels:
        folder_dims[level] = {}
        for axis in axes:
            folder_dims[level][axis]={}
    
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
        # Once dict_cooked_from_folder exists, get dimensions of level/axis for each folder
        for data_folder_name in data_folder_names:
            for level in levels:
                for axis in axes: 
                    folder_dims[level][axis] = len(dict_all[data_folder_name][level][axis])
            dict_dims[data_folder_name]=deepcopy(folder_dims)
                    
        if bool(dict_dims):
            # Only keep the largest dimensions for each level/axis
            for level in levels:
                for axis in axes:                            
                    for data_folder_name in data_folder_names:                
                        temp=dict_dims[data_folder_name][level][axis] 
                        if temp > folder_dims[level][axis]:
                            folder_dims[level][axis]=temp
        
            # For each trial, take the dictionary and reshape it (change number of iterations), output is the same NUMBER of labels for all axes.             
            try:
                for data_folder_name in data_folder_names:        
                    data_feature_extractor.extract_features(dict_all[data_folder_name],folder_dims,sliceLabels)
                    
                    # Create the allTrialsLables structure. It is organized by: state/trials/level/axis. Only contains labels.
                    allTrialLabels[data_folder_name]=deepcopy(dict_all[data_folder_name])  
            except:
                print 'error found in extract_features'
                type, value, tb = sys.exc_info()
                traceback.print_exc()
                ipdb.post_mortem(tb)
                
            # Save allTrialLabels to File using pickle
            with open(data_folder_prefix+'/allTrials_success.pickle', 'wb') as handle:
                pickle.dump(allTrialLabels, handle, protocol=pickle.HIGHEST_PROTOCOL)                          
            
            if not os.path.exists(os.path.join(base_dir, '..', 'my_training_data', 'img_of_success')):
                os.makedirs(os.path.join(base_dir, '..', 'my_training_data', 'img_of_success'))
                    
            try:
                # label s indicates SUCCESS. Have a file and a place to put images
                if output_per_one_trial_flag:
                    output_features.output_sample_one_trial(file_for_success_set, 's', dict_cooked_from_folder, os.path.join(base_dir, '..', 'my_training_data', 'img_of_success'))
                else:
                    if sliceLabels: # TODO Currently only execute if we are slciing. Need to modify image. 
                        output_features.output_sample_all_trial(file_for_success_set, 's', allTrialLabels,data_folder_names,numTrials,strat_dir)
            except:
                type, value, tb = sys.exc_info()
                traceback.print_exc()
                ipdb.post_mortem(tb)
        else:
            print 'The success dictionary dict_dims is empty'
    else:
        print 'The success dictionary dict_all is empty' 
    
#-------------------------------------------------------------------------
##FAILURE ANALYSIS
#------------------------------------------------------------------------_
    
if failureFlag:   

    # Initialize structures
    folder_dims={}
    dict_dims={}
    dict_all={}
    allTrialLabels={}    
    
    strategy=failure_strategy
    file_for_fail_set = open(os.path.join(base_dir, '..', 'my_training_data', strategy, 'training_set_of_fail'), 'w')    
    
    # Read failure data
    data_folder_prefix = os.path.join(results_dir, failure_strategy)
    orig_data_folder_names = os.listdir(data_folder_prefix)

    # Remove undesired folders
    for data_folder_name in orig_data_folder_names:
        if data_folder_name[:2] == 'ex' or data_folder_name[:2] == 'FC':
            failure_data_folder_names.append(data_folder_name)   
            
    numTrials=len(failure_data_folder_names)
            
    # Get max number of iterations for each level/axis for entire experiment sets      
    # Construct dictionary to hold dimensions. Need level and axis sub-spaces.
    for level in levels:
        folder_dims[level] = {}
        for axis in axes:
            folder_dims[level][axis]={}
    
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
        for data_folder_name in failure_data_folder_names:
            for level in levels:
                for axis in axes: 
                    folder_dims[level][axis] = len(dict_all[data_folder_name][level][axis])
            dict_dims[data_folder_name]=deepcopy(folder_dims)
                
        if bool(dict_dims):
            # Only keep the largest dimensions for each level/axis
            for level in levels:
                for axis in axes:                            
                    for data_folder_name in failure_data_folder_names:                
                        temp=dict_dims[data_folder_name][level][axis] 
                        if temp > folder_dims[level][axis]:
                            folder_dims[level][axis]=temp
                
            # For one trial, take the dictionary and reshape it (change number of iterations) so we have the same NUMBER of labels for all axes. 
            # Then only return the labels.
            # Currently we take the max number of iterations in any given trial/level/axis.
            try:
                for data_folder_name in failure_data_folder_names:        
                    data_feature_extractor.extract_features(dict_all[data_folder_name],folder_dims,sliceLabels)
                    allTrialLabels[data_folder_name]=deepcopy(dict_all[data_folder_name]) 
            except:
                type, value, tb = sys.exc_info()
                traceback.print_exc()
                ipdb.post_mortem(tb)
                
            # Save allTrialLabels to File using pickle
            with open(data_folder_prefix+'/allTrials_failure.pickle', 'wb') as handle:
                pickle.dump(allTrialLabels, handle, protocol=pickle.HIGHEST_PROTOCOL)                               
                
            if not os.path.exists(os.path.join(base_dir, '..', 'my_training_data', 'img_of_fail')):
                 os.makedirs(os.path.join(base_dir, '..', 'my_training_data', 'img_of_fail'))                 
            
            # Write labels and images to file. 2 choices: individual iterations or all iterations per state.            
            try:
                # label f indicates SUCCESS. Have a file and a place to put images
                if output_per_one_trial_flag:
                    output_features.output_sample_one_trial(file_for_fail_set, 'f', dict_cooked_from_folder, os.path.join(base_dir, '..', 'my_training_data', 'img_of_failure'))            
                else:
                    if sliceLabels: # TODO Currently only execute if we are slciing. Need to modify image. 
                        output_features.output_sample_all_trial(file_for_fail_set, 'f', allTrialLabels,failure_data_folder_names, numTrials,os.path.join(base_dir, '..', 'my_training_data',failure_strategy))
            except:
                type, value, tb = sys.exc_info()
                traceback.print_exc()
                ipdb.post_mortem(tb)
        else:
            print 'The failure dictionary dict_dims is empty'
    else:
        print 'The failure dictionary dict_all is empty'

    # Clear up
    folder_dims={}
    dict_dims={}
    dict_all={}
    allTrialLabels={}
#-------------------------------------------------------------------------
## Parse information by State 
#-------------------------------------------------------------------------
if hlStatesFlag:
    
    # Initialize structures
    folder_dims={}
    dict_dims={}
    dict_all={}
    allTrialLabels={}  
    
    #generate training data from SIM to classify automata states

    from inc.states import states
    files_for_states = {}
    
    # Folders for storing results
    if strategy=='':                        # if this path does not exist
        strategy=raw_input('Please enter a name for the strategy: \n')

    # Assign the current strategy directory    
    strat_dir = os.path.join(train_data_dir,strategy,)     
    
    # Open files for each of the states we will analyze
    for state in states:
        files_for_states[state] = open(os.path.join(strat_dir, "training_set_for_"+state), 'w')    

    data_folder_prefix = os.path.join(results_dir, success_strategy)
    orig_data_folder_names = os.listdir(data_folder_prefix)
    
    # Remove undesired folders
    for data_folder_name in orig_data_folder_names:
        if data_folder_name[:2] == '20':
            data_folder_names.append(data_folder_name)  
            
    numTrials=len(data_folder_names)
            
    # Get max number of iterations for each level/axis for entire experiment sets      
    # Construct dictionary to hold dimensions. Need level and axis sub-spaces.
    for state in states:
        folder_dims[state] = {}
        for level in levels:
            folder_dims[state][level] = {}
            for axis in axes:
                folder_dims[state][level][axis]={}            
            
    # Create a dictionary structure for all trials, RCBHT levels, and axis.
    for data_folder_name in data_folder_names:
        data_folder_full_path = os.path.join(data_folder_prefix, data_folder_name)
        if DB_PRINT:
            print data_folder_full_path            
        
        try:
            dict_cooked_from_folder = data_folder_parser.parse_folder(data_folder_full_path, split_by_states=True)
            if dict_cooked_from_folder == None:
                continue
            else:
                dict_all[data_folder_name]=dict_cooked_from_folder
        except:
            print 'error found in '+ data_folder_name
            type, value, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)                
        
    # Generate a dictionary of dimensions for all trials/states/levels/axis
    if bool(dict_all): 
        for data_folder_name in data_folder_names:
            for state in states:
                for level in levels: 
                    for axis in axes:
                        try:
                            folder_dims[state][level][axis] = len(dict_all[data_folder_name][state][level][axis])
                        except:
                            print 'error found in '+ len
                            type, value, tb = sys.exc_info()
                            traceback.print_exc()
                            ipdb.post_mortem(tb)
            dict_dims[data_folder_name]=deepcopy(folder_dims)
        # Only keep the largest dimensions for each state/level/axis
        if bool(dict_dims):     
            for data_folder_name in data_folder_names:
                for state in states:
                    for level in levels:
                        for axis in axes:                                                                        
                            temp=dict_dims[data_folder_name][state][level][axis] 
                            if temp > folder_dims[state][level][axis]:
                                folder_dims[state][level][axis]=temp
                                
            # For one trial, take the dictionary and reshape it (change number of iterations) so we have the same NUMBER of labels for all axes. 
            # Then only return the labels.
            # Currently we take the max number of iterations in any given trial/level/axis.
            allTrialLabels={}
            try:
                for data_folder_name in data_folder_names:
                    for state in dict_cooked_from_folder:                
#                       data_feature_extractor.extract_features(dict_cooked_from_folder[state],folder_dims[state])
                        data_feature_extractor.extract_features(dict_all[data_folder_name][state],folder_dims[state],sliceLabels)                                                                                          
                                    
            except:
                print 'error found in '+ state
                type, value, tb = sys.exc_info()
                traceback.print_exc()
                ipdb.post_mortem(tb)  
                
            # Create the allTrialsLables structure. It is organized by: state/trials/level/axis. Only contains labels.
            for state in dict_cooked_from_folder:
                allTrialLabels[state]={}
                for data_folder_name in data_folder_names:
                    allTrialLabels[state][data_folder_name]={}                        
                    allTrialLabels[state][data_folder_name]=deepcopy(dict_all[data_folder_name][state])

            # Save allTrialLabels to File using pickle
            with open(os.path.join(strat_dir,hlb_pickle), 'wb') as handle:
                pickle.dump(allTrialLabels, handle, protocol=pickle.HIGHEST_PROTOCOL)                               
                        
            ## Create directories to keep both labels and images in file
            for state in dict_cooked_from_folder:
                hlb_dir='img_of_'+state
                if not os.path.exists( os.path.join(strat_dir, hlb_dir) ):
                    os.makedirs(os.path.join(strat_dir, hlb_dir))
                
                # Write labels and images to file. 2 choices: individual iterations or all iterations per state.
                try:
                    # label 1 indicates SUCCESS. Have a file and a place to put images
                    if output_per_one_trial_flag:
                        output_features.output_sample_one_trial(files_for_states[state], str(states.index(state)), dict_cooked_from_folder[state], os.path.join(strat_dir, hlb_dir))
                    else:
                        if sliceLabels: # TODO Currently only execute if we are slciing. Need to modify image. 
                            output_features.output_sample_all_trial(files_for_states[state], str(states.index(state)), allTrialLabels[state],data_folder_names,numTrials,os.path.join(strat_dir,hlb_dir))
            
                except:
                    type, value, tb = sys.exc_info()
                    traceback.print_exc()
                    ipdb.post_mortem(tb)
        else:
            print 'The hlb dictionary dict_dims is empty'
    else:
            print 'dict_all from hlb states is not available'
            
            
#------------------------------------------------------------------------------

''' Take each of the state labels as well as the whole task label if they exist 
and use them for classification. Cross-fold validation will be used for testing. 
Cross validation will be implemented through sklearn's cross validation module 
KFold http://scikit-learn.org/stable/modules/cross_validation.html.
Currently doing offline classification

Jaccard Classification Structures:
-------------------------------------------------------------------------------
allTrialsLabels(dict):
-------------------------------------------------------------------------------
    states(4)
        trials(n)
            level(3)
                axis(6)
                    inside contains a series of labels..
-------------------------------------------------------------------------------

Classification
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
jaccard_final(numpy array):
-------------------------------------------------------------------------------
    kfolds x 
        3 levels x 
            4 states by 6 axis
array([[[[ 0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.]],

        [[ 0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.]],

        [[ 0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.]]],...)
-------------------------------------------------------------------------------
permutation_matrix(numpy array):
-------------------------------------------------------------------------------
    kfolds x 
        3 levels x 
            4 states x 
                4 states by 6 axis (classfication permutations)
        -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        It represents a way to represent the permutations (columns) of classification for each existing state (rows)
        # ... App_train \cap App_validate | App_train \cap Rot_validate ... | App_train \cap Mat_validate ... (4x(4x6)
        # ... Rot_train \cap App_validate | Rot_train \cap Rot_validate ... | Rot_train \cap Mat_validate 
        # ... 
        # ... Mat_train \cap App_validate | Mat_train \cap Rot_validate ... | Mat_train \cap Mat_validate
        -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
primitives:
   APP  [[ 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. ],
   ROT   [ 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. ],
   INS   [ 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. ],
   AMT   [ 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. ]],
composites:
        [[ 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. ],
         [ 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. ],
  ...    [ 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. ],
         [ 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. ]],
behaviors:         
        [[ 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. ],
         [ 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. ],
  ...    [ 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. ],
         [ 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. | 0.,  0.,  0.,  0.,  0.,  0. ]],

-------------------------------------------------------------------------------
classification_vector(numpy array):
-------------------------------------------------------------------------------
        3 levels x 
            4 states x 
                4 states 
        # This six probability values corresponding to each axis above are summed to have one probability per permutation.
        # This structure is summed for each level over all folds to compute an average. 
primtiives:                
  APP     [[ 0.,  0.,  0.,  0.]
  ROT      [ 0.,  0.,  0.,  0.]
  INS      [ 0.,  0.,  0.,  0.]
  MAT      [ 0.,  0.,  0.,  0.]] ... 
-------------------------------------------------------------------------------    
'''


#--------------------------------------------------------------------------            
if jaccard:        
#--------------------------------------------------------------------------

    # Folders for storing results
    if strategy=='':                        # if this path does not exist
        strategy=raw_input('Please enter a name for the strategy: \n')
        get_allTrial_Labels=1

        # Assign the current strategy directory    
        strat_dir = os.path.join(train_data_dir,strategy,)      
    
    # Create a classification folder and get the results folder
    directory='jaccard'
    classification_dir=os.path.join(strat_dir,directory)
    if not os.path.exists(classification_dir):
        os.makedirs(classification_dir) 
            
    # If you want to deserialize saved data
    if loadFromFileFlag:        
        with open(os.path.join(strat_dir,hlb_pickle), 'rb') as handle:
           allTrialLabels = pickle.load(handle)            

    # Get Folder names
    data_folder_prefix = os.path.join(results_dir, strategy)    
    
    # Initialize structures (from previous runs)
    folder_dims={}
    dict_dims={}
    dict_all={}      
    
    # Store current data folder names in a list so we can iterate it through it
    if not bool(data_folder_names):
        # Get Folder names
        #data_folder_prefix = os.path.join(base_dir, '..', 'my_data', success_strategy)
        data_folder_prefix = os.path.join(results_dir, strategy)
        orig_data_folder_names = os.listdir(data_folder_prefix)
        
        # Remove undesired folders
        for data_folder_name in orig_data_folder_names:
            if data_folder_name[:2] == '20':
                data_folder_names.append(data_folder_name)
                
        numTrials=len(data_folder_names)  
    #--------------------------------------------------------------------------         
    # kFold Setup
    #--------------------------------------------------------------------------
    # Initialize k-fold data to a valid integer
    kfold=numTrials
    
    # Crossfold training and testing generator
    kf=cross_validation.LeaveOneOut(numTrials)
                    
    # Generate structure to hold train/validate indeces according to the number of folds
    # Given that we are doing leave one out, we use numTrials as the number of folds   
    kf_list=[ [] for k in range(numTrials)]; foo=[]; temp=[]
    
    # Extract the indeces that belong to the training and validation lists    
    for t,v in kf:
        temp.append(list(t)), temp.append(list(v))              
    
    # Extract result onto a list
    for k in range(numTrials):
        for t in range(2):            
            for elems in range(len(temp[2*k+t])): # Need to extract the ints from the interior list
                foo.append(temp[2*k+t][elems])             
            kf_list[k].append(foo)
            foo=[]    
    
    train_len=len(kf_list[0][0])
    validate_len=len(kf_list[0][1])
    tvlen=[];tvlen.append(train_len);tvlen.append(validate_len)
    
    #--------------------------------------------------------------------------
       
    # Structures 
    # kfold_list:               contains indeces for train/validate trials for each fold: kfold x train|validate x elems   
    # jacard axis t/v list:     contains 1x6 final labels for train union set and test set. It is a 1x6 set per fold/train/state/level
    # permutation_matrix:       contains results of int/union permutations. level x states x (statesxaxes)       
    # state_probability array:  computes p(AB) like operations. squishes permutation. final state_probability under fold/level: state/state (4x4)    
    # avg_prob_vector:          computes normalized probabilities over folds. level x states x states
    # classification_vector:    computes percentage of correct classification for states. level x states x states
    # res_vector:               for each level, keeps classification results of diag(classification_vector) in a row. 3x4 for levels x states    
    jaccard_axis_train=[ [ [ [ [ [] for a in range(len(axes))]                        
                                        for l in range(l_len)]                
                                            for s in range(s_len)]
                                                for t in range(train_len)]
                                                    for k in range(kfold)]  
                                                
    jaccard_axis_validate=[ [ [ [ [ [] for a in range(len(axes))]                        
                                        for l in range(l_len)]                
                                            for s in range(s_len)]
                                                for t in range(validate_len)]                                                           
                                                    for k in range(kfold)]          
                                                    
    # Create state_probability: compute the distance between train/validate sets across permutations of states:
    # Within a given fold/level we want to compute the intersection of training with validation across states. 
    # ... App_train \cap App_validate | App_train \cap Rot_validate ... | App_train \cap Mat_validate ... (4x(4x6)
    # ... Rot_train \cap App_validate | Rot_train \cap Rot_validate ... | Rot_train \cap Mat_validate 
    # ... 
    # ... Mat_train \cap App_validate | Mat_train \cap Rot_validate ... | Mat_train \cap Mat_validate
    permutation_matrix   =np.zeros( ( kfold,l_len,s_len,( s_len*a_len ),1) ) 
    state_probability    =np.ones( ( kfold,l_len,s_len,s_len,1) )
                 
    avg_prob_vector      =np.zeros( ( l_len,s_len,s_len,1) )                     
    classification_vector=np.zeros( ( l_len,s_len,s_len,1) )                                  
                        
    # Populate jaccard_axis 
    # Separate training samples from one test trial (across folds).
    for k in range(kfold):
        for s,state in enumerate(states):
            for l,level in enumerate(levels):
                for t,tv in enumerate(train_validate):                                  
                    for idx,trial in enumerate(kf_list[k][t]): # index of train/test instances                        
                        if tv=='train':
                            if initFlag==0:    
                                for a,axis in enumerate(axes):
                                    jaccard_axis_train[k][idx][s][l][a]=set( allTrialLabels[state][ data_folder_names[ kf_list[k][t][idx] ] ][level][axis] ) 
                                
                                initTrialidx =trial
                                initTrialName=data_folder_names[trial]
                                initFlag     =1
                            
                            # Loop from 2nd element to last and perform the union of the training set. 
                            else:
                                for a,axis in enumerate(axes):                                    
                                    jaccard_axis_train[k][idx][s][l][a]=set( allTrialLabels[state][ data_folder_names[trial] ][level][axis] ) 
                        
                        else: # tv=='validation'
                            for a,axis in enumerate(axes):
                                jaccard_axis_validate[k][idx][s][l][a]=set( allTrialLabels[state][ data_folder_names[ kf_list[k][t][idx] ] ][level][axis] )
                                
                initFlag=0  # for new level         
                
    # Place the union of all training sets together in the first training of jaccard_axis_train                
    for k in range(kfold):
        for s in range(s_len):
            for l in range(l_len): 
                for idx in range(1,len(kf_list[k][0])): #for idx,trial in enumerate(kf_list[k][0],1): # train index starting at 1
                    for a in range(len(axes)):
                        jaccard_axis_train[k][0][s][l][a].union(jaccard_axis_train[k][idx][s][l][a]) 
                        
    # Perform the intersection of the training sets (we have taken their union) with the only validation set
    # Within a given fold/state/level we want to compute the intersection of all axes of training with validation across states. 
    # ... App_train \cap App_validate | App_train \cap Rot_validate ... | App_train \cap Mat_validate
    # ... Mat_train \cap App_validate | Mat_train \cap Rot_validate ... | Mat_train \cap Mat_validate
    # Iterate through 2D axis (6x6) one more time                        
    for k in range(kfold):
        for s in range(s_len):
            for l in range(l_len): 
                for p,a in itertools.product( range(s_len),range(a_len) ): # permutations of states with axes 4x6=24 label sets
                    permutation_matrix[k][l][s][p*a_len+a] = ( 
                                                                                                  #fixed trial index                                   state permutation 
                                                                float(  len( jaccard_axis_train[k][0][s][l][a].intersection(jaccard_axis_validate[k][0][p][l][a]) )) / 
                                                                float(  len( jaccard_axis_train[k][0][s][l][a].union       (jaccard_axis_validate[k][0][p][l][a]) )) 
                                                            )

            
    # State Probabilities
    # Compute product of probabilities for each fold/state/level (1x24)->(1x4)
    # Then normalize the probability by dividing by the sum
    for k in range(kfold):
        for l in range(l_len):
            for s in range(s_len): 
                for p,a in itertools.product( range(s_len),range(a_len) ):
                    state_probability[k][l][s][p] *= permutation_matrix[k][l][s][p*a_len+a]                   
                denominator=np.sum(state_probability[k][l][s])
                if denominator==0:
                    denominator=1.0 # num=0,avoids a division by zero to produce a nan
                for p in range(s_len):
                    state_probability[k][l][s][p] /=denominator
                    print state_probability[k][l][s][p]
                    
    # Save state probabilities
    #pickle
    with open(classification_dir+'/'+state_probabilities_pickle, 'wb') as handle:
        pickle.dump(state_probability, handle, protocol=pickle.HIGHEST_PROTOCOL)            
    # Save binary numpy to file (maybe more efficient than pickle and can be opened with np.loadtxt)
    fd = open(os.path.join(classification_dir,state_probabilities_pickle[:-7]+'.txt'), "wb")
    np.save(fd,state_probability)  
    fd.close()
    
#    Testing a more friendly printout. need to fix formating.
#    with file(os.path.join(classification_dir,state_probabilities_pickle[:-7]+'.txt'), 'w') as outfile:
#     outfile.write('# Array shape: {0}\n'.format(state_probability.shape))
#     for data_slice in state_probability:
#        np.savetxt(outfile, data_slice, fmt='%.5f %.5f %.5f %.5f')
#        outfile.write('# New slice\n')           
                    
    # Find average probabilities
    for l in range(l_len):        
        for s in range(s_len):               
            for k in range(kfold): 
                avg_prob_vector[l][s]+=state_probability[k][l][s] # which permutation was most likely for each state                
            # Divide matrix by number of folds
            avg_prob_vector[l]/=float(kfold)      
     
    # Find average correct classifications.          
    #   Find max value across state and use that to generate a binary matrix. It represents which states correctly performed the right classification.
    #   Sum accross folds, divide by the number of folds.
    #   Each diagonal entry tells you the accuracy per state.
    # The sum of the trace/4 tells you overall task accuracy
    for l in range(l_len):
        for k in range(kfold): 
            for s in range(s_len):               
                m=np.argmax(state_probability[k][l][s]) # which permutation was most likely for each state
                classification_vector[l][s][m]+=1
        # Once we have iterated by all the folds we can do an element-wise division by fold number        
        classification_vector[l]/=float(kfold)
            
    # Results presentation (print,pickle,file)
    res=np.zeros( (l_len,s_len) )
    for l in range(l_len):
        for ctr in range(s_len):
            res[l]=classification_vector[l].diagonal()
            print 'For level: [',l+1,'/',l_len,'] and State [',ctr+1,'/',s_len,'] accuracy is: ',res[l][ctr],'\n'
        print 'Task accuracy for level: ',l+1, ' is: ',np.sum(res[l])/float(s_len),'\n'

    # Save results
    #pickle
    with open(classification_dir+'/'+results_pickle, 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)            
    # Save numpy to file
    #np.savetxt( os.path.join(classification_dir,results_pickle[:-7]+'.txt'), res)  
    with file(os.path.join(classification_dir,results_pickle[:-7]+'.txt'), 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(res.shape))
        for data_slice in res:
            np.savetxt(outfile, data_slice, fmt='%0.3f')
            outfile.write('# New slice\n')
            
#------------------------------------------------------------------------------

''' Take each of the state labels as well as the whole task label if they exist 
and use them for classification. Cross-fold validation will be used for testing. 
Cross validation will be implemented through sklearn's cross validation module 
KFold http://scikit-learn.org/stable/modules/cross_validation.html.
Currently doing offline classification'''

#--------------------------------------------------------------------------            
if lcss:        
#--------------------------------------------------------------------------

    # Folders for storing results
    if strategy=='':                        # if this path does not exist
        strategy=raw_input('Please enter a name for the strategy: \n')
        get_allTrial_Labels=1

    # Assign the current strategy directory    
    strat_dir = os.path.join(train_data_dir,strategy)      
    
    # Create a classification folder and get the results folder
    directory='lcss'
    classification_dir=os.path.join(strat_dir,directory)
    if not os.path.exists(classification_dir):
        os.makedirs(classification_dir) 
            
    # If you want to deserialize saved data
    if not bool(allTrialLabels): 
        if loadFromFileFlag:
            with open(os.path.join(strat_dir,hlb_pickle), 'rb') as handle:
               allTrialLabels = pickle.load(handle)            

    # Get Folder names in the results directory
    data_folder_prefix = os.path.join(results_dir, strategy)  

    # Store current data folder names in a list so we can iterate it through it
    if not bool(data_folder_names):
        # Get Folder names
        #data_folder_prefix = os.path.join(base_dir, '..', 'my_data', success_strategy)
        data_folder_prefix = os.path.join(results_dir, strategy)
        orig_data_folder_names = os.listdir(data_folder_prefix)
        
        # Remove undesired folders
        for data_folder_name in orig_data_folder_names:
            if data_folder_name[:2] == '20':
                data_folder_names.append(data_folder_name)
                
        numTrials=len(data_folder_names)          
    #--------------------------------------------------------------------------         
    # kFold Setup
    #--------------------------------------------------------------------------
    # Initialize k-fold data to a valid integer
    kfold=numTrials
    
    # Crossfold training and testing generator
    kf=cross_validation.LeaveOneOut(numTrials)
                    
    # Generate structure to hold train/validate indeces according to the number of folds
    # Given that we are doing leave one out, we use numTrials as the number of folds   
    kf_list=[ [] for k in range(numTrials)]; foo=[]; temp=[]
    
    # Extract the indeces that belong to the training and validation lists    
    for t,v in kf:
        temp.append(list(t)), temp.append(list(v))              
    
    # Extract result onto a list
    for k in range(numTrials):
        for t in range(2):            
            for elems in range(len(temp[2*k+t])): # Need to extract the ints from the interior list
                foo.append(temp[2*k+t][elems])             
            kf_list[k].append(foo)
            foo=[]    
    
    train_len=len(kf_list[0][0])
    validate_len=len(kf_list[0][1])
    tvlen=[];tvlen.append(train_len);tvlen.append(validate_len)
    
    # Initialize structures (from previous runs)
    folder_dims={}
    dict_dims={}
    dict_all={}     
    
    #--------------------------------------------------------------------------
    # Store current data folder names in a list so we can iterate it through it
    folder_names_list=list(data_folder_names)
       
    # Structures 
    # kfold_list:               contains indeces for train/validate trials for each fold: kfold x train|validate x elems   
    # lcss_trials_mat:          for n trials a nxn matrix that will capture the lcss for a given fold/level/state/axis. Useful for offline data compilation.
    # dictionary:               stores only the existing lcss outputs from the matrix to create an mx2 list. 
    #                           the 2nd column will indicate the frequency with which ther term appeared, for fold/level/state/axis
    # similarity:               takes similarity values for all axes for a given state/level/fold. 
    # results:                  keep results for 3 levels and 4 states 3x4.
    lcss_trials_mat=[ [ [ [ [ [ [] for ff in range(2)]                      # [lcss entry | frequency]
                                    for tt in range(1)]
                                        for aa in range(a_len)]                        
                                            for ss in range(s_len)]
                                                for ll in range(l_len)]
                                                    for kk in range(kfold)]
                                                
    permutation_matrix   =np.zeros( ( kfold,l_len,s_len,( s_len*a_len ),1) ) 
    sim_state_metric     =np.zeros( ( kfold,l_len,s_len,s_len,1) )    

    similarity=[ [ [ [] for ss in range(s_len)]                
                          for ll in range(l_len)]
                              for kk in range(kfold)]   
                              
    classification_vector=np.zeros( ( l_len, s_len, s_len,1) )  
    avg_prob_vector      =np.zeros( ( l_len, s_len, s_len,1) )                                        
#------------------------------------------------------------------------------
# Compute LCSS for training samples for each fold/level/state/axis
# Separate training samples from one test trial (across folds).
# Compare the first appropriate training element, with the rest. Keep the longest lcss.
# Take this oportunity to encode rcbht labels into an a-z alphabet for simpler string comparison later on.     
#    if loadFromFileFlag:
#        if os.path.isfile(os.path.join(strat_dir,trial_lcss_mat_pickle)):                                               
#            with open(os.path.join(strat_dir,trial_lcss_mat_pickle),'rb') as handle:
#                lcss_trials_mat=pickle.load(handle)  
    
    # IF there is no data (not loaded from file) then create lcss_trials_mat                             
    if not bool(lcss_trials_mat[0][0][0][0][0][0]):
        for k in range(kfold):
            for l,level in enumerate(levels):
                for s,state in enumerate(states):
                    for a,axis in enumerate(axes):
                        for i in xrange(train_len):   
                            if i==0:                # on the first round of permutations only calculate the encoding once. after that, strSeq1 will be the same.
                                # Perform similarity calculations across trials                #train index
                                strSeq1 = allTrialLabels[state][ data_folder_names[ kf_list[k][0][0] ]][level][axis] 
                                lbl.encodeRCBHTList(strSeq1,level)    
    
                            else: # don't evaluate the diagonal (same trial)
                                strSeq2 = allTrialLabels[state][ data_folder_names[ kf_list[k][0][i] ]][level][axis]
                                lbl.encodeRCBHTList(strSeq2,level)
                                try:     
                                    if DB_PRINT:
                                        print 'fold: ' + str(k) + ' level: ' + str(l) + ' state: ' + str(s) + ' axes: ' + str(a) + ' index i: ' + str(i)
                                    temp=lcs.LCS.longestCommonSubsequence(strSeq1,strSeq2)  # may sometimes have no overlap ""
                                except:
                                    print 'error found in lcss computation'
                                    type, value, tb = sys.exc_info()
                                    traceback.print_exc()
                                    ipdb.post_mortem(tb)  
                                # ONLY record an lcss if its longer than existing instances in current or previous trials                                
                                # For the first trial run:
                                
                                # 2. Find the max value. If already exists, increase counter.
                                #currMax=max(lcss_trials_mat[k][l][s][a][i][0] or ['e'],key=len)
                                currMax=len(lcss_trials_mat[k][l][s][a][0][0])
                                if currMax==0: #'e': # if list is empty assign directly
                                    lcss_trials_mat[k][l][s][a][0][0]=deepcopy(temp) # Place results in the first row
                                    if lcss_trials_mat[k][l][s][a][0][1]==[]:
                                        lcss_trials_mat[k][l][s][a][0][1]=1 # count the number of times this value appears
                                    else:
                                        lcss_trials_mat[k][l][s][a][0][1]+=1 # count the number of times this value appears
                                    currMax=len(lcss_trials_mat[k][l][s][a][0][0])
                                else:                                                                                                            
                                    if len(temp) > currMax:
                                        # 2. Find the max value. If already exists, increase counter.
                                        lcss_trials_mat[k][l][s][a][0][0]=deepcopy(temp)
                                        if lcss_trials_mat[k][l][s][a][0][1]==[]:
                                            lcss_trials_mat[k][l][s][a][0][1]=1 # count the number of times this value appears
                                        else:
                                            lcss_trials_mat[k][l][s][a][0][1]+=1  
                                        currMax=len(lcss_trials_mat[k][l][s][a][0][0])
        
    
        # Very time consuming structure to produce. Save it here.     
        with open(os.path.join(strat_dir,'oneloop_',trial_lcss_mat_pickle),'wb') as handle:        
            pickle.dump(lcss_trials_mat,handle,protocol=pickle.HIGHEST_PROTOCOL)        
        
    # Dictionary     
    # Go through the structure once again. This time we will squash repeated lcss across the different trial combinations and end up
    # with a dictionary for each fold/level/state/axis.
#    tempIdx=[]
#    for k in range(kfold):
#        for l,level in enumerate(levels):
#            for s,state in enumerate(states):
#                for a,axis in enumerate(axes):
#                    tempLen=len(lcss_trials_mat[k][l][s][a])
#                    i=0
#                    while i<tempLen:
#                        for j in range(tempLen):   
#                            if i!=j:
#                                # IF the string is the same, collect the index
#                                if lcss_trials_mat[k][l][s][a][i][0]==lcss_trials_mat[k][l][s][a][j][0]:
#                                    tempIdx.append(j)                                                       # keep the index
#                                    lcss_trials_mat[k][l][s][a][i][1]+=lcss_trials_mat[k][l][s][a][j][1]    # add ctr 
#                                
#                        # Remove all repeated elements starting from the back. This forms the dictionary.
#                        for j in list(reversed(range(len(tempIdx)))):
#                            del lcss_trials_mat[k][l][s][a][ tempIdx[j] ]
#                        tempLen=len(lcss_trials_mat[k][l][s][a])
#                        tempIdx=[]
#                        i+=1
#                        
#                    # One more round to catch a few empty lists left by mistake
#                    mm=0
#                    tempLen=len(lcss_trials_mat[k][l][s][a])                    
#                    while mm < tempLen:                   
#                        if lcss_trials_mat[k][l][s][a][mm][0]==[]: # Do not increment counter if we delete an entry, since list shrinks
#                            del lcss_trials_mat[k][l][s][a][mm]
#                            tempLen=len(lcss_trials_mat[k][l][s][a])
#                        else:
#                            mm+=1                                                     
                    
    #--------------------------------------------------------------------------                
    # Similarity Permutation Matrix.
    # For each fold/level there are m states each with m tests (each with n axis)
    # ... App_train \cap App_validate | App_train \cap Rot_validate ... | App_train \cap Mat_validate ... (mx(mxn)
    # ... Rot_train \cap App_validate | Rot_train \cap Rot_validate ... | Rot_train \cap Mat_validate 
    # ... 
    # ... Mat_train \cap App_validate | Mat_train \cap Rot_validate ... | Mat_train \cap Mat_validate
    #--------------------------------------------------------------------------                                      
    for k in range(kfold):
        for l,level in enumerate(levels):
            for s,state in enumerate(states):
                for p,a in itertools.product( range(s_len),range(a_len) ):
                                                                            
                    dict_len=len(lcss_trials_mat[k][l][s][a])
                    if dict_len == 0:
                        scalar = 1
                        if DB_PRINT:
                            print 'dictionary length is 0 for fold: ' + str(k) + ' level: ' + str(l) + ' state: ' + str(s) + ' axes: ' + str(a)
                    else:
                        scalar=1.0/dict_len
                    
                    # Extract an lcss string between the single validate instance and each of the dictionary entries
                    for i in range(dict_len):                          
                        
                        if i==0:
                            if DB_PRINT:
                                print 'fold: ' + str(k) + ' level: ' + str(l) + ' state: ' + str(s) + ' axes: ' + str(a)
                            # Get encode validate string for permutated state          #validate index
                            strSeq1 = allTrialLabels[ states[p] ][ data_folder_names[ kf_list[k][1][0] ]][level][ axes[a] ] 
                            lbl.encodeRCBHTList(strSeq1,level)                              # still need to encode the validate instance

                        # Get ith dictionary for training data for current state
                        strSeq2     = lcss_trials_mat[k][l][s][a][i][0]
                        if strSeq2!=[]: # check for leftover emtpy lists (need to fix)
                            strSeq2_freq= lcss_trials_mat[k][l][s][a][i][1]
                                                                  
                            temp=lcs.LCS.longestCommonSubsequence(strSeq1,strSeq2)  
                            dist=len(temp)
                               
                            # Add metric for all axes, it gives a full states determination
                            if dist!=0:
                                permutation_matrix[k][l][s][p*a_len+a] += log10(strSeq2_freq) + log10(dist)
                                #permutation_matrix[k][l][s][p*a_len+a] += 1.0/log10(strSeq2_freq) + pow(dist,2) # TODO Need to add condition for zero difsion
                                #permutation_matrix[k][l][s][p*a_len+a] += 1.0/strSeq2_freq + pow(dist,2)
                    permutation_matrix[k][l][s][p*a_len+a] *= scalar
                    
                    
    # Save Permutation Metric
    #pickle
#    with open(classification_dir+'/'+permutation_matrix_pickle, 'wb') as handle:
#        pickle.dump(sim_state_metric, handle, protocol=pickle.HIGHEST_PROTOCOL) 
#           
#    # Save binary numpy to file (maybe more efficient than pickle and can be opened with np.loadtxt)
#    fd = open(os.path.join(classification_dir,permutation_matrix_pickle[:-7]+'.txt'), "wb")
#    np.save(fd,sim_state_metric)  
#    fd.close()                 
     
                   
    # Similarity State Metric
    # Compute sum of similarities across axes for each fold/state/level (1x24)->(1x4)
    for k in range(kfold):
        for l in range(l_len):
            for s in range(s_len): 
                for p,a in itertools.product( range(s_len),range(a_len) ):                    
                    sim_state_metric[k][l][s][p] += permutation_matrix[k][l][s][p*a_len+a]                   

                        
    # Save Similarity State Metric
    #pickle
    with open(classification_dir+'/'+sim_state_metric_pickle, 'wb') as handle:
        pickle.dump(sim_state_metric, handle, protocol=pickle.HIGHEST_PROTOCOL) 
           
    # Save binary numpy to file (maybe more efficient than pickle and can be opened with np.loadtxt)
    fd = open(os.path.join(classification_dir,sim_state_metric_pickle[:-7]+'.txt'), "wb")
    np.save(fd,sim_state_metric)  
    fd.close()                    
     
    # A. Find average correct classifications.          
    #   Find max value across state and use that to generate a binary matrix. It represents which states correctly performed the right classification.
    #   Sum accross folds, divide by the number of folds.
    #   Each diagonal entry tells you the accuracy per state.
    # The sum of the trace/4 tells you overall task accuracy
    for l in range(l_len):    
        for s in range(s_len):          
            for k in range(kfold): 
                if np.sum(sim_state_metric[k][l][s]!=0):
                    m=np.argmax(sim_state_metric[k][l][s]) # which permutation was most likely for each state
                    classification_vector[l][s][m]+=1
                    
            # Once we have iterated by all the folds we can do an element-wise division by fold number        
            classification_vector[l][s]/=float(kfold)

    # Save Classfication Vector
    #pickle
    with open(classification_dir+'/'+classification_matrix_pickle, 'wb') as handle:
        pickle.dump(sim_state_metric, handle, protocol=pickle.HIGHEST_PROTOCOL) 
           
    # Save binary numpy to file (maybe more efficient than pickle and can be opened with np.loadtxt)
    fd = open(os.path.join(classification_dir,classification_matrix_pickle[:-7]+'.txt'), "wb")
    np.save(fd,sim_state_metric)  
    fd.close()             
        
    # B. Also find average similarity metric value and divided by the total. Compare with A.
    for l in range(l_len):        
        for s in range(s_len):               
            for k in range(kfold): 
                avg_prob_vector[l][s]+=sim_state_metric[k][l][s] # which permutation was most likely for each state   
                
            # Divide matrix by number of folds
            avg_prob_vector[l][s]/=float(kfold)        
            
            # Now we want to normalize the 4 values per given state
            avg_prob_vector[l][s]=np.true_divide(avg_prob_vector[l][s],np.sum(avg_prob_vector[l][s]))
            
    # Save avg_prob_vector_pickle
    #pickle
    with open(classification_dir+'/'+avg_prob_vector_pickle, 'wb') as handle:
        pickle.dump(sim_state_metric, handle, protocol=pickle.HIGHEST_PROTOCOL) 
           
    # Save binary numpy to file (maybe more efficient than pickle and can be opened with np.loadtxt)
    fd = open(os.path.join(classification_dir,avg_prob_vector_pickle[:-7]+'.txt'), "wb")
    np.save(fd,sim_state_metric)  
    fd.close()             
            
    # Results presentation (print,pickle,file)
    res=np.zeros( (l_len,s_len) )
    for l in range(l_len):
        for ctr in range(s_len):
            res[l]=classification_vector[l].diagonal()
            print 'For level: [',l+1,'/',l_len,'] and State [',ctr+1,'/',s_len,'] accuracy is: ',res[l][ctr],'\n'
        print 'Task accuracy for level: ',l+1, ' is: ',np.sum(res[l])/float(s_len),'\n'

    # Save results
    #pickle
    with open(classification_dir+'/'+lcss_results_pickle, 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)            
    # Save numpy to file
    #np.savetxt( os.path.join(classification_dir,results_pickle[:-7]+'.txt'), res)  
    with file(os.path.join(classification_dir,lcss_results_pickle[:-7]+'.txt'), 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(res.shape))
        for data_slice in res:
            np.savetxt(outfile, data_slice, fmt='%0.3f')
            outfile.write('# New slice\n')            
            
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

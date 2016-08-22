''' 
TODO: classify using 3 level labels together.
TODO: online classification.

Currently this program requires clean data set. Meaning: 3 folders only: Segments, 
Composites, and llBehaviors. Each folder must have only 6 files associated with _Fx, 
_Fy, _Fz, _Mx, _My, and _Mz. There should also be a State.dat file outside the three folders.
If more data is encountered it will not be processed at this time.

Structures:
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

# system
import os
import shutil

# General
from copy import deepcopy
import cPickle as pickle

# matrix
import numpy as np 

# Iterations
import itertools

# parsing
import util.output_features                     as output_features
import data_parser.data_folder_parser           as data_folder_parser
import feature_extractor.data_feature_extractor as data_feature_extractor

# classification
from sklearn import cross_validation
#from datasketch import MinHash

# debugging
import ipdb, traceback,sys#,code
ipdb.set_trace()

#------------------------------------------------------------------------------

# Globals
global DB_PRINT
DB_PRINT=1

#------------------------------------------------------------------------------

global states
global levels
global axes

# lists
states=['approach','rotation','insertion','mating']
levels=['primitive', 'composite', 'llbehavior']
axes=['Fx','Fy','Fz','Mx','My','Mz']
train_validate=['train','validate']

# lenghts
s_len=len(states)
l_len=len(levels)
a_len=len(axes)
tv_len=len(train_validate)
#------------------------------------------------------------------------------

## Flags
initFlag=0

successFlag=0
failureFlag=0
hlStatesFlag=1
classification=1

# For success/failure/hlStates there are 2 types of output: (i) the output per one trial, (ii) the output per all trais
output_per_one_trial_flag=0 

# For classification:
get_allTrial_Labels=0
#------------------------------------------------------------------------------

# What kind of success_strategy will you analyze
success_strategy='SIM_HIRO_ONE_SA_SUCCESS'
failure_strategy="SIM_HIRO_ONE_SA_ERROR_CHARAC_Prob"
strategy=success_strategy # default value. used in hblstates

#------------------------------------------------------------------------------

# Pickle folders
hlb_pickle                  ='allTrials_hlbStates.pickle'
state_probabilities_pickle  ='state_prob_onelevel.pickle'
results_pickle              ='results.pickle'
#------------------------------------------------------------------------------
# Folder names
data_folder_names=[]        # Filtered to only take relevant folders
orig_data_folder_names=[]
failure_data_folder_names=[]

#------------------------------------------------------------------------------

# Set program paths
results_dir='/home/vmrguser/sc/research/AIST/Results/ForceControl/'
cur_dir = os.path.dirname(os.path.realpath(__file__))
base_dir = cur_dir
os.chdir(base_dir)

#------------------------------------------------------------------------------

# my training data
directory='my_training_data'
train_data_dir=os.path.join(base_dir, '..', directory)

if not os.path.exists(train_data_dir):
    os.makedirs(train_data_dir)
    
# If the directory already exists, recursively remove it.     
else:
    shutil.rmtree(train_data_dir)
    os.makedirs(train_data_dir)

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
                    data_feature_extractor.extract_features(dict_all[data_folder_name],folder_dims)
                    
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
                    data_feature_extractor.extract_features(dict_all[data_folder_name],folder_dims)
                    allTrialLabels[data_folder_name]=deepcopy(dict_all[data_folder_name]) 
            except:
                type, value, tb = sys.exc_info()
                traceback.print_exc()
                ipdb.post_mortem(tb)
                
            # Save allTrialLabels to File using pickle
            with open(data_folder_prefix+'/allTrials_failure.pickle', 'wb') as handle:
                pickle.dump(allTrialLabels, handle, protocol=pickle.HIGHEST_PROTOCOL)            
            # To load use:
            #with open('filename.pickle', 'rb') as handle:
            #   unserialized_data = pickle.load(handle)                    
                
            if not os.path.exists(os.path.join(base_dir, '..', 'my_training_data', 'img_of_fail')):
                 os.makedirs(os.path.join(base_dir, '..', 'my_training_data', 'img_of_fail'))                 
            
            # Write labels and images to file. 2 choices: individual iterations or all iterations per state.            
            try:
                # label f indicates SUCCESS. Have a file and a place to put images
                if output_per_one_trial_flag:
                    output_features.output_sample_one_trial(file_for_fail_set, 'f', dict_cooked_from_folder, os.path.join(base_dir, '..', 'my_training_data', 'img_of_failure'))            
                else:
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
        files_for_states[state] = open(os.path.join(base_dir,'..', 'my_training_data', "training_set_for_"+state), 'w')    

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
                        data_feature_extractor.extract_features(dict_all[data_folder_name][state],folder_dims[state])                                                                                          
                                    
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
            # To load use:
            #with open('filename.pickle', 'rb') as handle:
            #   unserialized_data = pickle.load(handle)                            
                        
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

''' Take each of the state labels as well as the whole task label if they exist and use them for classification.
Cross-fold validation will be used for testing. Cross validation will be implemented through sklearn's cross validation module KFold
http://scikit-learn.org/stable/modules/cross_validation.html.
Currently doing offline classification'''            
if classification:        

    # Folders for storing results
    if strategy=='':                        # if this path does not exist
        strategy=raw_input('Please enter a name for the strategy: \n')
        get_allTrial_Labels=1

        # Assign the current strategy directory    
        strat_dir = os.path.join(train_data_dir,strategy,)      
    
    # Create a classification folder and get the results folder
    directory='classification'
    classification_dir=os.path.join(strat_dir,directory)
    if not os.path.exists(classification_dir):
        os.makedirs(classification_dir) 
            
    # If you want to deserialize saved data
    if get_allTrial_Labels:        
        with open(os.path.join(strat_dir,hlb_pickle), 'rb') as handle:
           unserialized_data = pickle.load(handle)            

    # Get Folder names
    data_folder_prefix = os.path.join(results_dir, strategy)    
    
    # Initialize structures (from previous runs)
    folder_dims={}
    dict_dims={}
    dict_all={}      
           
    # Minhash Objects to estimate the jaccard distance for test and validation
    #tFx,tFy,tFz,tMx,tMy,tMz=MinHash(),MinHash(),MinHash(),MinHash(),MinHash(),MinHash() 
    #vFx,vFy,vFz,vMx,vMy,tMz=MinHash(),MinHash(),MinHash(),MinHash(),MinHash(),MinHash()      
    
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
    # Store current data folder names in a list so we can iterate it through it
    folder_names_list=list(data_folder_names)
       
    # Structures 
    # jacard axis t/v list:     contains 1x6 final train union set and test set. It is a 1x6 set per state/level/axis/train_validate (2x4x3x6)
    # state_probability array:  contains the final state_probability under fold/level: state/state (4x4)
    # permutation_matrix:       contains classification results permutations. level x states x (statesxaxes)    
    # avg_prob_vector:          containes normalized probabilities. level x states x states
    # classification_vector:    contains normalized correct classifications. level x states x states
    # kfold_list:               contains indeces for train/validate trials for each fold: kfold x train|validate x elems
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
                        
    # Populate jaccard_axis according to training trials and one test trial across folds.    
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
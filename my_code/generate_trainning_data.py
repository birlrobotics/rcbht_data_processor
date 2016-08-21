''' Currently this program requires clean data set. Meaning: 3 folders only: Segments, 
Composites, and llBehaviors. Each folder must have only 6 files associated with _Fx, 
_Fy, _Fz, _Mx, _My, and _Mz. There should also be a State.dat file outside the three folders.
If more data is encountered it will not be processed at this time.'''

# system
import os
import shutil

# General
from copy import deepcopy
import cPickle as pickle

# math
from math import ceil
import numpy as np 

# Iterations
import itertools

# parsing
import util.output_features                     as output_features
import data_parser.data_folder_parser           as data_folder_parser
import feature_extractor.data_feature_extractor as data_feature_extractor

# classification
from sklearn import cross_validation
from datasketch import MinHash

# debugging
import ipdb, traceback,sys#,code
#ipdb.set_trace()

#------------------------------------------------------------------------------

# Globals
global DB_PRINT
DB_PRINT=0

#------------------------------------------------------------------------------

global states
global levels
global axes

# lists
states=['approach','rotation','insertion','mating']
levels=['primitive', 'composite', 'llbehavior']
axes=['Fx','Fy','Fz','Mx','My','Mz']
train_validate=['train','validate']
#------------------------------------------------------------------------------

## Flags
successFlag=0
failureFlag=0
hlStatesFlag=1
classification=1

output_per_one_trial_flag=0 # if true, output is performed for each trial for all axis. Otherwise all trials for each axis is displayed.

#------------------------------------------------------------------------------

# What kind of success_strategy will you analyze
success_strategy='SIM_HIRO_ONE_SA_SUCCESS_T'
failure_strategy="SIM_HIRO_ONE_SA_ERROR_CHARAC_Prob"
strategy=success_strategy # default value. used in hblstates

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
            # To load use:
            #with open('filename.pickle', 'rb') as handle:
            #   unserialized_data = pickle.load(handle)                
            
            if not os.path.exists(os.path.join(base_dir, '..', 'my_training_data', 'img_of_success')):
                os.makedirs(os.path.join(base_dir, '..', 'my_training_data', 'img_of_success'))
                    
            try:
                # label s indicates SUCCESS. Have a file and a place to put images
                if output_per_one_trial_flag:
                    output_features.output_sample_one_trial(file_for_success_set, 's', dict_cooked_from_folder, os.path.join(base_dir, '..', 'my_training_data', 'img_of_success'))
                else:
                    output_features.output_sample_all_trial(file_for_success_set, 's', allTrialLabels,data_folder_names,numTrials,os.path.join(base_dir, '..', 'my_training_data',strategy))
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
            with open(data_folder_prefix+'/allTrials_hlbStates.pickle', 'wb') as handle:
                pickle.dump(allTrialLabels, handle, protocol=pickle.HIGHEST_PROTOCOL)            
            # To load use:
            #with open('filename.pickle', 'rb') as handle:
            #   unserialized_data = pickle.load(handle)                            
                        
            ## Create directories to keep both labels and images in file
            for state in dict_cooked_from_folder:
                hlb_dir=strategy + '/' + 'img_of_'+state
                if not os.path.exists(os.path.join(base_dir, '..', 'my_training_data', hlb_dir)):
                    os.makedirs(os.path.join(base_dir, '..', 'my_training_data', hlb_dir))
                
                # Write labels and images to file. 2 choices: individual iterations or all iterations per state.
                try:
                    # label 1 indicates SUCCESS. Have a file and a place to put images
                    if output_per_one_trial_flag:
                        output_features.output_sample_one_trial(files_for_states[state], str(states.index(state)), dict_cooked_from_folder[state], os.path.join(base_dir, '..', 'my_training_data', 'img_of_'+state))
                    else:
                        output_features.output_sample_all_trial(files_for_states[state], str(states.index(state)), allTrialLabels[state],data_folder_names,numTrials,os.path.join(base_dir, '..', 'my_training_data',hlb_dir))
            
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
    
    # Initialize structures (from previous runs)
    folder_dims={}
    dict_dims={}
    dict_all={}    
    
    get_allTrial_Labels=0
    
    # lists
    states=['mating']
    levels=[ 'llbehavior']
    axes=['My']
    train_validate=['train','validate']
    
    # If you want to deserialize saved data
    if get_allTrial_Labels:
        data_folder_prefix = os.path.join(results_dir, success_strategy)
        with open(data_folder_prefix+'/allTrials_hlbStates.pickle', 'rb') as handle:
           unserialized_data = pickle.load(handle)    
           
    # Minhash Objects to estimate the jaccard distance for test and validation
    tFx,tFy,tFz,tMx,tMy,tMz=MinHash(),MinHash(),MinHash(),MinHash(),MinHash(),MinHash() 
    vFx,vFy,vFz,vMx,vMy,tMz=MinHash(),MinHash(),MinHash(),MinHash(),MinHash(),MinHash()      
    
    # Initialize k-fold data to a valid integer
    kfold=10
    
    # Store current data folder names in a list so we can iterate it through it
    folder_names_list=list(data_folder_names)
       
    # Structures 
    # 1. crossfold dic:        contains labels for training under the state/level/traint_test/axis structure: 4xnumTrialsx3x6
    # 2. jacard axis_list:   contains 1x6 final train union set and test set. It is a 1x6 set per state/level/axis/train_validate (2x4x3x6)
    # 3. jacard_final array:   contains the final jaccard distances under state/level (4x3x1)
    crossfold={}
    jaccard_axis=[ [ [ [ [ [] for a in range(len(axes))]                          # innermost entry
                                for l in range(len(levels))] 
                                    for s in range(len(states))]
                                        for t in range(len(train_validate))]
                                            for k in range(kfold)]       
    # Create jacard_final: compute the distance between train/validate sets across permutations of states:
    # Within a given fold/level we want to compute the intersection of training with validation across states. 
    # ... App_train \cap App_validate | App_train \cap Rot_validate ... | App_train \cap Mat_validate ... (4x(4x6)
    # ... Rot_train \cap App_validate | Rot_train \cap Rot_validate ... | Rot_train \cap Mat_validate 
    # ... 
    # ... Mat_train \cap App_validate | Mat_train \cap Rot_validate ... | Mat_train \cap Mat_validate
    jaccard_final       =np.zeros( ( kfold,len(levels),len(states),len(axes)) )
    classification_matrix=np.zeros( ( kfold,len(levels),len(states),len( len(states)*len(axes) ),1) )
    classification_vector=np.zeros( ( len(levels),len(states),len(states),1) )
    
    # Counters
    k=0 # kfold
    s=0 # state
    l=0 # level
    a=0 # axis
    t=0 # training and validation
    
    # Create our crossfold dic and jacard axis dict
    for state in states:   
        crossfold[state]={}        
        for level in levels:
            crossfold[state][level]={}            
            for t in train_validate:
                crossfold[state][level][t]={}                
                for axis in axes:
                    crossfold[state][level][t][axis]={}
                    
    # Crossfold training and testing generator
    kf=cross_validation.KFold(numTrials, n_iter=numTrials) # equivalent to leave one out
    # Extract result onto a lists
    kf_list=list(kf)
    # Extract the indeces that belong to the training and validation lists    
    train_idx,validate_idx=kf_list

    # Populate the crossfold and run tests according to the number of crossfolds. 
    # NEed to average at the end using the np.array mean method
    # TODO: integrate enumerate to have both indeces and dictionary entries
    for k in range(kfold):
        for s,state in enumerate(states):
            for l,level in enumerate(levels):
                idx=0 # used to transition from training to testing
                for t,tv in enumerate(train_validate):   
                    r=len(kf_list[0][idx])                                  
                    for trial in xrange(r): # len of train/test instances                           
                        # Crossfold: copy all training items AND one test itme           
                        crossfold[state][level][tv].update(allTrialLabels[state][kf_list[trial][idx]][level])
                    idx=idx+1    
                    
                    # All trial/test entries for a given k-fold/state/level/axis will be ready. Get their union. only one entry left
                    # This stucture will then need to be used to compute the jaccard distance (intersection/union) with the test set for all states/levels in a given fold.
                    for a,axis in enumerate(axes):
                        
                        # Get labels 1st training set in a given axis
                        jaccard_axis[k][t][s][l][a]=set(crossfold[state][level][tv][axis][0]) 
                        
                        # Loop from 2nd element to last and perform the union of the training set. 
                        # We are doing leave-one-out cross validation, so we don't do this for the test set
                        if len(r) > 1:
                            for i in xrange(1,r):
                                jaccard_axis[k][t][s][l][a].union(set(crossfold[state][level][tv][axis][i]))                    

                    # Within a given fold/level we want to compute the intersection of all axes of training with validation across states. 
                    # ... App_train \cap App_validate | App_train \cap Rot_validate ... | App_train \cap Mat_validate
                    # ... Mat_train \cap App_validate | Mat_train \cap Rot_validate ... | Mat_train \cap Mat_validate
                    # Iterate through 2D axis (6x6) one more time
                    for i,j in itertools.product(range(len(axes)),range(len(axes))):
                        jaccard_final[k][l][i][j]= ( float(  len(jaccard_axis[k][0][s][l][i].intersection(jaccard_axis[k][1][s][l][j]))) / 
                                                     float(  len(jaccard_axis[k][0][s][l][i]).union      (jaccard_axis[k][1][s][l][j]))
                                                    )

            
    # Last stage:
    # A. Compute a single normalized probability for each test. 
    # B. Compute the average value across folds.
    #    i.  From the state permutations (4x4) we need to find the maximum in each row to find the best candidate. 
    #    ii. Then, we need to know if the correct state was detected. To this end, we look at the diagonals.
    for k in range(kfold):
        for s in range(len(states)):
            for l in range(len(levels)):
                
                # Sum the probabilities of all axes to get a final probability number and then normalize.
                for i,a in itertools.product(range(len(states)),range(len(axes))):
                    classification_matrix[k][l][s][i*a] += jaccard_final[k][l][s][a]
                classification_matrix[k][l][s]/=len(axes)
                
    # Add across folds
    for l in range(len(levels)):
        for s in range(len(states)):
            for k in range(kfold):                      
                # Sum accross folds
                classification_vector[l][s] +=classification_matrix[k][l][s]
                
            # Print info
            print 'classification_vector row: ' + s + ' and values: ' + classification_vector[l][s]
                                                          
                    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
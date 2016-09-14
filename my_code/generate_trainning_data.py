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
import types
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
from inc.states import states

# classification
from sklearn import cross_validation

#from datasketch import MinHash

# LCS
import lcs.LCS
import lcs.rcbht_lbl_conversion as lbl

from datasketch import MinHash
import collections
#------------------------------------------------------------------------------

# Globals
global DB_PRINT
DB_PRINT=1

#------------------------------------------------------------------------------

global states
global levels
global axes


class wrenchActionGrammar:

    def __init__(self,ss,fs='SIM_HIRO_ONE_SA_ERROR_CHARAC_Prob',armSide='right'):
        # lists
        self.states         =['approach','rotation','insertion','mating']
        self.levels         =['primitive', 'composite', 'llbehavior']
        self.axes           =['Fx','Fy','Fz','Mx','My','Mz']
        self.train_validate =['train','validate']

        # Dictionaries
        self.allTrialLabels={}

        # lenghts
        self.s_len=len(self.states)
        self.l_len=len(self.levels)
        self.a_len=len(self.axes)
        self.tv_len=len(self.train_validate)
        #------------------------------------------------------------------------------

        ## Flags
        #self.initFlag        =0              # Used for an initialization routine in creating jaccard axis
        self.sliceLabels     =1              # Determines whether we slice labels so that they all axes have equal number of labels
        self.loadFromFileFlag=1              # Determines wehther we load saved structures to the program when they have been computed once.

        #self.successFlag     =0              # Find labels for entire task
        #self.failureFlag     =0
        #self.hlStatesFlag    =0              # Separate labels by state
        # For success/failure/hlStates there are 2 types of output: (i) the output per one trial, (ii) the output per all trais
        self.output_per_one_trial_flag=0

        #------------------------------------------------------------------------------
        # stream processing
        self.get_allTrial_Labels=0
        #------------------------------------------------------------------------------
        #self.lcss=1
        #------------------------------------------------------------------------------

        # Robot configuration
        self.leftArm=0
        self.rightArm=0
        self.armSide=str(armSide.lower())

        if self.armSide == 'left':
            self.leftArm=1
        if self.armSide == 'right':
            self.rightArm=1
        self.armFlag=[self.leftArm,self.rightArm]
        if self.armFlag[0] and self.armFlag[1]:
            self.armIndex=2;
        else:
            self.armIndex=1;

        #------------------------------------------------------------------------------
        # What kind of success_strategy will you analyze
        self.success_strategy=ss
        self.failure_strategy=fs
        self.strategy=self.success_strategy # default value. used in hblstates

        #------------------------------------------------------------------------------
        # Pickle folders
        #------------------------------------------------------------------------------

        # jaccard
        self.hlb_pickle                  = '_allTrials_hlbStates.pickle'
        self.state_probabilities_pickle  = 'state_prob_onelevel.pickle'
        self.results_pickle              = 'results.pickle'

        #------------------------------------------------------------------------------
        # lcss
        self.trial_lcss_mat_pickle         = '_trial_lcss_mat.pickle'
        self.sim_state_metric_pickle       = '_sim_state_metric.pickle'
        self.lcss_results_pickle           = '_lcss_results.pickle'
        self.permutation_matrix_pickle     = '_permutation_matrix.pickle'
        self.classification_matrix_pickle  = '_classification_matrix.pickle'
        self.avg_prob_vector_pickle        = '_avg_prob_vector.pickle'
        self.acuracy_numTrials_pickle      = '_accuracy_numTrials.pickle'

        #------------------------------------------------------------------------------
        # Folder names
        self.data_folder_names           =[]        # Filtered to only take relevant folders
        self.orig_data_folder_names      =[]
        self.failure_data_folder_names   =[]

        #------------------------------------------------------------------------------

        # Set program paths
        self.results_dir ='/home/vmrguser/sc/research/AIST/Results/ForceControl/'
        self.cur_dir     = os.path.dirname(os.path.realpath(__file__))
        self.base_dir    = self.cur_dir
        os.chdir(self.base_dir)

        #------------------------------------------------------------------------------

        # my training data
        self.directory='my_training_data'
        self.train_data_dir=os.path.join(self.base_dir, '..', self.directory)

        if not os.path.exists(self.train_data_dir):
            os.makedirs(self.train_data_dir)

        # Experimental folder
        self.exp_dir=os.path.join(self.train_data_dir,self.strategy)

        # armSide Directory: save results according to left/right arm.
        self.armSide_dir=os.path.join(self.exp_dir,self.armSide)

        # allTrialLabels path to pickle file
        self.allTrialLabels_path=os.path.join(self.armSide_dir,self.armSide+self.hlb_pickle)

        # img_directory for successful strategies
        self.allTrials_success_dir=os.path.join(self.armSide_dir,'allTrials_success')
        if not os.path.exists( self.allTrials_success_dir ):
            os.makedirs( self.allTrials_success_dir )

        # img directory for failure strategies
        self.allTrials_failure_dir=os.path.join(self.armSide_dir,'allTrials_failure')
        if not os.path.exists( self.allTrials_failure_dir ):
            os.makedirs( self.allTrials_failure_dir )

        # img_directory: contains images that show color coded patters for labeles for all levels/states/axes.
        for state in self.states:
            self.hlb_dir='img_of_'+state
            if not os.path.exists( os.path.join(self.allTrials_success_dir, self.hlb_dir) ):
                os.makedirs(os.path.join(self.allTrials_success_dir, self.hlb_dir))

    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------

    # What kind of data should we collect?
    # - Success
    # - Failure
    # + Generate high level states data

    # 1. Get data for success tasks for a given success_strategy
#    if successFlag:
#
#        # Dictionary building blocks
#        folder_dims={}
#        dict_dims={}
#        dict_all={}
#        allTrialLabels={}
#
#        # Assign the current strategy directory
#        strategy             = success_strategy
#        exp_dir            = os.path.join(train_data_dir,strategy,)
#        file_for_success_set = open(os.path.join(exp_dir, 'training_set_of_success'), 'w')
#
#
#        # Get Folder names
#        #data_folder_prefix = os.path.join(base_dir, '..', 'my_data', success_strategy)
#        data_folder_prefix = os.path.join(results_dir, strategy)
#        orig_data_folder_names = os.listdir(data_folder_prefix)
#
#        # Remove undesired folders
#        for data_folder_name in orig_data_folder_names:
#            if data_folder_name[:2] == '20':
#                data_folder_names.append(data_folder_name)
#
#        numTrials=len(data_folder_names)
#
#        # Get max number of iterations for each level/axis for entire experiment sets
#        # Construct dictionary to hold dimensions. Need level and axis sub-spaces.
#        for level in levels:
#            folder_dims[level] = {}
#            for axis in axes:
#                folder_dims[level][axis]={}
#
#        # Create a dictionary structure for all trials, RCBHT levels, and axis.
#        for data_folder_name in data_folder_names:
#            data_folder_full_path = os.path.join(data_folder_prefix, data_folder_name)
#            if DB_PRINT:
#                print data_folder_full_path
#
#            dict_cooked_from_folder = data_folder_parser.parse_folder(data_folder_full_path,armFlag,self.levels,split_by_states=False)
#            if dict_cooked_from_folder == None:
#                continue
#            else:
#                dict_all[data_folder_name]=dict_cooked_from_folder
#
#        if bool(dict_all):
#            # Once dict_cooked_from_folder exists, get dimensions of level/axis for each folder
#            for data_folder_name in data_folder_names:
#                for level in levels:
#                    for axis in axes:
#                        folder_dims[level][axis] = len(dict_all[data_folder_name][level][axis])
#                dict_dims[data_folder_name]=deepcopy(folder_dims)
#
#            if bool(dict_dims):
#                # Only keep the largest dimensions for each level/axis
#                for level in levels:
#                    for axis in axes:
#                        for data_folder_name in data_folder_names:
#                            commonString=dict_dims[data_folder_name][level][axis]
#                            if commonString > folder_dims[level][axis]:
#                                folder_dims[level][axis]=commonString
#
#                # For each trial, take the dictionary and reshape it (change number of iterations), output is the same NUMBER of labels for all axes.
#                try:
#                    for data_folder_name in data_folder_names:
#                        data_feature_extractor.extract_features(dict_all[data_folder_name],folder_dims,sliceLabels)
#
#                        # Create the allTrialsLables structure. It is organized by: state/trials/level/axis. Only contains labels.
#                        allTrialLabels[data_folder_name]=deepcopy(dict_all[data_folder_name])
#                except:
#                    print 'error found in extract_features'
#                    type, value, tb = sys.exc_info()
#                    traceback.print_exc()
#                    ipdb.post_mortem(tb)
#
#                # Save allTrialLabels to File using pickle
#                with open(data_folder_prefix+'/allTrials_success.pickle', 'wb') as handle:
#                    pickle.dump(allTrialLabels, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#                if not os.path.exists(os.path.join(base_dir, '..', 'my_training_data', 'img_of_success')):
#                    os.makedirs(os.path.join(base_dir, '..', 'my_training_data', 'img_of_success'))
#
#                try:
#                    # label s indicates SUCCESS. Have a file and a place to put images
#                    if output_per_one_trial_flag:
#                        output_features.output_sample_one_trial(file_for_success_set, 's', dict_cooked_from_folder, os.path.join(base_dir, '..', 'my_training_data', 'img_of_success'))
#                    else:
#                        if sliceLabels: # TODO Currently only execute if we are slciing. Need to modify image.
#                            output_features.output_sample_all_trial(file_for_success_set, 's', allTrialLabels,data_folder_names,numTrials,exp_dir)
#                except:
#                    type, value, tb = sys.exc_info()
#                    traceback.print_exc()
#                    ipdb.post_mortem(tb)
#            else:
#                print 'The success dictionary dict_dims is empty'
#        else:
#            print 'The success dictionary dict_all is empty'
#
#    #-------------------------------------------------------------------------
#    ##FAILURE ANALYSIS
#    #------------------------------------------------------------------------
#
#    if failureFlag:
#
#        # Initialize structures
#        folder_dims={}
#        dict_dims={}
#        dict_all={}
#        allTrialLabels={}
#
#        strategy=failure_strategy
#        file_for_fail_set = open(os.path.join(base_dir, '..', 'my_training_data', strategy, 'training_set_of_fail'), 'w')
#
#        # Read failure data
#        data_folder_prefix = os.path.join(results_dir, failure_strategy)
#        orig_data_folder_names = os.listdir(data_folder_prefix)
#
#        # Remove undesired folders
#        for data_folder_name in orig_data_folder_names:
#            if data_folder_name[:2] == 'ex' or data_folder_name[:2] == 'FC':
#                failure_data_folder_names.append(data_folder_name)
#
#        numTrials=len(failure_data_folder_names)
#
#        # Get max number of iterations for each level/axis for entire experiment sets
#        # Construct dictionary to hold dimensions. Need level and axis sub-spaces.
#        for level in levels:
#            folder_dims[level] = {}
#            for axis in axes:
#                folder_dims[level][axis]={}
#
#        # Get full path for each folder name
#        for data_folder_name in failure_data_folder_names:
#            data_folder_full_path = os.path.join(data_folder_prefix, data_folder_name)
#            if DB_PRINT:
#                print data_folder_full_path
#
#            # Get dictionary cooked from all folders
#            dict_cooked_from_folder = data_folder_parser.parse_folder(data_folder_full_path,armFlag,self.levels,split_by_states=False)
#            if dict_cooked_from_folder == None:
#                continue
#            else:
#                dict_all[data_folder_name]=dict_cooked_from_folder
#
#        # Once dict_cooked_from_folder exists, get dimensions of level/axis for each folder
#        if bool(dict_all):
#            for data_folder_name in failure_data_folder_names:
#                for level in levels:
#                    for axis in axes:
#                        folder_dims[level][axis] = len(dict_all[data_folder_name][level][axis])
#                dict_dims[data_folder_name]=deepcopy(folder_dims)
#
#            if bool(dict_dims):
#                # Only keep the largest dimensions for each level/axis
#                for level in levels:
#                    for axis in axes:
#                        for data_folder_name in failure_data_folder_names:
#                            temp=dict_dims[data_folder_name][level][axis]
#                            if temp > folder_dims[level][axis]:
#                                folder_dims[level][axis]=temp
#
#                # For one trial, take the dictionary and reshape it (change number of iterations) so we have the same NUMBER of labels for all axes.
#                # Then only return the labels.
#                # Currently we take the max number of iterations in any given trial/level/axis.
#                try:
#                    for data_folder_name in failure_data_folder_names:
#                        data_feature_extractor.extract_features(dict_all[data_folder_name],folder_dims,sliceLabels)
#                        allTrialLabels[data_folder_name]=deepcopy(dict_all[data_folder_name])
#                except:
#                    type, value, tb = sys.exc_info()
#                    traceback.print_exc()
#                    ipdb.post_mortem(tb)
#
#                # Save allTrialLabels to File using pickle
#                with open(data_folder_prefix+'/allTrials_failure.pickle', 'wb') as handle:
#                    pickle.dump(allTrialLabels, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#                if not os.path.exists(os.path.join(base_dir, '..', 'my_training_data', 'img_of_fail')):
#                     os.makedirs(os.path.join(base_dir, '..', 'my_training_data', 'img_of_fail'))
#
#                # Write labels and images to file. 2 choices: individual iterations or all iterations per state.
#                try:
#                    # label f indicates SUCCESS. Have a file and a place to put images
#                    if output_per_one_trial_flag:
#                        output_features.output_sample_one_trial(file_for_fail_set, 'f', dict_cooked_from_folder, os.path.join(base_dir, '..', 'my_training_data', 'img_of_failure'))
#                    else:
#                        if sliceLabels: # TODO Currently only execute if we are slciing. Need to modify image.
#                            output_features.output_sample_all_trial(file_for_fail_set, 'f', allTrialLabels,failure_data_folder_names, numTrials,os.path.join(base_dir, '..', 'my_training_data',failure_strategy))
#                except:
#                    type, value, tb = sys.exc_info()
#                    traceback.print_exc()
#                    ipdb.post_mortem(tb)
#            else:
#                print 'The failure dictionary dict_dims is empty'
#        else:
#            print 'The failure dictionary dict_all is empty'
#
#        # Clear up
#        folder_dims={}
#        dict_dims={}
#        dict_all={}
#        allTrialLabels={}
    #-------------------------------------------------------------------------
    ## Parse information by State
    #-------------------------------------------------------------------------
    def computeStatesLabels(self):

        # First we check to see if the output structure of this function has already been created, if so. Skip the work.
         # Check to see if the file exists. If the file exists and we want to load it...
        if os.path.isfile(os.path.join(self.armSide_dir,self.hlb_pickle) ):
            if self.loadFromFileFlag:
                with open(os.path.join(self.armSide_dir,self.hlb_pickle), 'rb') as handle:
                    self.allTrialLabels = pickle.load(handle)

                    # Initialize structures
                    self.folder_dims={}
                    self.dict_dims={}
                    self.dict_all={}
                    self.allTrialLabels={}

                    self.files_for_states = {}

                    # Folders for storing results
                    if self.strategy=='':                        # if this path does not exist
                        self.strategy=raw_input('Please enter a name for the strategy: \n')
                        # Assign the current strategy directory
                        self.exp_dir = os.path.join(self.train_data_dir,self.strategy,)

                    # Collect folder names located in the sucess results directory
                    self.data_folder_prefix = os.path.join(self.results_dir, self.success_strategy)
                    self.orig_data_folder_names = os.listdir(self.data_folder_prefix)

                    # Remove undesired folders
                    for data_folder_name in self.orig_data_folder_names:
                        if data_folder_name[:2] == '20':
                            self.data_folder_names.append(data_folder_name)

                    self.numTrials=len(self.data_folder_names)
        else:
            # Initialize structures
            self.folder_dims={}
            self.dict_dims={}
            self.dict_all={}
            self.allTrialLabels={}

            self.files_for_states = {}
            # Open files for each of the states we will analyze
            for state in self.states:
                self.files_for_states[state] = open(os.path.join(self.exp_dir, "training_set_for_"+state), 'w')

            # Folders for storing results
            if self.strategy=='':                        # if this path does not exist
                self.strategy=raw_input('Please enter a name for the strategy: \n')
                # Assign the current strategy directory
                self.exp_dir = os.path.join(self.train_data_dir,self.strategy,)

            # Collect folder names located in the sucess results directory
            self.data_folder_prefix = os.path.join(self.results_dir, self.success_strategy)
            self.orig_data_folder_names = os.listdir(self.data_folder_prefix)

            # Remove undesired folders
            for data_folder_name in self.orig_data_folder_names:
                if data_folder_name[:2] == '20':
                    self.data_folder_names.append(data_folder_name)

            self.numTrials=len(self.data_folder_names)

            # Get max number of iterations for each level/axis for entire experiment sets
            # Construct dictionary to hold dimensions. Need level and axis sub-spaces.
            for state in self.states:
                self.folder_dims[state] = {}
                for level in self.levels:
                    self.folder_dims[state][level] = {}
                    for axis in self.axes:
                        self.folder_dims[state][level][axis]={}

            # Create a dictionary structure for all trials, RCBHT levels, and axis.
            for data_folder_name in self.data_folder_names:
                self.data_folder_full_path = os.path.join(self.data_folder_prefix, data_folder_name)
                if DB_PRINT:
                    print self.data_folder_full_path

                self.dict_cooked_from_folder = data_folder_parser.parse_folder(self.data_folder_full_path, self.armFlag, self.levels, split_by_states=True)
                if self.dict_cooked_from_folder == None:
                    continue
                else:
                    self.dict_all[data_folder_name]=self.dict_cooked_from_folder

            # Generate a dictionary of dimensions for all trials/states/levels/axis
            if bool(self.dict_all):
                for data_folder_name in self.data_folder_names:
                    for state in self.states:
                        for level in self.levels:
                            for axis in self.axes:
                                self.folder_dims[state][level][axis] = len(self.dict_all[data_folder_name][state][level][axis])

                    self.dict_dims[data_folder_name]=deepcopy(self.folder_dims)

                # Only keep the largest dimensions for each state/level/axis
                if bool(self.dict_dims):
                    for data_folder_name in self.data_folder_names:
                        for state in self.states:
                            for level in self.levels:
                                for axis in self.axes:
                                    temp=self.dict_dims[data_folder_name][state][level][axis]
                                    if temp > self.folder_dims[state][level][axis]:
                                        self.folder_dims[state][level][axis]=temp

                    # For one trial, take the dictionary and reshape it (change number of iterations) so we have the same NUMBER of labels for all axes.
                    # Then only return the labels.
                    # Currently we take the max number of iterations in any given trial/level/axis.
                    self.allTrialLabels={}
                    for data_folder_name in self.data_folder_names:
                        for state in self.dict_cooked_from_folder:
    #                       data_feature_extractor.extract_features(dict_cooked_from_folder[state],folder_dims[state])
                            data_feature_extractor.extract_features(self.dict_all[data_folder_name][state],self.folder_dims[state],self.sliceLabels)

                    # Create the allTrialsLables structure. It is organized by: state/trials/level/axis. Only contains labels.
                    for state in self.dict_cooked_from_folder:
                        self.allTrialLabels[state]={}
                        for data_folder_name in self.data_folder_names:
                            self.allTrialLabels[state][data_folder_name]={}
                            self.allTrialLabels[state][data_folder_name]=deepcopy(self.dict_all[data_folder_name][state])

                    # Save allTrialLabels to File using pickle
                    with open( self.allTrialLabels_path, 'wb') as self.handle:
                        pickle.dump(self.allTrialLabels, self.handle, protocol=pickle.HIGHEST_PROTOCOL)

                    # Write labels and images to file. 2 choices: individual iterations or all iterations per state.
                    # label 1 indicates SUCCESS. Have a file and a place to put images
                    for state in self.dict_cooked_from_folder:

                        if self.output_per_one_trial_flag:
                            # Open files for each of the states we will analyze
                            tmp_dir = open(os.path.join(self.armSide_dir, "training_set_for_"+state), 'w')
                            self.output_features.output_sample_one_trial(self.files_for_states[state], str(self.states.index(state)), self.dict_cooked_from_folder[state], tmp_dir)

                        else:
                            tmp_dir=os.path.join(self.allTrials_success_dir,'img_of_'+state)
                            if self.sliceLabels: # TODO Currently only execute if we are slciing. Need to modify image.
                                output_features.output_sample_all_trial(self.files_for_states[state], str(self.states.index(state)), self.allTrialLabels[state],self.data_folder_names,self.numTrials,tmp_dir)

                else:
                    print 'The hlb dictionary dict_dims is empty'
            else:
                    print 'dict_all from hlb states is not available'



    #------------------------------------------------------------------------------

    ''' Take each of the state labels as well as the whole task label if they exist
    and use them for classification. Cross-fold validation will be used for testing.
    Cross validation will be implemented through sklearn's cross validation module
    KFold http://scikit-learn.org/stable/modules/cross_validation.html.
    Currently doing offline classification'''

    #--------------------------------------------------------------------------
    def lcss(self,levels):
    #--------------------------------------------------------------------------

        # Compute states for desired level(s)
        if isinstance(levels,types.ListType):
            self.levels=levels
            self.l_len=len(levels)

        # Folders for storing results
        if self.strategy=='':                        # if this path does not exist
            self.strategy=raw_input('Please enter a name for the strategy: \n')
            self.get_allTrial_Labels=1

            # Assign the current strategy directory
            self.exp_dir = os.path.join(self.train_data_dir,self.strategy)

    #--------------------------------------------------------------------------
    # Directories
    #--------------------------------------------------------------------------
        # Create a classification folder and get the results folder
        self.directory='lcss'
        self.classification_dir=os.path.join(self.armSide_dir,self.directory)
        if not os.path.exists(self.classification_dir):
            os.makedirs(self.classification_dir)

        # Create a path for the lcss_trials_mat matrix
        self.trial_lcss_mat_pickle_path=os.path.join(self.classification_dir,self.armSide + self.trial_lcss_mat_pickle)

        # Create a path for the similarity_permutation_matrix
        self.sim_permutation_matrix_pickle_path=os.path.join(self.classification_dir,self.armSide + self.permutation_matrix_pickle)

        # Create a path for the similarity matrix
        self.sim_state_metric_pickle_path=os.path.join(self.classification_dir,self.armSide + self.sim_state_metric_pickle)

        # Create a path for the classification matrix
        self.classification_matrix_pickle_path=os.path.join(self.classification_dir,self.armSide + self.classification_matrix_pickle)

        # Create a path for the results matrix for a single arm
        self.lcss_results_pickle_path=os.path.join(self.classification_dir,self.armSide + self.lcss_results_pickle)

        # Create a path for the accumulated accuracy results matrix for a single arm
        self.acuracy_numTrials_pickle_path=os.path.join(self.classification_dir,self.armSide + self.acuracy_numTrials_pickle)

        # Make sure you have access to the allTrialsLabels structures. Otherwise can't go on. First check if available from memory, otherwise from file.
        if not bool(self.allTrialLabels): #TODO: Cannot retrieve from memory after creating in computeStatesLabels... why?
            if os.path.isfile( self.allTrialLabels_path ) and self.loadFromFileFlag:
                    with open( self.allTrialLabels_path, 'rb') as handle:
                        self.allTrialLabels = pickle.load(handle)
            else:
                raise Exception('allTrialsLabels is not available. Cannot classify with empty structure')

        ## TODO: when you fix the loading of allTrialsLabels from memory, this next section of code should go in an else statement
        # Get Folder names in the results directory
        self.data_folder_prefix = os.path.join(self.results_dir, self.strategy)

        # Store current data folder names in a list so we can iterate it through it
        if not bool(self.data_folder_names):
            # Get Folder names
            #data_folder_prefix = os.path.join(base_dir, '..', 'my_data', success_strategy)
            self.data_folder_prefix = os.path.join(self.results_dir, self.strategy)
            self.orig_data_folder_names = os.listdir(self.data_folder_prefix)

            # Remove undesired folders
            for data_folder_name in self.orig_data_folder_names:
                if data_folder_name[:2] == '20':
                    self.data_folder_names.append(data_folder_name)

            self.numTrials=len(self.data_folder_names)

        #--------------------------------------------------------------------------
        # kFold Setup
        #--------------------------------------------------------------------------
        # Initialize k-fold data to a valid integer for the number of experiments

        # We will do so under a for loop that moves from 1 trial to total_num trials.
        # We want to test the accuracy of the approach for different number of trials.
#        self.results_dim=self.s_len+2 # The extra two numbers are time and the product of each of the states accuracy
#        self.accuracy_numTrials   =np.zeros( (self.numTrials-1,self.results_dim) ) #-1 trial because it is used for validation in "leave-one-out" cross validation.
#        for kfold in xrange(1,self.numTrials):
        kfold=self.numTrials

        # Crossfold training and testing generator
        self.kf=cross_validation.LeaveOneOut(self.numTrials)

        # Generate structure to hold train/validate indeces according to the number of folds
        # Given that we are doing leave one out, we use numTrials as the number of folds
        self.kf_list=[ [] for k in xrange(self.numTrials)]; self.foo=[]; self.temp=[]

        # Extract the indeces that belong to the training and validation lists
        for t,v in self.kf:
            self.temp.append(list(t)), self.temp.append(list(v))

        # Extract result onto a list
        for k in range(self.numTrials):
            for t in range(2):
                for elems in range(len(self.temp[2*k+t])): # Need to extract the ints from the interior list
                    self.foo.append(self.temp[2*k+t][elems])
                self.kf_list[k].append(self.foo)
                self.foo=[]

        self.train_len=len(self.kf_list[0][0])
        self.validate_len=len(self.kf_list[0][1])
        self.tvlen=[];self.tvlen.append(self.train_len);self.tvlen.append(self.validate_len)

        #--------------------------------------------------------------------------

        # Structures
        # kfold_list:               contains indeces for train/validate trials for each fold: kfold x train|validate x elems
        # lcss_trials_mat:          for n trials a nxn matrix that will capture the lcss for a given fold/level/state/axis. Useful for offline data compilation.
        # dictionary:               stores only the existing lcss outputs from the matrix to create an mx2 list.
        #                           the 2nd column will indicate the frequency with which ther term appeared, for fold/level/state/axis
        # similarity:               takes similarity values for all axes for a given state/level/fold.
        # results:                  keep results for 3 levels and 4 states 3x4.
        self.lcss_trials_mat=[ [ [ [ [] for aa in range(self.a_len)]
                                           for ss in range(self.s_len)]
                                               for ll in range(self.l_len)]
                                                   for kk in range(kfold)]

        self.similarity_permutation_matrix   =np.zeros( ( kfold,self.l_len,self.s_len,( self.s_len*self.a_len ),1) )
        self.sim_state_metric                =np.zeros( ( kfold,self.l_len,self.s_len,self.s_len,1) )

        self.similarity=[ [ [ [] for ss in range(self.s_len)]
                                      for ll in range(self.l_len)]
                                          for kk in range(kfold)]

        self.classification_vector=np.zeros( ( self.l_len, self.s_len, self.s_len,1) )
        self.avg_prob_vector      =np.zeros( ( self.l_len, self.s_len, self.s_len,1) )

        #------------------------------------------------------------------------------
        # Compute LCSS for training samples for each fold/level/state/axis
        # Separate training samples from one test trial (across folds).
        # 1. Compare training sequence 0 (in given kfold) with ith training sequence (i!=0)
        # 2. Compare that self.similarity with contents in self.lcss_trials_mat[k][l][s][a][0]
        #    - If structure already exists, increase frequency by 1.
        #    - Otherwise append.
        # 3. Testing
        #    - Compare validation string with list of common sequences.
        #    - Take match self.similarity distance and its frequency and use it to compute the self.similarity metric.
        #
        # Note: RCBHT strings will be encoded into an a-z alphabet for simpler string
        # comparison (single char) and in case we integrate multiple levels.
         #------------------------------------------------------------------------------
        #    if loadFromFileFlag:
        #        if os.path.isfile(os.path.join(exp_dir,trial_lcss_mat_pickle)):
        #            with open(os.path.join(exp_dir,trial_lcss_mat_pickle),'rb') as handle:
        #                self.lcss_trials_mat=pickle.load(handle)
        #---------------------------------------------------------------------------------------

        # lcss_trials_mat struc: if there is no data (not loaded from file) then create self.lcss_trials_mat
        self.computeLCSS_flag=1
#        if os.path.isfile( self.trial_lcss_mat_pickle_path ):
#            if not bool(self.lcss_trials_mat[0][0][0][0]):
#                if self.loadFromFileFlag:
#                    with open(self.trial_lcss_mat_pickle_path, 'rb') as handle:
#                        self.lcss_trials_mat = pickle.load(handle)
#                else:
#                    self.computeLCSS_flag=0
#            else:
#                self.computeLCSS_flag=0

        if self.computeLCSS_flag:
                for kk in range(kfold):
                    for ll,level in enumerate(self.levels):
                        for ss,state in enumerate(self.states):
                            for aa,axis in enumerate(self.axes):
                                self.currMax=0 # For each new axis, reset self.similarity length counter
                                for ii in xrange(self.train_len):
                                    if ii==0:                # on the first round of permutations only calculate the encoding once. after that, strSeq1 will be the same.
                                        # Perform self.similarity calculations across trials                #train index
                                        self.strSeq1 = self.allTrialLabels[state][ self.data_folder_names[ self.kf_list[kk][0][0] ]][level][axis]
                                        lbl.encodeRCBHTList(self.strSeq1,level)

                                    else: # don't evaluate the diagonal (same trial)
                                        self.strSeq2 = self.allTrialLabels[state][ self.data_folder_names[ self.kf_list[kk][0][ii] ]][level][axis]
                                        lbl.encodeRCBHTList(self.strSeq2,level)
                                        if DB_PRINT:
                                            print 'fold: ' + str(kk) + ' level: ' + str(ll) + ' state: ' + str(ss) + ' axes: ' + str(aa) + ' index ii: ' + str(ii)

                                        # Testing Counter Approach
                                        if ss==2:
                                            self.strSeq1=collections.Counter(self.strSeq1)
                                            self.strSeq2=collections.Counter(self.strSeq2)
                                            #self.strSeq1.update(self.strSeq2)
                                            avg_list=[self.strSeq1,self.strSeq2]
                                            avg_ctr= {k : sum( t[k] for t in avg_list)/2. for k in avg_list[0] }
                                            self.lcss_trials_mat[kk][ll][ss][aa]=collections.Counter(avg_ctr)
                                        else:
                                            # 1. Compare training sequence 0 with ith sequence
                                            self.commonString=lcs.LCS.longestCommonSequence(self.strSeq1,self.strSeq2)  # may sometimes have no overlap ""

                                            # 2. Insert common string/frequency into the 0th index
                                            # Check length of 0th index

                                            self.list_len=len(self.lcss_trials_mat[kk][ll][ss][aa])
                                            if self.list_len==0:
                                                self.temp=deepcopy([self.commonString,1])
                                                self.lcss_trials_mat[kk][ll][ss][aa].append(self.temp)
                                            else:
                                                for jj in xrange(self.list_len):
                                                    if self.commonString == self.lcss_trials_mat[kk][ll][ss][aa][jj][0]:
                                                        # A. Seen before
                                                        self.lcss_trials_mat[kk][ll][ss][aa][jj][1]+=1
                                                        break;
                                                # B. Not seen before
                                                else:
                                                    self.temp=[self.commonString,1]
                                                    self.lcss_trials_mat[kk][ll][ss][aa].append(self.temp)

                # Very time consuming structure to produce. Save it here.
                with open( self.trial_lcss_mat_pickle_path,'wb') as handle:
                    pickle.dump(self.lcss_trials_mat,handle,protocol=pickle.HIGHEST_PROTOCOL)

        #--------------------------------------------------------------------------
        # Similarity Permutation Matrix.
        #--------------------------------------------------------------------------
        # Compare validation test with dictionary in each level/state/axis. We expect high self.similarity for the same state.
        # For each fold/level there are m states each with m tests (each with n axis)
        # ... App_train \cap App_validate | App_train \cap Rot_validate ... | App_train \cap Mat_validate ... (mx(mxn)
        # ... Rot_train \cap App_validate | Rot_train \cap Rot_validate ... | Rot_train \cap Mat_validate
        # ...
        # ... Mat_train \cap App_validate | Mat_train \cap Rot_validate ... | Mat_train \cap Mat_validate
        #--------------------------------------------------------------------------
        for kk in range(kfold):
            for ll,level in enumerate(self.levels):
                for ss,state in enumerate(self.states):
                    for pp,aa in itertools.product( range(self.s_len),range(self.a_len) ):

                        if ss!=2:
                            self.dict_len=len(self.lcss_trials_mat[kk][ll][ss][aa])
                        else:
                            self.dict_len=1 # just 1 counter collection


                        if self.dict_len == 0:
                            print 'dictionary length is 0 for fold: ' + str(kk) + ' level: ' + str(ll) + ' state: ' + str(ss) + ' axes: ' + str(aa)

                        # Extract an lcss string between the single validate instance and each of the dictionary entries for a given axis
                        for ii in xrange(self.dict_len):

                            # Get validation string for current level/state/axis
                            if ii==0:
                                if DB_PRINT:
                                    print 'fold: ' + str(kk) + ' level: ' + str(ll) + ' state: ' + str(ss) + ' axes: ' + str(aa)

                                self.strSeq_validate = self.allTrialLabels[ self.states[pp] ][ self.data_folder_names[ self.kf_list[kk][1][0] ]][level][ self.axes[aa] ]
                                # Get encode validate string for permutated state as a single string                        #validate index
                                lbl.encodeRCBHTList(self.strSeq_validate,level)                              # still need to encode the validate instance
                                if ss!=2:
                                    self.strSeq_validate = ''.join(self.strSeq_validate)

                            # Get iith list as a single string for training data for current level/state/axis
                            if ss!=2:
                                self.strSeq_train     = ''.join(self.lcss_trials_mat[kk][ll][ss][aa][ii][0])
                            else:
                                self.strSeq_train = self.lcss_trials_mat[kk][ll][ss][aa] # copy the one existing counter object

                            if self.strSeq_train!=[]: # TODO check for leftover emtpy lists (need to fix)

                                ## Check for equality
                                if ss!=2:
                                    if self.strSeq_validate==self.strSeq_train:
                                        self.similarity=len(self.strSeq_train)
                                        self.strSeq_train_freq= self.lcss_trials_mat[kk][ll][ss][aa][ii][1]

                                        # Add metric for all axes, it gives a full states determination
                                        if self.similarity!=0:
                                            self.similarity_permutation_matrix[kk][ll][ss][pp*self.a_len+aa] += log10(self.strSeq_train_freq) + log10(self.similarity)
                                            #self.similarity_permutation_matrix[kk][ll][ss][pp*self.a_len+aa] += 1.0/self.strSeq_train_freq + pow(self.similarity,2)
    #                                    if self.similarity!=0 and log10(self.strSeq_train_freq)!=0:
    #                                        self.similarity_permutation_matrix[kk][ll][ss][pp*self.a_len+aa] += 1.0/log10(self.strSeq_train_freq) + pow(self.similarity,2) # TODO Need to add condition for zero difsion
                                else:
                                        self.strSeq_validate=collections.Counter(self.strSeq_validate)
                                        self.strSeq_train.subtract(self.strSeq_validate) # We want to reward those with error =0
                                        self.similarity=len(self.strSeq_train)
                                        # Add metric for all axes, it gives a full states determination
                                        if self.similarity!=0:
                                            for freq_index in self.strSeq_train:
                                                value=-1*abs(self.strSeq_train[freq_index])
                                                self.similarity_permutation_matrix[kk][ll][ss][pp*self.a_len+aa] +=  pow(2,value)


        #--------------------------------------------------------------------------
        # Save Permutation Metric
        #--------------------------------------------------------------------------
        with open(self.sim_permutation_matrix_pickle_path, 'wb') as handle:
            pickle.dump(self.sim_state_metric, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Save binary numpy to file (maybe more efficient than pickle and can be opened with np.loadtxt)
        self.fd = open(self.sim_permutation_matrix_pickle_path[:-7]+'.txt', "wb")
        np.save(self.fd,self.sim_state_metric)
        self.fd.close()


        #--------------------------------------------------------------------------
        # Similarity State Metric
        #--------------------------------------------------------------------------
        # Compute sum of similarities across axes for each fold/state/level (1x24)->(1x4)
        for kk in range(kfold):
            for ll in range(self.l_len):
                for ss in range(self.s_len):
                    for pp,aa in itertools.product( range(self.s_len),range(self.a_len) ):
                        self.sim_state_metric[kk][ll][ss][pp] += self.similarity_permutation_matrix[kk][ll][ss][pp*self.a_len+aa]


        #--------------------------------------------------------------------------
        # Save Similarity State Metric
        #--------------------------------------------------------------------------
        #pickle
        with open(self.sim_state_metric_pickle_path, 'wb') as handle:
            pickle.dump(self.sim_state_metric, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Save binary numpy to file (maybe more efficient than pickle and can be opened with np.loadtxt)
        self.fd = open(self.sim_state_metric_pickle_path[:-7]+'.txt', "wb")
        np.save(self.fd,self.sim_state_metric)
        self.fd.close()

        #--------------------------------------------------------------------------
        # A. Find average correct classifications
        #--------------------------------------------------------------------------
        #   Find max value across state similarities. If a given state had the largest
        #   self.similarity, then it is deemed to have been selected.
        #   Basially we count how many times each state was selected. At the end,
        #   the total number is divided by the number of folds to get an avg. over folds.
        #   Look at the diagonal to find out the correct classfication accuracy.
        # Approach:
        # Sum accross folds, divide by the number of folds.
        # The sum of the trace/4 tells you overall task accuracy
        for ll in range(self.l_len):
            for ss in range(self.s_len):
                for kk in range(kfold):
                    if np.sum(self.sim_state_metric[kk][ll][ss]!=0):
                        m=np.argmax(self.sim_state_metric[kk][ll][ss]) # which permutation was most likely for each state
                        self.classification_vector[ll][ss][m]+=1

                # Once we have iterated by all the folds we can do an element-wise division by fold number
                self.classification_vector[ll][ss]/=float(kfold)

        # Save Classfication Vector
        #pickle
        with open(self.classification_matrix_pickle_path, 'wb') as handle:
            pickle.dump(self.sim_state_metric, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Save binary numpy to file (maybe more efficient than pickle and can be opened with np.loadtxt)
        self.fd = open(self.classification_matrix_pickle_path[:-7]+'.txt', "wb")
        np.save(self.fd,self.sim_state_metric)
        self.fd.close()

        #--------------------------------------------------------------------------
        # B. Also find average self.similarity metric value and divided by the total. Compare with A.
        #--------------------------------------------------------------------------
        for l in range(self.l_len):
            for s in range(self.s_len):
                for k in range(kfold):
                    self.avg_prob_vector[l][s]+=self.sim_state_metric[k][l][s] # which permutation was most likely for each state

                # Divide matrix by number of folds
                self.avg_prob_vector[l][s]/=float(kfold)

                # Now we want to normalize the 4 values per given state
                self.fold_sum=np.sum(self.avg_prob_vector[l][s])
                if self.fold_sum != 0:
                    self.avg_prob_vector[l][s]=np.true_divide(self.avg_prob_vector[l][s],self.fold_sum)
                else:
                   self.avg_prob_vector[l][s]=np.zeros( (self.s_len,1) )

        # Save avg_prob_vector_pickle
        #pickle
        with open(self.classification_dir + '/' + self.avg_prob_vector_pickle, 'wb') as handle:
            pickle.dump(self.sim_state_metric, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Save binary numpy to file (maybe more efficient than pickle and can be opened with np.loadtxt)
        self.fd = open(os.path.join(self.classification_dir,self.avg_prob_vector_pickle[:-7]+'.txt'), "wb")
        np.save(self.fd,self.sim_state_metric)
        self.fd.close()

        # Results presentation (print,pickle,file)
        self.res=np.zeros( (self.l_len,self.s_len) )
        for l in range(self.l_len):
            for ctr in range(self.s_len):
                self.res[l]=self.classification_vector[l].diagonal()
                print 'For level: [',l+1,'/',self.l_len,'] and State [',ctr+1,'/',self.s_len,'] accuracy is: ',self.res[l][ctr],'\n'
            #print 'Task accuracy for level: ',l+1, ' is: ',self.res.prod(),'\n'
        #print str(kfold) + ' ' + str(self.res[0][0]) + ' ' + str(self.res[0][1]) + ' ' + str(self.res[0][2]) + ' ' + str(self.res[0][3]) + ' ' + str(self.res.prod())
        #self.accuracy_numTrials[kfold-1]=np.array( [kfold, self.res[0][0], self.res[0][1], self.res[0][2], self.res[0][3], self.res.prod()] )
        # Save results
        #picklec
        with open( self.lcss_results_pickle_path, 'wb') as handle:
            pickle.dump(self.res, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Save the results numpy matrix to file
        np.savetxt( self.lcss_results_pickle_path[:-7]+'.txt', self.res, '%0.2f')
        with file(self.lcss_results_pickle_path[:-7]+'.txt', 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(self.res.shape))
            for data_slice in self.res:
                np.savetxt(outfile, data_slice, fmt='%0.3f')
                outfile.write('# New slice\n')

        #----------------------------------------------------------------------
#        # Save the accumulated results matrix for kfolds from 1 to numTrials
#        np.savetxt(self.acuracy_numTrials_pickle_path, self.accuracy_numTrials, '%0.2f')
#
#        # Plot
#        import matplotlib.pyplot as plt
#        plt.plot(self.accuracy_numTrials[:,0], self.accuracy_numTrials[:,5])
#        plt.show()
#
#        return self.classification_vector
    #------------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def lcss_accumulated_accuracy_test(self,levels):
    #--------------------------------------------------------------------------

        # Compute states for desired level(s)
        if isinstance(levels,types.ListType):
            self.levels=levels
            self.l_len=len(levels)

        # Folders for storing results
        if self.strategy=='':                        # if this path does not exist
            self.strategy=raw_input('Please enter a name for the strategy: \n')
            self.get_allTrial_Labels=1

            # Assign the current strategy directory
            self.exp_dir = os.path.join(self.train_data_dir,self.strategy)

        #--------------------------------------------------------------------------
        # Directories
        #--------------------------------------------------------------------------

        # Create a classification folder and get the results folder
        self.directory='lcss'
        self.classification_dir=os.path.join(self.armSide_dir,self.directory)
        if not os.path.exists(self.classification_dir):
            os.makedirs(self.classification_dir)

        # Create a path for the lcss_trials_mat matrix
        self.trial_lcss_mat_pickle_path=os.path.join(self.classification_dir,self.armSide + self.trial_lcss_mat_pickle)

        # Create a path for the accumulated accuracy results matrix for a single arm
        self.acuracy_numTrials_pickle_path=os.path.join(self.classification_dir,self.armSide + self.acuracy_numTrials_pickle)

        #--------------------------------------------------------------------------
        # Load structures Must have been computed for all levels/states/axes
        #--------------------------------------------------------------------------

        # Make sure you have access to the allTrialsLabels structures. Otherwise can't go on. First check if available from memory, otherwise from file.
        if not bool(self.allTrialLabels): #TODO: Cannot retrieve from memory after creating in computeStatesLabels... why?
            if os.path.isfile( self.allTrialLabels_path ) and self.loadFromFileFlag:
                    with open( self.allTrialLabels_path, 'rb') as handle:
                        self.allTrialLabels = pickle.load(handle)
            else:
                raise Exception('allTrialsLabels is not available. Cannot classify with empty structure')


        # trial_lcss_mat structure. Otherwise can't go on. First check if available from memory, otherwise from file.
        #if not self.lcss_trials_mat in globals():
        if os.path.isfile( self.trial_lcss_mat_pickle_path ) and self.loadFromFileFlag:
                with open( self.trial_lcss_mat_pickle_path, 'rb') as handle:
                    self.lcss_trials_mat = pickle.load(handle)
        else:
            raise Exception('lcss_trials_mat is not available. Cannot classify with empty structure')

        ## TODO: when you fix the loading of allTrialsLabels from memory, this next section of code should go in an else statement
        # Get Folder names in the results directory
        self.data_folder_prefix = os.path.join(self.results_dir, self.strategy)

        # Store current data folder names in a list so we can iterate it through it
        if not bool(self.data_folder_names):
            # Get Folder names
            #data_folder_prefix = os.path.join(base_dir, '..', 'my_data', success_strategy)
            self.data_folder_prefix = os.path.join(self.results_dir, self.strategy)
            self.orig_data_folder_names = os.listdir(self.data_folder_prefix)

            # Remove undesired folders
            for data_folder_name in self.orig_data_folder_names:
                if data_folder_name[:2] == '20':
                    self.data_folder_names.append(data_folder_name)

            self.numTrials=len(self.data_folder_names)

        #--------------------------------------------------------------------------
        # kFold Setup
        #--------------------------------------------------------------------------
        # Initialize k-fold data to a valid integer for the number of experiments

        # lcss_trials_mat struc: if there is no data (not loaded from file) then create self.lcss_trials_mat
        self.computeLCSS_flag=1
        if os.path.isfile( self.trial_lcss_mat_pickle_path ):
            if not bool(self.lcss_trials_mat[0][0][0][0]):
                if self.loadFromFileFlag:
                    with open(self.trial_lcss_mat_pickle_path, 'rb') as handle:
                        self.lcss_trials_mat = pickle.load(handle)
                else:
                    self.computeLCSS_flag=0
            else:
                self.computeLCSS_flag=0

        # We will do so under a for loop that moves from 1 trial to total_num trials.
        # We want to test the accuracy of the approach for different number of trials.
        self.results_dim=self.s_len+2 # The extra two numbers are time and the product of each of the states accuracy
        self.accuracy_numTrials   =np.zeros( (self.numTrials-1,self.results_dim) ) #-1 trial because it is used for validation in "leave-one-out" cross validation.
        for kfold in xrange(1,self.numTrials):

            # Crossfold training and testing generator
            self.kf=cross_validation.LeaveOneOut(self.numTrials)

            # Generate structure to hold train/validate indeces according to the number of folds
            # Given that we are doing leave one out, we use numTrials as the number of folds
            self.kf_list=[ [] for k in xrange(self.numTrials)]; self.foo=[]; self.temp=[]

            # Extract the indeces that belong to the training and validation lists
            for t,v in self.kf:
                self.temp.append(list(t)), self.temp.append(list(v))

            # Extract result onto a list
            for k in range(self.numTrials):
                for t in range(2):
                    for elems in range(len(self.temp[2*k+t])): # Need to extract the ints from the interior list
                        self.foo.append(self.temp[2*k+t][elems])
                    self.kf_list[k].append(self.foo)
                    self.foo=[]

            self.train_len=len(self.kf_list[0][0])
            self.validate_len=len(self.kf_list[0][1])
            self.tvlen=[];self.tvlen.append(self.train_len);self.tvlen.append(self.validate_len)

            #--------------------------------------------------------------------------
            # Structures
            # kfold_list:               contains indeces for train/validate trials for each fold: kfold x train|validate x elems
            # lcss_trials_mat:          for n trials a nxn matrix that will capture the lcss for a given fold/level/state/axis. Useful for offline data compilation.
            # dictionary:               stores only the existing lcss outputs from the matrix to create an mx2 list.
            #                           the 2nd column will indicate the frequency with which ther term appeared, for fold/level/state/axis
            # similarity:               takes similarity values for all axes for a given state/level/fold.
            # results:                  keep results for 3 levels and 4 states 3x4.

            self.similarity_permutation_matrix   =np.zeros( ( kfold,self.l_len,self.s_len,( self.s_len*self.a_len ),1) )
            self.sim_state_metric                =np.zeros( ( kfold,self.l_len,self.s_len,self.s_len,1) )

            self.similarity=[ [ [ [] for ss in range(self.s_len)]
                                          for ll in range(self.l_len)]
                                              for kk in range(kfold)]

            self.classification_vector=np.zeros( ( self.l_len, self.s_len, self.s_len,1) )
            self.avg_prob_vector      =np.zeros( ( self.l_len, self.s_len, self.s_len,1) )

            #------------------------------------------------------------------------------
            # Loaded from file
            # Compute LCSS for training samples for each fold/level/state/axis

            #--------------------------------------------------------------------------
            # Similarity Permutation Matrix.
            #--------------------------------------------------------------------------
        for kk in range(kfold):
            for ll,level in enumerate(self.levels):
                for ss,state in enumerate(self.states):
                    for pp,aa in itertools.product( range(self.s_len),range(self.a_len) ):

                        if ss!=2:
                            self.dict_len=len(self.lcss_trials_mat[kk][ll][ss][aa])
                        else:
                            self.dict_len=1 # just 1 counter collection


                        if self.dict_len == 0:
                            print 'dictionary length is 0 for fold: ' + str(kk) + ' level: ' + str(ll) + ' state: ' + str(ss) + ' axes: ' + str(aa)

                        # Extract an lcss string between the single validate instance and each of the dictionary entries for a given axis
                        for ii in xrange(self.dict_len):

                            # Get validation string for current level/state/axis
                            if ii==0:
                                if DB_PRINT:
                                    print 'fold: ' + str(kk) + ' level: ' + str(ll) + ' state: ' + str(ss) + ' axes: ' + str(aa)

                                self.strSeq_validate = self.allTrialLabels[ self.states[pp] ][ self.data_folder_names[ self.kf_list[kk][1][0] ]][level][ self.axes[aa] ]
                                # Get encode validate string for permutated state as a single string                        #validate index
                                lbl.encodeRCBHTList(self.strSeq_validate,level)                              # still need to encode the validate instance
                                if ss!=2:
                                    self.strSeq_validate = ''.join(self.strSeq_validate)

                            # Get iith list as a single string for training data for current level/state/axis
                            if ss!=2:
                                self.strSeq_train     = ''.join(self.lcss_trials_mat[kk][ll][ss][aa][ii][0])
                            else:
                                self.strSeq_train = self.lcss_trials_mat[kk][ll][ss][aa] # copy the one existing counter object

                            if self.strSeq_train!=[]: # TODO check for leftover emtpy lists (need to fix)

                                ## Check for equality
                                if ss!=2:
                                    if self.strSeq_validate==self.strSeq_train:
                                        self.similarity=len(self.strSeq_train)
                                        self.strSeq_train_freq= self.lcss_trials_mat[kk][ll][ss][aa][ii][1]

                                        # Add metric for all axes, it gives a full states determination
                                        if self.similarity!=0:
                                            self.similarity_permutation_matrix[kk][ll][ss][pp*self.a_len+aa] += log10(self.strSeq_train_freq) + log10(self.similarity)
                                            #self.similarity_permutation_matrix[kk][ll][ss][pp*self.a_len+aa] += 1.0/self.strSeq_train_freq + pow(self.similarity,2)
    #                                    if self.similarity!=0 and log10(self.strSeq_train_freq)!=0:
    #                                        self.similarity_permutation_matrix[kk][ll][ss][pp*self.a_len+aa] += 1.0/log10(self.strSeq_train_freq) + pow(self.similarity,2) # TODO Need to add condition for zero difsion
                                else:
                                        self.strSeq_validate=collections.Counter(self.strSeq_validate)
                                        self.strSeq_train.subtract(self.strSeq_validate) # We want to reward those with error =0
                                        self.similarity=len(self.strSeq_train)
                                        # Add metric for all axes, it gives a full states determination
                                        if self.similarity!=0:
                                            for freq_index in self.strSeq_train:
                                                value=-1*abs(self.strSeq_train[freq_index])
                                                self.similarity_permutation_matrix[kk][ll][ss][pp*self.a_len+aa] +=  pow(2,value)

            #--------------------------------------------------------------------------
            # Similarity State Metric
            #--------------------------------------------------------------------------
            # Compute sum of similarities across axes for each fold/state/level (1x24)->(1x4)
            for kk in range(kfold):
                for ll in range(self.l_len):
                    for ss in range(self.s_len):
                        for pp,aa in itertools.product( range(self.s_len),range(self.a_len) ):
                            self.sim_state_metric[kk][ll][ss][pp] += self.similarity_permutation_matrix[kk][ll][ss][pp*self.a_len+aa]

            #--------------------------------------------------------------------------
            # Average correct classifications
            #--------------------------------------------------------------------------
            for ll in range(self.l_len):
                for ss in range(self.s_len):
                    for kk in range(kfold):
                        if np.sum(self.sim_state_metric[kk][ll][ss]!=0):
                            m=np.argmax(self.sim_state_metric[kk][ll][ss]) # which permutation was most likely for each state
                            self.classification_vector[ll][ss][m]+=1

                    # Once we have iterated by all the folds we can do an element-wise division by fold number
                    self.classification_vector[ll][ss]/=float(kfold)

            #--------------------------------------------------------------------------
            # Results presentation (print,pickle,file)
            #--------------------------------------------------------------------------
            self.res=np.zeros( (self.l_len,self.s_len) )
            for l in range(self.l_len):
                for ctr in range(self.s_len):
                    self.res[l]=self.classification_vector[l].diagonal()

                print 'Task accuracy for level: ',l+1, ' is: ',self.res.prod(),'\n'
            print str(kfold) + ' ' + str(self.res[0][0]) + ' ' + str(self.res[0][1]) + ' ' + str(self.res[0][2]) + ' ' + str(self.res[0][3]) + ' ' + str(self.res.prod())
            self.accuracy_numTrials[kfold-1]=np.array( [kfold, self.res[0][0], self.res[0][1], self.res[0][2], self.res[0][3], self.res.prod()] )

        #----------------------------------------------------------------------
        # Save the accumulated results matrix for kfolds from 1 to numTrials
        np.savetxt(self.acuracy_numTrials_pickle_path, self.accuracy_numTrials, '%0.2f')

        # Plot
        import matplotlib.pyplot as plt
        plt.plot(self.accuracy_numTrials[:,0], self.accuracy_numTrials[:,5])
        plt.show()

        return self.accuracy_numTrials
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if __name__== '__main__':
    try:
        # Define strategies to analyze
        success_strategy='SIM_HIRO_TWO_SA_SUCCESS'
        failure_strategy='SIM_HIRO_ONE_SA_ERROR_CHARAC_Prob'
        levels=['llbehavior'] # choose one or more from ['primitive', 'composite', 'llbehavior']
#------------------------------------------------------------------------------

        #lWAG=wrenchActionGrammar(success_strategy,failure_strategy,'left') # Instantiate the class for right arm
        #lWAG.computeStatesLabels()
        #l_classification_vec=lWAG.lcss(levels)
        #print l_classification_vec
        #l_acc_num_trials=lWAG.lcss_accumulated_accuracy_test(levels)

        rWAG=wrenchActionGrammar(success_strategy,failure_strategy,'right') # Instantiate the class for right arm
        #rWAG.computeStatesLabels()
        #r_classification_vec=rWAG.lcss(levels)
        r_acc_num_trials=rWAG.lcss_accumulated_accuracy_test(levels)
        print r_acc_num_trials

    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)

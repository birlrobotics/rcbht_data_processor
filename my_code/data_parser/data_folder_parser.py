'''This program will parse a folders data into different experiments, and for 
each experiment, three types of data: "Segments, Composites, and llBehaviors. 
Within each of these folders, it will extract data for each of the six axis: Fx,
Fy,Fz,Mx,My,Mz", and within each of these axis it will extract the corresponding 
label, start time, and end time.

TODO: If one folder throws an exception abandon it and continue to process'''

import os
import data_file_parser
import copy
#import sys

cur_dir = os.path.dirname(os.path.realpath(__file__))
base_dir = os.path.join(cur_dir, '..')
os.chdir(base_dir)

def parse_folder(folder_path, split_by_states = False, which_arm="right"):
    import re
    if which_arm == "right":
        file_pattern = '[a-zA-Z0-9]+_[a-zA-Z0-9]+.txt'
    else:
        file_pattern = "[a-zA-Z0-9]+_[a-zA-Z0-9]+_L.txt"

    file_name_checker = re.compile(file_pattern)

    dict_cooked_from_folder = {}

    dict_for_folder_path = {}
    dict_for_folder_path['primitive'] = os.path.join(folder_path, 'Segments')
    dict_for_folder_path['composite'] = os.path.join(folder_path, 'Composites')
    dict_for_folder_path['llbehavior'] = os.path.join(folder_path, 'llBehaviors')

    if not os.path.isdir(dict_for_folder_path['primitive']) or\
        not os.path.isdir(dict_for_folder_path['composite']) or\
        not os.path.isdir(dict_for_folder_path['llbehavior']):
        print 'bad folder with uncompleted 3-level data.'
        return None

    # This section does the data parsing by calling: data_file_parser.parse_file(file)
    for level in ['primitive', 'composite', 'llbehavior']:
        dict_cooked_from_folder[level] = {}
        for file_name in os.listdir(dict_for_folder_path[level]):
            check_result = file_name_checker.match(file_name)
            if check_result:
                pass
            else:
                print "%s is passed"%(file_name,)
                continue
 
            # Extract Fx,Fy,...,Mz. TODO: Separate right from left
            axis = file_name.split('_')[1][:2]
            file_path = os.path.join(dict_for_folder_path[level], file_name)
            file = open(file_path, 'r') # open the corresponding data file

            print file_path
            d = data_file_parser.parse_file(file) # create list of dic with entries for each bloc of data
            if d == None:
                return None
            dict_cooked_from_folder[level][axis] = d
            file.close()   
   
        # Check whether or not you want to split data by state Approach/Rot/Ins/Mating
    if not split_by_states:
        return dict_cooked_from_folder
    else:
        state_file_name = None
        for file_name in os.listdir(folder_path): # iterate through folder names until we find State
            if file_name[:7] == 'R_State':
                state_file_name = file_name
                break

        if state_file_name == None:
            print 'bad folder with no state.dat.'
            return None

               # Create the state vector
        state_file_path = os.path.join(folder_path, state_file_name)
        state_file = open(state_file_path, 'r')
        state_time_mark = map(float, state_file.read().strip().replace('\r', '').split('\n')) #Get time entries

        #the first one is always 0.0, we don't want it
        state_time_mark.pop(0)

        #some State.dat don't have the end time of the experiment
        #if in this case, we append it ourselves
        if len(state_time_mark) == 3:
            state_time_mark.append(dict_cooked_from_folder['llbehavior']['Fx'][-1]['t2End']) # Look for key Finish and ret value
        
        # print state_time_mark
        return splitor(dict_cooked_from_folder, state_time_mark)

''' Looks at the state transition times as well as labels starting and ending times and appropriately splits labels across states'''
def splitor(dict_cooked_from_folder, state_time_mark):
    from inc.states import states
    levels = ['primitive', 'composite', 'llbehavior']
    axes   = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']

    dict_with_states = {}
        
        # generate a dictionary that has each of the 4 states, each of the 3 levels, and each of the 6 axes.
    for state in states:
        dict_with_states[state] = {}
        for level in levels:
            dict_with_states[state][level] = {}
            for axis in axes:
                dict_with_states[state][level][axis] = []

    for level in levels:
        if level == 'primitive':
            end_time_key = 'Finish'
            start_time_key = 'Start'
        else:
            end_time_key = 't2End'
            start_time_key = 't1Start'

        for axis in axes:        
            # print '='*20
            # print state_time_mark
            # print map(lambda x:x[end_time_key], dict_cooked_from_folder[level][axis])

                     # Extract data for pertinent level and axis
            dicts_cooked_from_iterations = dict_cooked_from_folder[level][axis]
            iter_amount = len(dicts_cooked_from_iterations)

            state_idx = 0
            iter_idx = 0

                    # Separate data per state
            while iter_idx < iter_amount:
                iter_end_time = dicts_cooked_from_iterations[iter_idx][end_time_key]
                if iter_end_time < state_time_mark[state_idx]:
                    dict_with_states[states[state_idx]][level][axis].append(dicts_cooked_from_iterations[iter_idx])
                    iter_idx += 1
     
                              # Final Condition where last item = end state time
                elif iter_end_time == state_time_mark[state_idx]: #normally the last entry
                    dict_with_states[states[state_idx]][level][axis].append(dicts_cooked_from_iterations[iter_idx])
                    iter_idx += 1
                    state_idx += 1 # Move to the next state
     
                              # Situation where a label occupies more than one state. We resolve to make a copy of that label with a cut-off end-time 
                              # and use the same lable as the first one to the next state. 
                else:
                    tmp_dict = copy.deepcopy(dicts_cooked_from_iterations[iter_idx])
                             
                                        # Replace this state's end time to this copy of that label
                    tmp_dict[end_time_key] = state_time_mark[state_idx]
                    dicts_cooked_from_iterations[iter_idx][start_time_key] = state_time_mark[state_idx]

                    dict_with_states[states[state_idx]][level][axis].append(tmp_dict)
                    # Move to the next state                    
                    state_idx += 1 

            # for state in states:
            #     print '-'*20
            #     print dict_with_statesdicts_cooked_from_iterations[state][level][axis][0][start_time_key],
            #     print map(lambda x:x[end_time_key], dict_with_states[state][level][axis])

            # raw_input()

    return dict_with_states

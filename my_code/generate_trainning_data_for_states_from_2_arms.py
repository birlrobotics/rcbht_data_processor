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
    success_strategy='SIM_HIRO_TWO_SA_SUCCESS'

    # Set program paths
    results_dir="/Users/sklaw_mba/Desktop/ex/dr_juan_proj/workshop/data"
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = cur_dir
    os.chdir(base_dir)

    strategy=success_strategy    
    hlb_dir=strategy

    # Folder names
    data_folder_names=[]        # Filtered to only take relevant folders
    orig_data_folder_names=[]

    data_folder_prefix = os.path.join(results_dir, success_strategy)
    orig_data_folder_names = os.listdir(data_folder_prefix)
    
    # Remove undesired folders
    for data_folder_name in orig_data_folder_names:
        data_folder_names.append(data_folder_name)  


    for now_arm in ["right", "left"]:
        # Dictionary building blocks
        folder_dims={}
        dict_dims={}
        dict_all={}
        allTrialLabels={}

        files_for_states = {}
        
        # Open files for each of the states we will analyze

        if not os.path.exists(os.path.join(base_dir, '..', 'my_training_data', hlb_dir)):
            os.makedirs(os.path.join(base_dir, '..', 'my_training_data', hlb_dir))
        if not os.path.exists(os.path.join(base_dir, '..', 'my_training_data', hlb_dir, now_arm)):
            os.makedirs(os.path.join(base_dir, '..', 'my_training_data', hlb_dir, now_arm))
        for state in states:
            files_for_states[state] = open(os.path.join(base_dir,'..', 'my_training_data', hlb_dir, now_arm, "training_set_for_"+state), 'w')    

        # Create a dictionary structure for all trials, RCBHT levels, and axis.
        for data_folder_name in data_folder_names:
            data_folder_full_path = os.path.join(data_folder_prefix, data_folder_name)
            if DB_PRINT:
                print data_folder_full_path            
            
            dict_cooked_from_folder = data_folder_parser.parse_folder(data_folder_full_path, split_by_states=True, which_arm=now_arm)
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
            for data_folder_name in dict_all:
                for state in states:                
                    data_feature_extractor.extract_features(dict_all[data_folder_name][state], folder_dims)                                                                                          
                

            for state in states:
                allTrialLabels[state]={}
                for data_folder_name in dict_all:
                    allTrialLabels[state][data_folder_name]={}                        
                    allTrialLabels[state][data_folder_name]=deepcopy(dict_all[data_folder_name][state])                        
                        

            ## Create directories to keep both labels and images in file
            for state in states:
                # Write labels and images to file. 2 choices: individual iterations or all iterations per state.
                # label 1 indicates SUCCESS. Have a file and a place to put images
                output_features.output_sample_one_trial(files_for_states[state], 
                    str(states.index(state)), 
                    allTrialLabels[state], 
                    os.path.join(base_dir,'..', 'my_training_data', hlb_dir, now_arm, "img_of_"+state))

                files_for_states[state].close()

        else:
                raise Exception('dict_all from hlb states is not available')


    #combine data from left&right arms
    combined_folder = "combined"
    if not os.path.exists(os.path.join(base_dir, '..', 'my_training_data', hlb_dir, combined_folder)):
        os.makedirs(os.path.join(base_dir, '..', 'my_training_data', hlb_dir, combined_folder))
    for state in states:
        print state


        import numpy as np
        left_mat = np.genfromtxt(os.path.join(base_dir,'..', 'my_training_data', hlb_dir, "left", "training_set_for_"+state), dtype='string', delimiter=',')
        right_mat = np.genfromtxt(os.path.join(base_dir,'..', 'my_training_data', hlb_dir, "right", "training_set_for_"+state), dtype='string', delimiter=',')

        sample_amount_left = left_mat.shape[0]
        sample_amount_right = right_mat.shape[0]
        
        if sample_amount_left != sample_amount_right:
            raise Exception("sample amounts of state %s varied between left and right arm(%s vs %s)"%(state, str(sample_amount_left), str(sample_amount_right)))

        left_labels = left_mat[:, -1:]
        right_labels = right_mat[:, -1:]

        if not np.array_equal(left_labels, right_labels):
            raise Exception("labels in left arm should be identical to the right.")

        left_feature = left_mat[:, :-1]
        right_feature = right_mat[:, :-1]

        combined_mat_T = np.concatenate((left_feature.T, right_feature.T, left_labels.T), axis=0)
        combined_mat = combined_mat_T.T

        combined_file = open(os.path.join(base_dir,'..', 'my_training_data', hlb_dir, combined_folder, "training_set_for_"+state), 'w')    
        np.savetxt(combined_file, combined_mat, delimiter=',', fmt="%s")


        output_pixels = []
        combined_feature = combined_mat[:, :-1]
        img_height, img_width = combined_feature.shape
       
        for i in np.nditer(combined_feature):
            import inc.color_plate as color_plate
            kelly_colors_hex = color_plate.kelly_colors_hex

            output_pixels.append(kelly_colors_hex[int(np.asscalar(i))])


        from PIL import Image
        output_img = Image.new("RGB", (img_width, img_height)) # mode,(width,height)
        output_img.putdata(output_pixels)
        zoom = 1 
        output_img = output_img.resize((img_width*zoom, img_height*zoom))
        output_img.save(os.path.join(base_dir,'..', 'my_training_data', hlb_dir, combined_folder, "img_of_"+state+".png"))
        output_img.save(os.path.join(base_dir,'..', 'my_training_data', hlb_dir, combined_folder, "img_of_"+state+".eps"))

main();

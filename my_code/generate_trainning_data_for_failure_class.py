import os
import shutil

from copy import deepcopy

import util.output_features                     as output_features
import data_parser.data_folder_parser           as data_folder_parser
import feature_extractor.data_feature_extractor as data_feature_extractor

import traceback,sys#,code
from inc.config import states, levels, axes, failure_class_name_to_id

def classify_folder(folder_name):
    import re
    class_name = re.sub('[^\+\-xyr]', '', folder_name)
    return class_name

    
        

def main():
    ## Flags
    data_folder = "SIM_HIRO_ONE_SA_ERROR_CHARAC_prob"

    # Set program paths
    base_data_dir="/Users/sklaw_mba/Desktop/ex/dr_juan_proj/workshop/data"
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = cur_dir
    os.chdir(base_dir)

    # my training data
    result_directory ='my_training_data'

    data_folder_prefix = os.path.join(base_data_dir, data_folder)
        
    data_folder_names = os.listdir(data_folder_prefix)

    dict_all = {}
    map_from_folder_to_class = {}
    for data_folder_name in data_folder_names:
        data_folder_full_path = os.path.join(data_folder_prefix, data_folder_name)
        dict_cooked_from_folder = data_folder_parser.parse_folder(data_folder_full_path)
        if dict_cooked_from_folder == None:
            continue
        else:
            dict_all[data_folder_name]=dict_cooked_from_folder
            class_name = classify_folder(data_folder_name)
            map_from_folder_to_class[data_folder_name] = class_name

    folder_dims = {}

    for level in levels:
        folder_dims[level] = {}
        for axis in axes:
            folder_dims[level][axis]=0

    for data_folder_name in dict_all:
        for level in levels:
            for axis in axes: 
                temp = len(dict_all[data_folder_name][level][axis])
                if temp > folder_dims[level][axis]:
                    folder_dims[level][axis] = temp

    group_by_class = {}
    for data_folder_name in dict_all:
        class_name = map_from_folder_to_class[data_folder_name]
        if class_name not in group_by_class:
            group_by_class[class_name] = {} 
        group_by_class[class_name][data_folder_name] = dict_all[data_folder_name]
            
        
    os.makedirs(os.path.join(base_dir, '..', result_directory, data_folder))
    for class_name in group_by_class:
        for data_folder_name in group_by_class[class_name]:        
            print data_folder_name
            data_feature_extractor.extract_features(group_by_class[class_name][data_folder_name], folder_dims)
        signature = 'training_set_of_failure_class_%s' % class_name
        f = open(os.path.join(base_dir, '..', result_directory, data_folder, signature), 'w')
        output_features.output_sample_one_trial(f, str(failure_class_name_to_id[class_name]), group_by_class[class_name], os.path.join(base_dir,'..', result_directory, data_folder, "%s.png"%signature))


main();

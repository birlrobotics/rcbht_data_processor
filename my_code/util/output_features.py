import os
from PIL import Image

# Flags
initFlag=0
# Lists
import inc.config as config
levels= config.levels
axes  = config.axes

import inc.color_plate as color_plate
kelly_colors_hex = color_plate.kelly_colors_hex

# Prints one image for each iterataion containing data for FxyzMxyz (if we do this) we require that FxyzMxyz are all of the same length
# Same length data is produced in data_feature_extractor.p::slice_and_find_mode
def output_sample_one_trial(file, label, dict_all, img_path):
    output_pixels = []
    for data_folder_name, dict_cooked_from_folder in dict_all.iteritems():
        for level in dict_cooked_from_folder:
            for axis in dict_cooked_from_folder[level]:
                if dict_cooked_from_folder[level][axis] == None:
                    continue 
                # Extract all features for a level/axis and create a list. Use the list to write to file. 
                import inc.label_mapping as label_mapping
                list_of_features = [label_mapping.label_mapping_dict[level][i] for i in dict_cooked_from_folder[level][axis]]
                file.write(','.join(list_of_features)+',')
                output_pixels += [kelly_colors_hex[int(i)] for i in list_of_features]                
        file.write(label+'\n')

    img_height = len(dict_all)
    img_width = len(output_pixels)/img_height


    output_img = Image.new("RGB", (img_width, img_height)) # mode,(width,height)
    output_img.putdata(output_pixels)
    zoom = 1 
    output_img = output_img.resize((img_width*zoom, img_height*zoom))
    output_img.save(img_path+".png")
    output_img.save(img_path+".eps")


# Prints one image for each axis of all trials. 
def output_sample_all_trial(file, label, dict_cooked_allFiles,folder_names,numTrials,training_data_dir):

    # Globals    
    global initFlag
    
    # Image Directory
    if label=='s':
        img_dir='allTrials_success'
    elif label=='f':
        img_dir='allTrials_failure'
    else:
        img_dir=''
    
    # Initialization
    output_pixels    = []
    list_of_features = [] # for each level: axis: numLables by numTrials
    output_width = None
    
    lev=0
    ax =0
    
    for level in levels:
        initFlag=0  # Used to reset output_width, computed later when folder_name is available
        ax=0        # reset axis counter
        for axis in axes:
            
            # One file per axis (for each level, for all training examples)            
            axisFile=open( os.path.join(training_data_dir,level[:3]+axis+'.txt'), 'w' )                    
            
            for folder_name in folder_names:
               
                if dict_cooked_allFiles[folder_name][level][axis] == None:
                    break
                if not initFlag:                    
                    output_width = len(dict_cooked_allFiles[folder_name][level][axis]) 
                    initFlag=1                                  
                
                # Extract features for one axis and one level but for all folders
                import inc.label_mapping as label_mapping                                                
                list_of_features.append([label_mapping.label_mapping_dict[level][i] for i in dict_cooked_allFiles[folder_name][level][axis]])
                
            # Copy list_of_features to another strcutre where we will change to color
                
            # Once we have iterated through all the trials for one axis, write to file and display coded map.
            for trial in range(numTrials):                
                axisFile.write('\t'.join(list_of_features[trial])+'\t')  
                axisFile.write(label+'\n')
                
                # Create another list of the same dimensions as list_of_features  to store the color encoding for the labels                                        
                output_pixels += [kelly_colors_hex[int(i)] for i in list_of_features[trial]]                
            axisFile.close()                        

            # Create an image template for the corresponding number of slices and number of trials
            output_img = Image.new("RGB", (output_width, numTrials)) # mode,(width,height)
            
            # Insert colors int the structure
            output_img.putdata(output_pixels)
            zoom = 50 
            output_img = output_img.resize((output_width*zoom, 6*zoom))
            output_img.save(os.path.join(training_data_dir,img_dir, level[:3]+axis+'.png'))
            
            # Clearing Up            
            output_pixels=[]
            list_of_features=[]
            # Increment integer counter for axis
            ax+=1
            
        # Increment integer counter for level
        lev+=1
            

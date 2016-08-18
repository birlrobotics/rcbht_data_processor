import os
from PIL import Image

# Flags
initFlag=0
# Lists
levels=['primitive', 'composite', 'llbehavior']
axes  =['Mx', 'My', 'Mz', 'Fx', 'Fy', 'Fz']

def hex_to_rgb(RGBint):
    Blue =  RGBint & 255
    Green = (RGBint >> 8) & 255
    Red =   (RGBint >> 16) & 255
    return (Red, Green, Blue)

kelly_colors_hex = [
    0xFFB300, # Vivid Yellow
    0x803E75, # Strong Purple
    0xFF6800, # Vivid Orange
    0xA6BDD7, # Very Light Blue
    0xC10020, # Vivid Red
    0xCEA262, # Grayish Yellow
    0x817066, # Medium Gray

    # The following don't work well for people with defective color vision
    0x007D34, # Vivid Green
    0xF6768E, # Strong Purplish Pink
    0x00538A, # Strong Blue
    0xFF7A5C, # Strong Yellowish Pink
    0x53377A, # Strong Violet
    0xFF8E00, # Vivid Orange Yellow
    0xB32851, # Strong Purplish Red
    0xF4C800, # Vivid Greenish Yellow
    0x7F180D, # Strong Reddish Brown
    0x93AA00, # Vivid Yellowish Green
    0x593315, # Deep Yellowish Brown
    0xF13A13, # Vivid Reddish Orange
    0x232C16, # Dark Olive Green
    ]

kelly_colors_hex = map(hex_to_rgb, kelly_colors_hex)

img_name_count = 0

# Prints one image for each iterataion containing data for FxyzMxyz (if we do this) we require that FxyzMxyz are all of the same length
# Same length data is produced in data_feature_extractor.p::slice_and_find_mode
def output_sample_one_trial(file, label, dict_cooked_from_folder,img_folder):
    global img_name_count
    output_pixels = []
    output_width = None

    for level in dict_cooked_from_folder:
        axis_already_out = 0
        for axis in dict_cooked_from_folder[level]:
            if dict_cooked_from_folder[level][axis] == None:
                break
            output_width = len(dict_cooked_from_folder[level][axis])
            axis_already_out += 1
            
            # Extract all features for a level/axis and create a list. Use the list to write to file. 
            import inc.label_mapping as label_mapping
            list_of_features = [label_mapping.label_mapping_dict[level][i] for i in dict_cooked_from_folder[level][axis]]
            file.write('\t'.join(list_of_features)+'\t')
            
            # For each label provide an equivallent color.
            output_pixels += [kelly_colors_hex[int(i)] for i in list_of_features]
    file.write(label+'\n')

    # Create an image template for the corresponding number of slices and 6 axis
    output_img = Image.new("RGB", (output_width, 6)) # mode,(width,height)
    
    # Insert colors int the structure
    output_img.putdata(output_pixels)
    zoom = 30
    output_img = output_img.resize((output_width*zoom, 6*zoom))
    output_img.save(os.path.join(img_folder, str(img_name_count)+'.png'))
    img_name_count += 1
    file.close()
    
# Prints one image for each axis of all trials. 
def output_sample_all_trial(file, label, dict_cooked_allFiles,folder_names,numTrials,training_data_dir):

    # Globals    
    global img_name_count
    global initFlag
    
    # Image Directory
    if label=='1':
        img_dir='allTrials_success'
    else:
        img_dir='allTrials_failure'
    
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
            zoom = 30
            output_img = output_img.resize((output_width*zoom, 6*zoom))
            output_img.save(os.path.join(training_data_dir,img_dir, level[:3]+axis+'.png'))
            
            # Clearing Up            
            output_pixels=[]
            list_of_features=[]
            # Increment integer counter for axis
            ax+=1
            
        # Increment integer counter for level
        lev+=1
            
import os
import inc.color_plate as color_plate
from PIL import Image
kelly_colors_hex = color_plate.kelly_colors_hex

def output_one_streaming_exp(base_dir, class_name, experiment_name, array_of_streaming_dicts):
    print "protective drop"
    return
    print "have fun!" 
    output_pixels = []

    if not os.path.exists(os.path.join(base_dir, class_name)):
        os.makedirs(os.path.join(base_dir, class_name))
    os.makedirs(os.path.join(base_dir, class_name, experiment_name))

    output_file_for_streams = open(
        os.path.join(
            base_dir, 
            class_name, 
            experiment_name, 
            "output_file_for_streams.txt"
        ),
        "w"
    )

    for one_streaming_dict in array_of_streaming_dicts:
        for level in one_streaming_dict: 
            for axis in one_streaming_dict[level]:
                import inc.label_mapping as label_mapping
                list_of_features = [label_mapping.label_mapping_dict[level][i] for i in one_streaming_dict[level][axis]]
                output_file_for_streams.write(','.join(list_of_features))
                output_pixels += [kelly_colors_hex[int(i)] for i in list_of_features]
        output_file_for_streams.write('\n')


    img_height = len(array_of_streaming_dicts)
    img_width = len(output_pixels)/img_height
    
    output_img = Image.new("RGB", (img_width, img_height)) # mode,(width,height)
    output_img.putdata(output_pixels)
    zoom = 1 
    output_img = output_img.resize((img_width*zoom, img_height*zoom))
    output_img.save(
        os.path.join(
            base_dir, 
            class_name, 
            experiment_name, 
            "output_img__for_streams.png"
        )
    )

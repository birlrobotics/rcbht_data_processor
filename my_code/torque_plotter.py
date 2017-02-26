import os
import numpy as np

def torque_file_plotter(file_handle, output_signature, output_dir):
    rows = []
    for l in file_handle.readlines():
        items = l.strip().split('\t')
        if len(items) != 7:
            continue
        rows.append([float(i) for i in items])
        
    mat = np.array(rows)
    
    lines = mat.T.tolist()    
    
    output_file = open(os.path.join(output_dir, output_signature+".txt"), "w")
    output_file.close()

    x = lines[0]
    y1 = lines[1]
    y2 = lines[2]
    y3 = lines[3]
    y4 = lines[4]
    y5 = lines[5]
    y6 = lines[6]

    import matplotlib.pyplot as plt
    plt.plot(x, y1, label="y1")
    plt.plot(x, y2, label="y2")
    plt.plot(x, y3, label="y3")
    plt.plot(x, y4, label="y4")
    plt.plot(x, y5, label="y5")
    plt.plot(x, y6, label="y6")

    plt.ylabel('probability')
    plt.xlabel('time splice')
    plt.title(output_signature)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(os.path.join(output_dir, output_signature+".eps"), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, output_signature+".png"), bbox_inches='tight')
    plt.clf()
    

def main():

    success_strategy='SIM_HIRO_ONE_SA_SUCCESS'

    # Set program paths
    results_dir="/Users/sklaw_mba/Desktop/ex/dr_juan_proj/workshop/data"
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = cur_dir
    os.chdir(base_dir)

    # Folder names
    data_folder_names=[]        # Filtered to only take relevant folders
    orig_data_folder_names=[]

    data_folder_prefix = os.path.join(results_dir, success_strategy)
    orig_data_folder_names = os.listdir(data_folder_prefix)

    # Remove undesired folders
    for data_folder_name in orig_data_folder_names:
        data_folder_names.append(data_folder_name)  


    torque_file_name = "R_Torques.dat"


    output_dir = os.path.join(base_dir, '..', 'torque_plots', success_strategy) 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for data_folder_name in data_folder_names:
        data_folder_full_path = os.path.join(data_folder_prefix, data_folder_name)
        try:
            torque_file = open(os.path.join(data_folder_full_path, torque_file_name), "r")
        except IOError as e:
            continue
        print data_folder_name

        output_signature = data_folder_name
        torque_file_plotter(torque_file, output_signature, output_dir)
        
        torque_file.close()
        



if __name__ == "__main__":
    main()

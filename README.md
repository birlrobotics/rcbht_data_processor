# purpose of this repo
This repo serves as a "data cooker" that it can process/stream robot data so that the processed/streamed data can be used for machine learning/online classification.

# the data for which this repo is designed
* the data set of one-arm real robot data
  * data of failed experiments: https://github.com/rojas70/REAL_HIRO_ONE_SA_ERROR_CHARAC.git
  * data of successful experiments: https://github.com/rojas70/REAL_HIRO_ONE_SA_SUCCESS.git
* the data set of one-arm simulated robot data
  * data of failed experiments: https://github.com/rojas70/SIM_HIRO_ONE_SA_ERROR_CHARAC_Prob.git
  * data of successful experiments: https://github.com/rojas70/SIM_HIRO_ONE_SA_SUCCESS.git
* the data set of two-arm simulated robot data
  * data of successful experiments: https://github.com/rojas70/SIM_HIRO_TWO_SA_SUCCESS.git

# what can this repo do
1. generate training samples for machine learning.  
    The following scripts are designed to generate training samples:
    * ./my_code/generate_trainning_data_for_failure_class.py
    * ./my_code/generate_trainning_data_for_one_class_SVM_trained_with_solely_success_data_AND_corresponding_failure_data_for_verification.py
    * ./my_code/generate_trainning_data_for_states.py
    * ./my_code/generate_trainning_data_for_states_from_2_arms.py
    * ./my_code/generate_trainning_data_for_success_and_failure.py

    The training samples generated by these scripts are stored in:
    * ./my_training_data

1. stream experiments for online classification  
    The following scripts are desgiend to stream experiments:
    * ./my_code/generate_streaming_data.py

    You can tune the streaming interval in this script:
    * ./my_code/experiment_streamer/experiment_streamer.py

    The streamed experiments are stored in:
    * ./my_streaming_experiments

1. plot torque data in experiments  
    The following script is used to plot torque data:
    * ./my_code/torque_plotter.py

    The generated graphs are stored in:
    * ./torque_plots

# related repo
The repo in charge of training and online classification is https://github.com/birlrobotics/parse_rcbht_data_online_classification.git
~                                                                                                                                      

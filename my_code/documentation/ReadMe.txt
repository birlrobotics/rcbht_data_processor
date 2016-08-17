The code assumes the following file structure:

./my_code
./my_training_data
./my_data/
./my_data/data_003_SIM_HIRO_SA_Success-master
./my_data/data_004_SIM_HIRO_SA_ErrorCharac_Prob-master

    To run the code, cd into my_code, and run python generate_trainning_data.py
    I try to make the code as self-explanatory as possible...(though may not be the case) It uses data_parser to parse those data inside a data folder. Then it uses feature_extractor to down-sampling the data so that the data can be of same length. And finally it uses util.output_features to output a sample.
    You can see the training data I already cooked in the attached my_trainning_data.zip. It contains 6 files, and 6 folders. Each file contains many training samples of a same label. And the img folders contain visualized training sample(Fx, Fy.... Mz as y axis, sampling time as x axis)
    My github is blocked because I tried too many wrong password...I will update a more detailed README there

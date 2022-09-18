# AudioIMU_new

Github page to paper: **AudioIMU: Enhancing Inertial Sensing-Based Activity Recognition with Acoustic Models** (to appear at ACM ISWC 2022)

### IMU model

Derive DeepConvLSTM activity recognition models based on IMU inputs only: _lab_motion_train.py_

### Teacher models

Train and evaluate the teacher model 1 (audio inputs): _lab_audio_train.py_

Train and evaluate the teacher model 2 (audio + IMU inputs): _lab_multimodal_train.py_

### Student model guided by teacher outputs:

Train and evaluate the student models with 15 participants: _joint_trainfixlr_loso_individual.py_ 

If you want to do a parameter search for your own setting (especially if you experiment with a new model architecture or your own data), you can do something similar to script: _main_args_individuals.py_

If you just want to run inference for the participants' data based on our developed models, you can do something similar to script: _sample_inference.ipynb_

====

All the model architectures and FFT functions are wrapped up in _models.py_ 

Weights of our tested models can be accessed at: https://doi.org/10.18738/T8/S5RTFH. The data is of the name: rawAudioSegmentedData_window_10_hop_0.5_Test_NEW.pkl

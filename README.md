# AudioIMU_new

Github page to paper: **AudioIMU: Enhancing Inertial Sensing-Based Activity Recognition with Acoustic Models** (to appear on ACM ISWC 2022)

### Student model

Derive DeepConvLSTM activity recognition models based on IMU inputs only: _lab_motion_train.py_

### Teacher models

Train and evaluate teacher model 1 (audio inputs): _lab_audio_train.py_

Train and evaluate teacher model 2 (audio + IMU inputs): _lab_multimodal_train.py_

### Student model guided by teacher outputs:

Train and evaluate with 15 participants: _joint_trainfixlr_loso_individual.py_ 

If you want to do a parameter search for your own setting, you can do someting similar as in script: _main_args_individuals.py_

====

All the model architectures and FFT functions are wrappped up in _models.py_ 

Weights of the pre-trained Audio CNN (adopted from Kong et al., 2020, see our reference for details) and our student models can be accessed at: https://dataverse.tdl.org/dataset.xhtml?persistentId=doi%3A10.18738%2FT8%2FS5RTFH&version=DRAFT

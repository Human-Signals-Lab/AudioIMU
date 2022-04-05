# AudioIMU_new

Github page to paper: **AudioIMU: IMU-based activity recognition guided by audio inputs**

### Student model

Derive DeepConvLSTM activity recognition models based on IMU inputs only: _lab_motion_train.py_

### Teacher models

Train and evaluate teacher model 1 (audio inputs): _lab_audio_train.py_

Train and evaluate teacher model 2 (audio + IMU inputs): _lab_multimodal_train.py_

### Student model guided by teacher outputs:

Train and evaluate with 15 participants: _joint_trainfixlr_loso_individual.py_ (Script to run multiple jobs: _main_args_individuals.py_)

====

All the model architectures and FFT functions are wrappped up in _models.py_ 

Weights of the pre-trained Audio CNN (from Kong et al., ICASSP 2020) and our student models can be accessed at: https://dataverse.tdl.org/dataset.xhtml?persistentId=doi%3A10.18738%2FT8%2FS5RTFH&version=DRAFT

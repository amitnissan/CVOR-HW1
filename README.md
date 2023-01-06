# CVOR-HW1

Welcome! 
In this project, we use the state-of-the-art YOLOv7
for tool+hand detection of medical doctors while for
the purpose of evaluation of the suturing task.

### QUICK START
1. git clone https://github.com/amitnissan/CVOR-HW1.git
2. cd CVOR-HW1/src
3. python3 environment.py --input INPUT --output OUTPUT [--weights_path WEIGHTS_PATH] [--history_frames HISTORY_FRAMES]

### AVAILABLE PARAMETERS
1. input **(required)** - the input for the model to run on - **can be either a video or an image**
2. output **(required)** - the location in which the result will be available in
3. weights_path **(optional)** - model weights, see next section for more info, the default value is explained in the added report
4. history_frames **(optional)** - the number of frames to be considered for the video predictions smoothing 

### AVAILABLE MODELS
Can be found in resources/models folder in this repository.
1. init.pt - A simple version (default configuration) of YOLOv7 with fine-tuning on our data.
2. adam-100.pt - YOLOv7 with fine-tuning of our data and using Adam optimizer instead of SGD, trained for 100 epochs. 
3. adam-300.pt - YOLOv7 with fine-tuning of our data and using Adam optimizer instead of SGD, trained for 300 epochs.
4. best_iou50.pt - YOLOv7 with fine-tuning of our data and an iou_threshold value of 0.5.

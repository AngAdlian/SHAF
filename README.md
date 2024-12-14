# Spatiotemporal Hybrid Asymptotic Framework for Human Motion Prediction

## Recommand Dependencies
* Cuda 11.0
* Python 3.7
* Pytorch 1.8.1

## Dataset URL
* H3.6M: http://vision.imar.ro/human3.6m/description.php
* CMU_Mocap: http://mocap.cs.cmu.edu/
* 3DPW: https://virtualhumans.mpi-inf.mpg.de/3DPW/


## Code Execution:
We provide the code and best-performing models for human motion prediction on the H3.6M dataset (both long-term and short-term), the CMU Mocap dataset (both long-term and short-term), and the 3DPW dataset. The performance results of the models are presented in the result.txt files located in the five respective directories.

### H3.6M
Please place the human motion sequence data files for H3.6M, such as S1, S5, etc., in the h36m/dataset directory.

* Short-term motion prediction: 

Please run the following command in the H36Mshort directory:

Run the pre-trained model：python main_h36m.py --past_length 10 --future_length 10 --channel 72 --model_name ckpt_43 --test

Train：python main_h36m.py --past_length 10 --future_length 10 --channel 72

* Long-term motion prediction: 

Please run the following command in the H36Mlong directory:

Run the pre-trained model：python main_h36m.py --past_length 10 --future_length 25 --channel 108 --model_name ckpt_long_12 --test

Train：python main_h36m.py --past_length 10 --future_length 25 --channel 108

### CMU_Mocap
Please place the training and testing datasets in the cmu_mocap/train or cmu_mocap/test directory.

* Short-term motion prediction: 

Please run the following command in the CMUshort directory:

Run the pre-trained model：python main_cmu.py --past_length 10 --future_length 10 --channel 96 --model_name ckpt_65 --test

Train：python main_cmu.py --past_length 10 --future_length 10 --channel 96

* Long-term motion prediction: 

Please run the following command in the CMUlong directory:

Run the pre-trained model：python main_cmu.py --past_length 10 --future_length 25 --channel 108 --model_name ckpt_long_66 --test

Train：python main_cmu.py --past_length 10 --future_length 25 --channel 108

### 3DPW
Please place the train, test, and validation files in the 3DPW/3DPW/sequenceFiles directory.

Please run the following command in the 3DPW directory:

Run the pre-trained model：python main_3DPW.py --past_length 10 --future_length 30 --channel 96 --model_name ckpt_17 --test

Train：python main_3DPW.py --past_length 10 --future_length 30 --channel 96






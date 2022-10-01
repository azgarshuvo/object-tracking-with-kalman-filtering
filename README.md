## LSTM networks for human trajectory prediction in simulated crowded scenes with embedded group and obstacle information

Crowd analysis is an important topic in computer vision, with challenges ranging from crowd dynamics modelling, crowd segmentation, crowd activity classification, abnormal behaviour detection, to density estimation and crowd behaviour prediction tasks. In prediction, we observe the motion histories of the subject in the scene and exploit that information. Traditional methods use one-step forecasting, an example is the Kalman filter, but this did not exploit all the available information in the input scenes. Recurrent Neural Networks have emerged as a viable solution to the prediction problem. Previously, researchers focused on anticipating the subject’s future trajectory based on a precise motion history. We can pay more attention to the whole scene on a macro scale instead of the micro interpretation, extracting coarse salient features. People moving in a group can follow coherent motion patterns that can be exploited. This work proposes an extension of the work done in below

* Test the performance when the input video is a simulated crowd
* Choose the best epoch to produce the results and work on optimizing the LSTM architecture that’s used.
* Add another tensor to encode additional information related to the scene or the subjects
* Improve UCY performance by conditioning grouping based on whole dataset metrics of social information
* Considering multiple neighbourhoods, not just one

More recently, neural networks have been employed in the trajectory prediction task. Emerging deep generative models such as RNN, LSTM, and VAE (Variational Auto-encoders) solve the long-term prediction task directly.

## Simulated Dataset
In this work, a simulated crowd movement dataset being used and demonstrate that the performance is in the same region as that of a real video.

**Simulated Dataset Characteristics**
| Parameter | Value |
| --- | --- |
| Video Length | 27 seconds |
| Frame Width | 2560 |
| Frame Height | 1440 |
| Frame Rate | 24 frmaes per second |
| Total Number of Frame | 648 |

## Data Preprocessing
The data required for the pedestrian tracking task includes: Frame ID, Pedestrian ID, x coordinate, y coordinate, Group ID. To get this, it was necessary to extract the frames from the video, do a detection of the objects in the video, filter out the persons in the frames, and track the detected person across the frames. 

Detection
* After video frmaing, used the object detection library YOLO version 3 to identify objects in the frame
* [Object detection from simulated video frame](https://github.com/azgarshuvo/object-detection-yolo-openCV)

Tracking
* Used Kalman filter-based Simple Online Real-time Tracking (SORT) tracker to track the detected pedestrians across frames
* The image plane tracks are then projected to the floor plane in the following way

Transformation
* Calculated a Homography matrix by mapping points on the target floor plane to points in the image
* Used this information to calculate the inverse perspective projection from image to ground plane
* The Homography, together with an image Cartesian x and y coordinates are used to calculate the real-world floor coordinates

**Training Parametrs for Simulated Video**
| Parameter | Value |
| --- | --- |
| Batch Size | 4 |
| Sequence Length | 3 |
| Predicition Length | 3 |

## Complete Report
[View report in pdf format](https://github.com/azgarshuvo/object-tracking-with-kalman-filtering/blob/main/Report/Final_Report_Pedestrian_Trajectory_Prediction_Computer_Vision_2022_Roy_Shuvo%20-%20Copy.pdf)

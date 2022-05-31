from typing import List, Dict
import numpy as np
import cv2
import sort
from utils import *
import pandas as pd


class Object_Detector():
  """
  YOLO-v3 based object detector. This YOLO-v3 is pretrained on MS-COCO dataset.
  """
  def __init__(self):
    self.network = cv2.dnn.readNet("yolo/weights/yolov3.weights","yolo/cfg/yolov3.cfg") #  "yolo/cfg/coco.data"


  def detect(self, img):
    """
    Parameters
    ----------
    img: PIL Input Image
    category: category of the object to filter(should be one of the categories from MS-COCO dataset)
  
    Returns
    ---------- 
    detections: List of detections. Each detection is a tuple of form (object_name, score, bbox).
    """  
    
    classes = []
    with open("yolo/data/coco.names", "r") as f: # read the coco dataset
        classes = f.read().splitlines()  

    # capture the height and width of every frame that we are going to use it scale back to the original image size
    height, width, _ = img.shape  # Frame shape (1440, 2560, 3) 

    # creating a blob input (image, scaling, size of the image) Shape (1, 3, 416, 416)
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)

    # passing the blob into input function
    self.network.setInput(blob)

    # getting the output layers name ['yolo_82', 'yolo_94', 'yolo_106']
    output_layers_names = self.network.getUnconnectedOutLayersNames()

    # getting the output layer list len 3 [0.9875224 , 0.99220854, 0.18105118, ..., 0. ,0.,0.]], dtype=float32)] 
    layerOutputs = self.network.forward(output_layers_names) 

    boxes = []
    confidences = []
    class_ids = [] # represent the predicted classes

    detections = [] 

    for output in layerOutputs: # extract the information from each of the input
        # print(type(output), output.shape) <class 'numpy.ndarray'> (507, 85) <class 'numpy.ndarray'> (2028, 85) <class 'numpy.ndarray'> (8112, 85)
        
        for detection in output: # extract the information from each of the output
            det_data = []
            scores = detection[5:]
            class_id = np.argmax(scores) 
            confidence = scores[class_id]
            # print(detection[0],detection[1]) 0.8738878 0.5129194

            if confidence > 0.5:   # 0.5
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height) 

                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                class_name = str(classes[class_id])
                if class_name == 'person':
                    det_data.append(class_name)
                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id) 
                    det_data.append(confidence)
                    det_data.append([x,y,w,h])
                # first 4 coeffcient is the location of the bounding box and the 5th element is the box confidence
            detections.append(tuple(det_data)) 
        # (obj, score, [cx,cy,w,h])
    return detections
    

def detect_and_track(video_filename: str) -> Dict[str, List]:
  """
  Detection and Tracking function based on YOLO-v3 object detector and kalman filter based SORT tracker.
  Parameters
    ----------
    video_frames: path to the video file. Video would be a 4 dimesional np array of shape <N, C, H, W>.
    
    Returns
    ----------
    tracks: Dictionary of tracks where each key is the objectID and value is the list of the center of the
    object on the floor plane.
  """

  tracks = {}
  person_detector = Object_Detector()
  person_tracker = sort.Sort()
  # 1. Start reading the video file frame by frame
  cap = cv2.VideoCapture(video_filename) 

  frameID = 0
  while cap.isOpened():
    frameID += 1

    try:
      # 2. Iterate through each frame in the video
      ret, frame = cap.read() 

      img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
      
      # If video end reached
      if not ret:
          break 

      # 3. Get the detections from the object detector
      detections = person_detector.detect(img)

      # 4. Transform the detected points on floor plane from camera image plane
      detections_on_floor_plane = []
      for (obj, score, [cx,cy,w,h]) in detections:
          #convert coordinates cx,cy,w,h to x1,y1,x2,y2. Project them onto floor plane and
          # reorder the results to (bbox, score, object_name)
          x1, y1, x2, y2 = get_corner_coordinates([cx, cy, w, h])
          detection = [x1, y1, x2, y2, score] 

          # 5. Find association of the detected objects and add the objects into list of tracks Using SORT.
          if detection is not None:
              # 6. Update the tracks
              tracked_persons = person_tracker.update(detection)
              for x1, y1, x2, y2, personid in tracked_persons:
                  # 7. For each tracked object, get the center pixel on the image plane and add it to the object trajectory.
                  center_pos = (int((x1 + x2)/2), int(y1 + y2)/2)
                  tracks[personid] = tracks.get(personid, []) + [center_pos]  
    except Exception as e:
      print(frameID,e)
      


if __name__ == '__main__':
  video_path= 'Videos/cam3_004.mp4'
  print(detect_and_track(video_path))
  
  

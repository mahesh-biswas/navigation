import os
import cv2
import requests
import numpy as np
import tensorflow as tf
import sys
import time
import six.moves.urllib as urllib
import tarfile
import zipfile
import socket
from collections import defaultdict
from io import StringIO
import matplotlib.pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from voice_modules import tts_offline as tts
from captioning_module import ic



MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
# MODEL_NAME = 'faster_rcnn_resnet101_coco_2017_11_08'

MODEL_FILE = MODEL_NAME + '.tar.gz'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

sess = tf.Session(graph=detection_graph)
# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# Output tensors are the detection boxes, scores, and classes
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

num_detections = detection_graph.get_tensor_by_name('num_detections:0')

s = socket.socket()
s.bind(('',3425))
print("bound")
s.listen(5)
print("listening...")
c,addr = s.accept()
print("connected to {}".format(addr))

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255   
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

url = "http://192.168.43.48:8080/shot.jpg"
def caption(delay_period):
    i=0
    print(delay_period)
    while True:
        if i%delay_period == 0:
            caption = ic.prediction_for(ic.fit(url=url))
            tts.speak("i sense that, "+caption)
        else:
                pass
        i+=1
# video = cv2.VideoCapture(0)
def navigate(sense_cycle=20):
    try:
        i=0
        while True:
            img_resp = requests.get(url)
            img_arr = np.array(bytearray(img_resp.content),dtype=np.uint8)
            frame = cv2.imdecode(img_arr,-1)
            # _, frame = video.read()

            # if i == sense_cycle:
            if i == -1:
                caption = ic.prediction_for(ic.fit(url=url))
                tts.speak("i sense that, "+caption)
                i=0

            # objects = []
            # class_str = ""
            frame_width = frame.shape[0]
            frame_height = frame.shape[1]
            rows, cols = frame.shape[:2]
            print("r: ",rows,"  c: ",cols)

            left_boundary = [int(cols*0.40), int(rows*0.95)]
            left_boundary_top = [int(cols*0.40), int(rows*0.20)]
            right_boundary = [int(cols*0.60), int(rows*0.95)]
            right_boundary_top = [int(cols*0.60), int(rows*0.20)]

            bottom_left  = [int(cols*0.20), int(rows*0.95)]
            top_left     = [int(cols*0.20), int(rows*0.20)]
            bottom_right = [int(cols*0.80), int(rows*0.95)]
            top_right    = [int(cols*0.80), int(rows*0.20)]
            vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
            
            print("ltop: ",left_boundary_top,"rtop: ",right_boundary_top)
            print(" l: ",left_boundary," r: ",right_boundary)
            print(top_left,"   :   ",top_right)
            print(bottom_left,"   :   ",bottom_right)

            cv2.line(frame,tuple(bottom_left),tuple(bottom_right), (255, 0, 0), 5)
            cv2.line(frame,tuple(bottom_right),tuple(top_right), (255, 0, 0), 5)
            cv2.line(frame,tuple(top_left),tuple(bottom_left), (255, 0, 0), 5)
            cv2.line(frame,tuple(top_left),tuple(top_right), (255, 0, 0), 5)

            copied = np.copy(frame)
            interested=region_of_interest(copied,vertices)
            # cv2.imshow("roi",interested)

            frame_expanded = np.expand_dims(interested, axis=0)

            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded})

            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=6,
                min_score_thresh=0.68)

            print(frame_width,frame_height)
            print("*"*30)
            print(type(boxes))
            print(boxes[0][0])
            print(boxes[0][0][1]*frame_width,"  :  ",boxes[0][0][0]*frame_height)
            print(boxes[0][0][3]*frame_width,"  :  ",boxes[0][0][2]*frame_height)
            ymin = int((boxes[0][0][0]*frame_width))
            xmin = int((boxes[0][0][1]*frame_height))
            ymax = int((boxes[0][0][2]*frame_width))
            xmax = int((boxes[0][0][3]*frame_height))

            # Result = np.array(frame[ymin:ymax,xmin:xmax])

            ymin_str='y min  = %.2f '%(ymin)
            ymax_str='y max  = %.2f '%(ymax)
            xmin_str='x min  = %.2f '%(xmin)
            xmax_str='x max  = %.2f '%(xmax)


            # cv2.putText(frame,ymin_str, (50, 50),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            # cv2.putText(frame,ymax_str, (50, 70),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
            # cv2.putText(frame,xmin_str, (50, 90),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
            # cv2.putText(frame,xmax_str, (50, 110),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

            print(scores.max())

            print("left_boundary[0],right_boundary[0] :", left_boundary[0], right_boundary[0])

            print("left_boundary[1],right_boundary[1] :", left_boundary[1], right_boundary[1])

            print("xmin, xmax :", xmin, xmax)
            print("ymin, ymax :", ymin, ymax)
            
            text = ""

            c.send("0".encode())
            re = c.recv(1024).decode()

            if float(re) < 150:
                if xmin >= left_boundary[0]:
                    print("move LEFT - 1st !!!")
                    text = "Move LEFT"
                elif xmax <= right_boundary[0]:
                    print("move Right - 2nd !!!")
                    text = "Move Right"
                elif xmin <= left_boundary[0] and xmax >= right_boundary[0]:
                    print("STOPPPPPP !!!! - 3nd !!!")
                    text = "STOP!"
                else:
                    print("STOPPPPPP and no obsticle detected in frame but detected in ultrasonic.")
                    text = "STOP!"
            else:
                text = "Move Forward"
            cv2.putText(frame,text, (300, 100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)
            cv2.line(frame,tuple(left_boundary),tuple(left_boundary_top), (255, 0, 0), 5)
            cv2.line(frame,tuple(right_boundary),tuple(right_boundary_top), (255, 0, 0), 5)
            cv2.imshow("Video",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): # If we type on the keyboard:
                c.send("1".encode())
                c.close()
                tts.speak("Goodbye!!")
                break
            tts.speak(text)
            i+=1
        print("Done")
        cv2.destroyAllWindows()
    except Exception as e:
        print("Exception Catught: {}".format(e))
        pass

import threading

if __name__ == "__main__":
    
    # caption_thread = threading.Thread(target=caption, args=(3,)) 

    # caption_thread.start()
    navigate()
    # print("THREAD started...")


    # caption_thread.join()
    # print("COMPLETED")
    pass
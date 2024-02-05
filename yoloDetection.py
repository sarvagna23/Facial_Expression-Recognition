import numpy as np
#import argparse
import cv2 as cv
#import subprocess
#import time
#import os 

def detectObject(CNNnet, total_layer_names, image_height, image_width, image, name_colors, class_labels,  
            Boundingboxes=None, confidence_value=None, class_ids=None, ids=None, detect=True):
    
    if detect:
        blob_object = cv.dnn.blobFromImage(image,1/255.0,(416, 416),swapRB=True,crop=False)
        CNNnet.setInput(blob_object)  #Sets the input for the YOLO model using the blob.
        cnn_outs_layer = CNNnet.forward(total_layer_names) #Forward passes the blob through the model to obtain detections.
        Boundingboxes, confidence_value, class_ids = listBoundingBoxes(cnn_outs_layer, image_height, image_width, 0.5)
        ids = cv.dnn.NMSBoxes(Boundingboxes, confidence_value, 0.5, 0.3) #Calls the listBoundingBoxes function to extract bounding boxes, confidence values, and class IDs for the detected objects.
        #non-maximum suppression (NMS) to filter out overlapping boxes and keep only the most confident ones.
        if Boundingboxes is None or confidence_value is None or ids is None or class_ids is None:
           raise '[ERROR] unable to draw boxes.'
        image = labelsBoundingBoxes(image, Boundingboxes, confidence_value, class_ids, ids, name_colors, class_labels)
        #Calls the labelsBoundingBoxes function to draw bounding boxes
    return image, Boundingboxes, confidence_value, class_ids, ids
    #returns the modified image along with the bounding boxes, confidence values, class IDs, and IDs (NMS indices).
''' labels for bounding obj'''
def labelsBoundingBoxes(image, Boundingbox, conf_thr, classID, ids, color_names, predicted_labels):
    if len(ids) > 0:
        for i in ids.flatten():
            # draw boxes
            (x, y) = (Boundingbox[i][0], Boundingbox[i][1])
            (w, h) = (Boundingbox[i][2], Boundingbox[i][3])
            color = [int(c) for c in color_names[classID[i]]]
            cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(predicted_labels[classID[i]], conf_thr[i])
            cv.putText(image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            #For each detected object, it extracts the coordinates, dimensions, class color, and text label to display.
    return image

''' Bounding Box Listing: listBoundingBoxes

    Parses the detection results and extracts bounding box information, confidence values, and class IDs based on a confidence threshold.'''

def listBoundingBoxes(image, image_height, image_width, threshold_conf):#It takes the model's output, image height and width, and a confidence threshold as input.
    box_array = []
    confidence_array = []
    class_ids_array = []
    # iterates through the output and for each detected object:
    for img in image:
        for obj_detection in img:
            detection_scores = obj_detection[5:]
            class_id = np.argmax(detection_scores)
            #Calculates the coordinates and dimensions of the bounding box relative to the image size.
            confidence_value = detection_scores[class_id]
            #Filters out objects with confidence values below the specified threshold.
            if confidence_value > threshold_conf:
                Boundbox = obj_detection[0:4] * np.array([image_width, image_height, image_width, image_height])
                center_X, center_Y, box_width, box_height = Boundbox.astype('int')

                xx = int(center_X - (box_width / 2))
                yy = int(center_Y - (box_height / 2))
                #Stores bounding box coordinates, confidence values, and class IDs in separate lists.
                box_array.append([xx, yy, int(box_width), int(box_height)])
                confidence_array.append(float(confidence_value))
                class_ids_array.append(class_id)

    return box_array, confidence_array, class_ids_array
#The function returns the lists of bounding boxes, confidence values, and class IDs.
def displayImage(image):
    cv.imshow("Final Image", image)
    cv.waitKey(0)




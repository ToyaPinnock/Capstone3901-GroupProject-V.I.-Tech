from flask import Flask, render_template, Response
from camera import Video
import cv2
import numpy as np
from distancefinder import distance_finder, focal_length_finder

app=Flask(__name__)
camera=cv2.VideoCapture(0)

def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            thres = 0.70 # Threshold to detect object
            #Distance constants 
            KNOWN_DISTANCE = 45 #INCHES
            nms_threshold = 0.2

            classNames= ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush']
            classDistances=[51, 29, 23, 36, 27, 38, 31, 21, 33, 46, 54, 25, 59, 52, 26, 44, 49, 53, 56, 32, 34, 35, 20, 47, 24, 43, 58, 39, 55, 48, 41, 40, 30, 28, 37, 45, 60, 42, 57, 22, 50, 24, 45, 53, 49, 47, 54, 59, 41, 52, 27, 34, 46, 43, 21, 22, 56, 36, 23, 48, 55, 33, 42, 58, 51, 25, 38, 44, 50, 57, 30, 28, 35, 29, 37, 20, 39, 26, 60, 40, 60, 34, 31, 59, 35, 28, 39, 53, 42, 43, 55]
            class_dict=dict(zip(classNames, classDistances))

            configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
            weightsPath = 'frozen_inference_graph.pb'

            net = cv2.dnn_DetectionModel(weightsPath,configPath)
            net.setInputSize(320,320)
            net.setInputScale(1.0/ 127.5)
            net.setInputMean((127.5, 127.5, 127.5))
            net.setInputSwapRB(True)
            classIds, confs, bbox = net.detect(frame,confThreshold=thres)
            bbox = list(bbox)
            confs = list(np.array(confs).reshape(1,-1)[0])
            confs = list(map(float,confs))
            idxs = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
            if len(idxs) > 0:   
            # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    
                        (x, y) = (bbox[i][0], bbox[i][1])
                        (w, h) = (bbox[i][2], bbox[i][3])
                        x1,y1=x+w, y+h
                        Focal_length_found=focal_length_finder(KNOWN_DISTANCE,class_dict[classNames[classIds[i]-1]], 20)
                        distance = distance_finder(Focal_length_found, class_dict[classNames[classIds[i]-1]],w)
                        # draw a bounding box rectangle and label on the frame
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
                        cv2.putText(frame, classNames[classIds[i]-1].upper(), (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 2)
                        cv2.putText(frame,str(round(confs[i]*100,2)), (x + 100, y-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 2)
                        cv2.putText(frame, f'Dis: {round(distance,2)} inch', (x+5,y+13), cv2.FONT_HERSHEY_COMPLEX, 0.48, (0,255,0), 2)
                        ret,buffer=cv2.imencode('.jpg',frame)
                frame=buffer.tobytes()
                yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/homemode')
def cashmode():
    return render_template('homemode.html')
@app.route('/cashmode')
def homemode():
    return render_template('cashmode.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)
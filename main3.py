import detect
import tflite_runtime.interpreter as tflite
import time
import alter
from PIL import Image
from PIL import ImageDraw
import cv2
import numpy as np
import time
import os
import logging
logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')


from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

@app.route("/")
def index() :
    return render_template('index.html')


# .tflite interpreter
interpreter = tflite.Interpreter(
    os.path.join(os.getcwd(), "ssd_mobilenet_v2_fpnlite.tflite"),
    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
    )
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

WIDTH = 640
HEIGHT = 480 
type_list = ['got mask','no mask', 'wear incorrectly']
# Draws the bounding box and label for each object.
def draw_objects(image, objs):
    for obj in objs:
        bbox = obj.bbox
        
        cv2.rectangle(image,(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0, 255, 0),2)

        bbox_point_w = bbox.xmin + ((bbox.xmax-bbox.xmin) // 2)
        bbox_point_h = bbox.ymin + ((bbox.ymax-bbox.ymin) // 2) 
        logging.info(msg=obj.score)
        cv2.circle(image, (bbox_point_w, bbox.ymax-bbox.ymin), 5, (0,0,255),-1)
        cv2.putText(image, text='%d%%' % (obj.score*100), org=(bbox.xmin, bbox.ymin), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

        
def draw_and_show(box,classes,scores,num,frame):
    for i in range(int(num[0])):
        # print(scores[0][int(i)-1])
        if scores[0][i] > 0.8:
            y,x,bottom,right = box[0][i]
            x,right = int(x*WIDTH),int(right*WIDTH)
            y,bottom = int(y*HEIGHT),int(bottom*HEIGHT)
            class_type=type_list[int(classes[0][i])]
            label_size = cv2.getTextSize(class_type,cv2.FONT_HERSHEY_DUPLEX,0.5,1)
            cv2.rectangle(frame, (x, y), (right, bottom), (0,255,0), thickness=2)
            cv2.rectangle(frame,(x,y-18),(x+label_size[0][0],y),(0,255,0),thickness=-1)
            cv2.putText(frame,class_type,(x,y-5),cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return frame
        
def main():
    cap = cv2.VideoCapture(0)
    cap.set(3,WIDTH)
    cap.set(4,HEIGHT)
    while True:
        ret, image = cap.read()
        
        ####
        img_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_ = cv2.resize((img_*2/255)-1,(input_details[0]['shape'][1],input_details[0]['shape'][1]))
        img_ = img_[np.newaxis,:,:,:].astype('float32')
        output_frame = img_
        
        interpreter.set_tensor(input_details[0]['index'], output_frame)
        interpreter.invoke()
        boxes = interpreter.get_tensor(output_details[0]['index'])
        classes = interpreter.get_tensor(output_details[1]['index'])
        scores = interpreter.get_tensor(output_details[2]['index'])
        num = interpreter.get_tensor(output_details[3]['index'])
        output = [boxes,classes,scores,num]
        

        
        
        #### 
	      # image reshape
        #image = cv2.resize(image, dsize=(320, 320), interpolation=cv2.INTER_AREA)
	      # image BGR to RGB
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

       	#tensor = detect.input_tensor(interpreter=interpreter)[:, :] = image.copy() 
        #tensor.fill(0)  # padding        
        #interpreter.invoke()  # start
        
        #objs = detect.get_output(interpreter, 0.5, (1.0, 1.0))
        
        """if len(image):
            draw_objects(image, objs)"""
        
        
        frames = draw_and_show(*output, image)

        imgencode = cv2.imencode('.jpg', image)[1]
        stringData = imgencode.tostring()
        yield (b'--frame\r\n'
        b'Content-Type: text/plain\r\n\r\n' + stringData + b'\r\n')
        #image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        cv2.imshow('face detector', frames)

        k = cv2.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit # ESC exit
            break
    del(cap)


@app.route('/calc')
def calc() :
    alter.alert("hi")
    return Response(main(),
            mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    #main()
    app.run(host="localhost", debug=True, threaded=True)


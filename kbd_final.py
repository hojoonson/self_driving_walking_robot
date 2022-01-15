see__author__ = 'zhengwang'

import threading
import SocketServer
#import serial
import cv2
import numpy as np
import math
import socket
import calibration_kbd
import picam_calibration
# distance data measured by ultrasonic sensor
sensor_data = " "
mode = "STAY"
log = ""
#mtx,dist = picam_calibration.calibrationKBD()

#K=np.array([[138.23811939134242, 0.0, 169.2915905458952], [0.0, 138.21765527559086, 89.96752677866199], [0.0, 0.0, 1.0]])
#D=np.array([[-0.015822807276998765], [-0.05761685284693603], [0.06656241367580487], [-0.031006944095777388]])
K,D = calibration_kbd.calibrate_KBD()
#K,D = picam_calibration.calibrate_KBD()
K=K
D=D
m_or_a=""
print("K : ")
print(K)
print("D : ")
print(D)
def Manual_or_Auto():
    print("Select Manual or Auto")
    result = raw_input("Manual : m/M , Auto : a/A  >>>>")
    print(result)
    if result=="m" or result=="M":
        return "m"
    if result=="a" or result=="A":
        return "a"

m_or_a=Manual_or_Auto()

class DistanceToCamera(object):

    def __init__(self):
        # camera params
        self.alpha = 8.0 * math.pi / 180
        self.v0 = 119.865631204
        self.ay = 332.262498472

    def calculate(self, v, h, x_shift, image):
        # compute and return the distance from the target point to the camera
        d = h / math.tan(self.alpha + math.atan((v - self.v0) / self.ay))
        if d > 0:
            cv2.putText(image, "%.1fcm" % d,
                (image.shape[1] - x_shift, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return d


class ObjectDetection(object):
    def __init__(self):
        self.red_light = False
        self.green_light = False
        self.yellow_light = False

    def detectRED(self, cascade_classifier, gray_image, image):
        # y camera coordinate of the target point 'P'
        v = 0
        # detection
        cascade_obj = cascade_classifier.detectMultiScale(
            gray_image,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(16,16)
            #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        # draw a rectangle around the objects
        for (x_pos, y_pos, width, height) in cascade_obj:
            # image is (y,x)
            # format is BGR

            roi = image[y_pos:y_pos + height, x_pos:x_pos + width]
            roi=list(roi)
            for h in range(height):
                roi[h]=list(roi[h])
                for w in range(width):
                    roi[h][w]=list(roi[h][w])
            maxpixel=max(max(roi))

            roi = gray_image[y_pos:y_pos + height, x_pos:x_pos + width]
            roi = list(roi)
            for h in range(height):
                roi[h] = list(roi[h])
            maxheight = roi.index(max(roi))
            if max(maxpixel)==maxpixel[2]:
            #if maxheight < height/2:
                print(maxheight,height,height/2)
                print("RED")
                cv2.rectangle(image, (x_pos, y_pos), (x_pos+width, y_pos+height), (255, 255, 255), 2)
                v = y_pos + height - 5
                cv2.putText(image, 'RED light', (x_pos, y_pos-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return v


    def detectGREEN(self, cascade_classifier, gray_image, image):
    
        # y camera coordinate of the target point 'P'
        v = 0
        # detection
        cascade_obj = cascade_classifier.detectMultiScale(
            gray_image,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(16,16)
            #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        # draw a rectangle around the objects
        for (x_pos, y_pos, width, height) in cascade_obj:
            # image is (y,x)
            # format is BGR

            roi = image[y_pos:y_pos + height, x_pos:x_pos + width]
            roi=list(roi)
            for h in range(height):
                roi[h]=list(roi[h])
                for w in range(width):
                    roi[h][w]=list(roi[h][w])
            maxpixel=max(max(roi))

            roi = gray_image[y_pos:y_pos + height, x_pos:x_pos + width]
            roi = list(roi)
            for h in range(height):
                roi[h] = list(roi[h])
            maxheight = roi.index(max(roi))
            if max(maxpixel)==maxpixel[1]:
            #print(maxheight-height/2)
            #if maxheight > height/2:
                print(maxheight,height,height/2)
                print("GREEN")
                cv2.rectangle(image, (x_pos, y_pos), (x_pos+width, y_pos+height), (255, 255, 255), 2)
                v = y_pos + height - 5
                cv2.putText(image, 'GREEN light', (x_pos, y_pos-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return v

    def detectSTOP(self, cascade_classifier, gray_image, image):
        # y camera coordinate of the target point 'P'
        # detection
        mode=" "
        cascade_obj = cascade_classifier.detectMultiScale(
            gray_image,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(16,16),
            #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        for (x_pos, y_pos, width, height) in cascade_obj:
            # draw a rectangle around the objects
            #print(width,height)
            if(width>=50):
                cv2.rectangle(image, (x_pos, y_pos), (x_pos+width, y_pos+height), (255, 255, 255), 2)
                cv2.putText(image, 'STOP', (x_pos, y_pos-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                mode="STOP"

        return mode
class SensorDataHandler(SocketServer.BaseRequestHandler):

    data = " "

    def handle(self):
        global sensor_data
        global mode
        global log
        try:
            while self.data:
                self.data = self.request.recv(1024)
                self.request.sendall(mode)
                sensor_data = round(float(self.data.decode()), 1)
                #print ("Client IP : " + str(self.client_address[0]) + " Client PORT : " + str(self.client_address[1]) + " SENT")
                print ("Distance : "+str(sensor_data)+" cm"+"    " + mode+"   "+ log)
        finally:
            print "Connection closed on thread 2"

def command(array):
    no_restriction = 200 # max : 236
    last = array.size-1
    left_array = last // 3
    right_array = last - left_array
    out_line=""
    # stop
    forward_lane = 1
    for i in range(left_array//2, last-left_array//2):
        forward_lane = forward_lane * (array[i] < 100)

    left_lane = 0
    for i in range(0,left_array):
        if array[i] < no_restriction:
            left_lane = left_lane + (240-array[i]) / (left_array+1-i)

    right_lane = 0
    for i in range(right_array,last):
        if array[i] < no_restriction:
            right_lane = right_lane + (240-array[i]) / (i+1-right_array)


    if forward_lane == 1:
        if left_lane - right_lane > 200:
            #motor_mode(-1,0)
            out_line = 'Backward Left'
        else:
            #motor_mode(0,-1)
            out_line = 'Backward Right'
    elif left_lane < 300 and right_lane < 300:
        #motor_mode(1,1)
        out_line = 'Go Straight'
    elif left_lane > right_lane:
        #motor_mode(1,0)
        out_line = 'Foward Right'
    elif left_lane < right_lane:
        #motor_mode(0,1)
        out_line = 'Foward Left'

    return out_line

class VideoStreamHandler(SocketServer.StreamRequestHandler):

    # h1: stop sign
    h1 = 15.5 - 10  # cm
    # h2: traffic light
    h2 = 15.5 - 10


    obj_detection = ObjectDetection()
    #rc_car = RCControl()

    # cascade classifiers
    #red_cascade = cv2.CascadeClassifier('cascade_xml/RED.xml')
    #green_cascade = cv2.CascadeClassifier('cascade_xml/GREEN.xml')
    stop_cascade=cv2.CascadeClassifier('cascade_xml/STOP.xml')
    d_to_camera = DistanceToCamera()
    d_stop_sign = 25
    d_light = 25

    stop_start = 0              # start time when stop at the stop sign
    stop_finish = 0
    stop_time = 0
    drive_time_after_stop = 0

    def handle(self):

        global sensor_data
        global mode
        global log
        global K
        global D
        global m_or_a
        stream_bytes = ' '

        # stream video frames one by one
        try:
            while True:
                stream_bytes += self.rfile.read(1024)
                first = stream_bytes.find('\xff\xd8')
                last = stream_bytes.find('\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last+2]
                    stream_bytes = stream_bytes[last+2:]
                    gray = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),  cv2.IMREAD_GRAYSCALE)
                    image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

                    image_h, image_w,channels = image.shape
                    #image = image[image_h/2:,0:image_w]
                    #gray = gray[image_h/2:,0:image_w]
                    #image = cv2.resize(image, (0,0), fx=2, fy=2)
                    #gray = cv2.resize(image,(0,0),fx=2,fy=2)
                    #image_h, image_w,channels = image.shape


                    #map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
                    #undistorted_img = cv2.remap(image,map1,map2,cv2.INTER_LINEAR)

                    # lower half of the image
                    #half_gray = gray[120:240, :]

                    def select_white(image, white):
                        lower = np.uint8([white,white,white])
                        upper = np.uint8([255,255,255])
                        white_mask = cv2.inRange(image, lower, upper)
                        return white_mask


                    mask_image = select_white(image, 160)
                    mask_alt = cv2.cvtColor( mask_image, cv2.COLOR_GRAY2RGB)
                    w = .6
                    out_image = cv2.addWeighted(mask_alt, w, image, 1-w, 0)
                    wh_distance = np.zeros(int(image_w)/10)

                    for i in range(0,image_w-5,10):
                        for j in range(image_h-1, 0, -5):
                            if mask_image[j,i] > 220 or j==4:
                                wh_distance[i//10] = 220-j
                                break

                    self.obj_detection.detectSTOP(self.stop_cascade, gray, image)
                    # object detection
                    #v_param1 = self.obj_detection.detect(self.stop_cascade, gray, image)
                    #v_param2 = self.obj_detection.detect(self.light_cascade, gray, image)
                    #self.obj_detection.detectRED(self.red_cascade, gray, undistorted_img)
                    #sself.obj_detection.detectGREEN(self.green_cascade, gray, undistorted_img)
                    
                    #self.obj_detection.detectRED(self.red_cascade, gray, image)
                    #self.obj_detection.detectGREEN(self.green_cascade, gray, image)
                    # distance measurement
                    """
                    if v_param1 > 0 or v_param2 > 0:
                        d1 = self.d_to_camera.calculate(v_param1, self.h1, 300, image)
                        d2 = self.d_to_camera.calculate(v_param2, self.h2, 100, image)
                        self.d_stop_sign = d1
                        self.d_light = d2
                    """

                    #cv2.imshow("undistort",undistorted_img)
                    cv2.imshow("white", mask_image)
                    cv2.imshow('image', image)
                    cv2.imshow('grayimage', gray)


                    # reshape image
                    #image_array = half_gray.reshape(1, 38400).astype(np.float32)

                    # neural network makes prediction
                    #prediction = self.model.predict(image_array)
                    """Auto Mode"""
                    # stop conditions
                    log=""
                    
                    if m_or_a=="a":
                        if sensor_data is not None and sensor_data < 15 :
                            #print("Stop, obstacle in front")
                            mode = "STOP"
                            log = "Obstacle infront"
                        elif sensor_data is not None and self.obj_detection.detectSTOP(self.stop_cascade, gray, image)=="STOP":
                            mode = "STOP"
                            log = "Stopsign infront"
                            #self.rc_car.stop()
                        else:
                            mode = command(wh_distance)
                            
                    key = cv2.waitKey(1) & 0xFF
                    if m_or_a=="m":
                        """Manual Mode"""
                        if key == ord('w'):
                            mode = "Go Straight"
                        elif key == ord('a'):
                            mode = "Turn Left"
                        elif key == ord('s'):
                            mode = "Go Back"
                        elif key == ord('d'):
                            mode = "Turn Right"
                        elif key == ord('e'):
                            mode = "STOP"
                        else:
                            mode = "STAY"
                    if key == ord('q'):
                        break
                    

            cv2.destroyAllWindows()

        finally:
            print "Connection closed on thread 1"


class ThreadServer(object):
    def server_thread(host, port):
        server = SocketServer.TCPServer((host, port), VideoStreamHandler)
        server.serve_forever()

    def server_thread2(host, port):
        server = SocketServer.TCPServer((host, port), SensorDataHandler)
        server.serve_forever()

    ip=socket.gethostbyname(socket.getfqdn())
    print(ip)
    distance_thread = threading.Thread(target=server_thread2, args=(ip, 9998))
    distance_thread.start()
    video_thread = threading.Thread(target=server_thread(ip, 9999))
    video_thread.start()

if __name__ == '__main__':
    ThreadServer()

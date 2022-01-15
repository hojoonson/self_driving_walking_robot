import time
import gpio_dummy as GPIO
import cv2
import numpy as np

def select_white(image, white):
                    	
                    
def line_direction(image, upper_limit):
    
    
    return result, left_sum, right_sum, forward_sum

def forward():
    GPIO.output(motor1A, GPIO.LOW)
    GPIO.output(motor1B, GPIO.HIGH)
    GPIO.output(motor2A, GPIO.LOW)
    GPIO.output(motor2B, GPIO.HIGH)

def backward():
    GPIO.output(motor1A, GPIO.HIGH)
    GPIO.output(motor1B, GPIO.LOW)
    GPIO.output(motor2A, GPIO.HIGH)
    GPIO.output(motor2B, GPIO.LOW)

def right():
    GPIO.output(motor1A, GPIO.LOW)
    GPIO.output(motor1B, GPIO.HIGH)
    GPIO.output(motor2A, GPIO.LOW)
    GPIO.output(motor2B, GPIO.LOW)
    
def left():
    GPIO.output(motor1A, GPIO.LOW)
    GPIO.output(motor1B, GPIO.LOW)
    GPIO.output(motor2A, GPIO.LOW)
    GPIO.output(motor2B, GPIO.HIGH)
    
def stop():
    GPIO.output(motor1A, GPIO.LOW)
    GPIO.output(motor1B, GPIO.LOW)
    GPIO.output(motor2A, GPIO.LOW)
    GPIO.output(motor2B, GPIO.LOW)

motor1A = 16
motor1B = 18
motor2A = 11
motor2B = 13


GPIO.setmode(GPIO.BOARD)
GPIO.setup(motor1A, GPIO.OUT)
GPIO.setup(motor1B, GPIO.OUT)
GPIO.setup(motor2A, GPIO.OUT)
GPIO.setup(motor2B, GPIO.OUT)

path = './datafile/vision.png'
for i in range(10000):
    start_time = time.time()
    try:
        temp = cv2.imread('./datafile/vision.png')
        masked_image=select_white(temp,200)
        a=line_direction(masked_image,150)
        print(a)
        
        if a[0] == 'forward':
            forward()
        elif a[0] == 'left   ':
            left()
        elif a[0] == 'right  ':
            right()
        elif a[0] == 'backward':
            backward()
    except:
        stop()
    while(start_time + 0.1 >  time.time()): # 10 fps
        c=1 
    
GPIO.cleanup()
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 04:54:00 2018

@author: SHF_W
"""

import time
import gpio_dummy as GPIO
import cv2
import numpy as np

def select_white(image, white):
                    
                    
def line_direction(image, upper_limit):
    
    
    return result, left_sum, right_sum, forward_sum

def forward(left, right):  
    p1A.changeDutyCycle(0)
    p1B.changeDutyCycle(100)
    p2A.changeDutyCycle(0)
    p2B.changeDutyCycle(100)

def backward():
   
def right(left, right):
    
        
def left(left, right):
    
    
def stop():
    p1A.changeDutyCycle(0)
    p1B.changeDutyCycle(0)
    p2A.changeDutyCycle(0)
    p2B.changeDutyCycle(0) 

motor1A = 16
motor1B = 18
motor2A = 11
motor2B = 13


GPIO.setmode(GPIO.BOARD)
p1A = GPIO.PWM(motor1A, 600)
p1B = GPIO.PWM(motor1B, 600)
p2A = GPIO.PWM(motor2A, 600)
p2B = GPIO.PWM(motor2B, 600)
p1A.start(100)
p1B.start(100)
p2A.start(100)
p2B.start(100)


path = './datafile/vision.png'
for i in range(10000):
    start_time = time.time()
    try:
        temp = cv2.imread('./datafile/vision.png')
        masked_image=select_white(temp,200)
        a=line_direction(masked_image,150)
        print(a)
        
        if a[0] == 'forward':
            forward(a[1], a[2])
        elif a[0] == 'left   ':
            left(a[1],a[2])
        elif a[0] == 'right  ':
            right(a[1],a[2])
        elif a[0] == 'backward':
            backward()
    except:
        stop()
    while(start_time + 0.1 >  time.time()): # 10 fps
        c=1 
    
GPIO.cleanup()
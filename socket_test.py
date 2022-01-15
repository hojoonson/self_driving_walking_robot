__author__ = 'zhengwang'

import threading
import SocketServer
#import serial
import cv2
import numpy as np
import math
import socket

class ThreadServer(object):

    def server_thread(host, port):
        server = SocketServer.TCPServer((host, port), VideoStreamHandler)
        server.serve_forever()

    def server_thread2(host, port):
        server = SocketServer.TCPServer((host, port), SensorDataHandler)
        server.serve_forever()

    distance_thread = threading.Thread(target=server_thread2('165.132.138.161', 9998))
    distance_thread.start()
    video_thread = threading.Thread(target=server_thread('165.132.138.161', 9999))
    video_thread.start()

if __name__ == '__main__':
    ThreadServer()

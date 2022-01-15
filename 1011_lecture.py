import threading
import SocketServer
import cv2
import numpy as np
import math
import socket

class VideoStreamHandler(SocketServer.StreamRequestHandler):

    def handle(self):

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
                    image = image[image_h/2:,0:image_w]
                    gray = gray[image_h/2:,0:image_w]

                    cv2.imshow('image', image)
                    cv2.imshow('grayimage', gray)
                    cv2.waitKey(1) & 0xFF
            cv2.destroyAllWindows()

        finally:
            print "Connection closed on thread 1"


class ThreadServer(object):
    def server_thread(host, port):
        server = SocketServer.TCPServer((host, port), VideoStreamHandler)
        server.serve_forever()


    video_thread = threading.Thread(target=server_thread('165.132.138.161', 9999))
    video_thread.start()

if __name__ == '__main__':

    ThreadServer()

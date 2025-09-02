import threading
import queue
import cv2

q = queue.Queue()

def receive():
    cap = cv2.VideoCapture('udp://@127.0.0.1:8554?buffer_size=1048576&pkt_size=65535&fifo_size=1048576&timeout=5000000')
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ret, frame = cap.read()
    q.put(frame)
    while ret:
        ret, frame = cap.read()
        q.put(frame)

def display():
    while True:
        if q.empty() != True:
            frame = q.get()
            cv2.imshow('Video', frame)
        k = cv2.waitKey(1) & 0xff
        if k == 27:  # press 'ESC' to quit
            break

tr = threading.Thread(target=receive, daemon=True)
td = threading.Thread(target=display)
tr.start()
td.start()
td.join()
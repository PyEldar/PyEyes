#dsa
import threading
import time
import requests

import cv2
import numpy as np

class VideoReceiver:
    """Main class handling connections and detection threads"""
    imgs = {}
    urls = []
    video_sources = []
    recv_threads = []
    proccessed_images = {}
    number_of_eyes = {}

    stop_event = threading.Event()
    # for printing and cascadeClasifier.detectMultiScale - throws unspecified error without lock
    lock = threading.Lock()
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    def __init__(self, urls=None, video_sources=None):
        if urls is None:
            self.urls = []
        self.urls = urls
        if video_sources is None:
            self.video_sources = []
        self.video_sources = video_sources

    def register_url(self, url):
        self.urls.append(url)

    def register_video_source(self, video_source):
        self.video_sources.append(video_source)

    def start_threads(self):
        """starts receiving threads for every url and every video source"""

        for url in self.urls:
            self.recv_threads.append(threading.Thread(target=self.start_receive, args=(url, )))

        for video_source in self.video_sources:
            self.recv_threads.append(threading.Thread(target=self.start_receive_OPCV, args=(video_source, )))

        for thread in self.recv_threads:
            thread.start()

    def stop_threads(self):
        self.stop_event.set()

    def start_receive(self, url):
        """
            background thread that handles reading from a SINGLE URL
            and img saves to imgs dict with key that is threading.get_ident()
        """

        # face and eyes recognition is started from this thread too
        try:
            r = None
            self.imgs[str(threading.get_ident())] = None

            while not self.stop_event.is_set():
                try:
                    r = requests.get(url, stream=True, timeout=6)
                    break
                except requests.exceptions.Timeout:
                    time.sleep(1)
                except requests.exceptions.ConnectionError:
                    time.sleep(1)
                except requests.exceptions.RequestException as e:
                    print(e)

            if r and r.status_code == 200:
                t = threading.Thread(target=self.detect_eyes, args=(threading.get_ident(), ))
                t.start()

                with self.lock:
                    print("Started thread that shows img", threading.get_ident())

                with self.lock:
                    print("URL: " + url + " connected, status OK")

                bytes_array = bytes()

                try:
                    for chunk in r.iter_content(chunk_size=1024):

                        if self.stop_event.is_set():
                            break

                        bytes_array += chunk
                        a = bytes_array.find(b'\xff\xd8')
                        b = bytes_array.find(b'\xff\xd9')
                        if a != -1 and b != -1:
                            jpg = bytes_array[a:b+2]
                            bytes_array = bytes_array[b+2:]
                            self.imgs[str(threading.get_ident())] = jpg

                except requests.exceptions.ConnectionError:
                    print("Thread", threading.get_ident(), "Connection error - stopping, url:", url)

                except KeyboardInterrupt:
                    self.stop_event.set()
        except Exception as err:
            print(err)

        print("exiting receive")

    def detect_eyes(self, identificator):
        """background thread that does face and eyes recognition"""

        #wait until imgs are saved from start_receive thread
        while self.imgs[str(identificator)] is None and not self.stop_event.is_set():
            time.sleep(0.1)


        while not self.stop_event.is_set():
            try:

                img = cv2.imdecode(np.fromstring(self.imgs[str(identificator)], dtype=np.uint8), cv2.IMREAD_COLOR)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                with self.lock:
                    faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = img[y:y+h, x:x+w]

                    with self.lock:
                        eyes = self.eye_cascade.detectMultiScale(roi_gray)

                    self.number_of_eyes[str(identificator)] = len(eyes)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

                #save proccessed image so it can be shown
                self.proccessed_images[str(identificator)] = img

            except KeyboardInterrupt:
                break

        print("exiting detection")

    def show_imgs(self):
        """takes all proccessed imgs, joins them and shows final image"""

        print("entered show_imgs")
        # wait until at least one img is ready
        while not self.proccessed_images:
            time.sleep(0.05)

        while not self.stop_event.is_set():
            indexes = list(self.proccessed_images)
            img = self.proccessed_images[indexes[0]]
            indexes.pop(0)

            #can be joined horizontaly
            if len(self.proccessed_images) <= 3:
                for index in indexes:
                    img = np.concatenate((img, self.proccessed_images[index]), axis=1)

            #must be 2x3
            elif len(self.proccessed_images) == 6:
                for index in range(2):
                    img = np.concatenate((img, self.proccessed_images[index]), axis=1)

                img2 = self.proccessed_images[2]
                for index in range(3, 6):
                    img2 = np.concatenate((img, self.proccessed_images[index]), axis=1)

                img = np.concatenate((img, img2), axis=0)

            # must be 2x2
            elif len(self.proccessed_images) == 4:
                img = np.concatenate((img, self.proccessed_images[indexes[0]]), axis=1)
                img2 = np.concatenate((self.proccessed_images[indexes[1]], self.proccessed_images[indexes[2]]), axis=1)
                img = np.concatenate((img, img2), axis=0)

            # count all the eyes
            number_of_eyes = 0
            with self.lock:
                for value in list(self.number_of_eyes.values()):
                    number_of_eyes += value

            #print("Eyes detected:", number_of_eyes)

            cv2.imshow("cams", img)
            cv2.waitKey(50)

        print("Exiting show_imgs")

    def start_receive_OPCV(self, video_source):
        """
            background thread that handles reading from a SINGLE VIDEO SOURCE
            and the img is saved to imgs dict with key that is threading.get_ident()
        """

        # create VideoCapture instance - open camera
        cam = cv2.VideoCapture(video_source)

        # check if camera is opened
        if cam.isOpened():
            # start detect_eyes thread for imgs from this thread
            self.imgs[str(threading.get_ident())] = None
            t = threading.Thread(target=self.detect_eyes, args=(threading.get_ident(), ))
            t.start()


            # resolution
            cam.set(3, 640)
            cam.set(4, 480)
            #camera warmup
            time.sleep(0.2)
            # reading the image
            while not self.stop_event.is_set():
                try:
                    ret, img = cam.read()
                    # encode image as jpg = same type as from start_receive
                    self.imgs[str(threading.get_ident())] = cv2.imencode('.jpg', img)[1].tobytes()
                    # face eyes detection is not so fast so we can wait a while //THIS SETS FPS to max 20fps
                    time.sleep(0.05)
                except KeyboardInterrupt:
                    cam.release()
                    break
        else:
            print("Video source", video_source, "could not be opened")


vR = VideoReceiver(["http://192.168.43.1:8080/video", "http://192.168.0.102:8080/video"], [0, 1])
vR.start_threads()

try:
    vR.show_imgs()
except Exception as err:
    print(err)
    vR.stop_event.set()
except KeyboardInterrupt:
    vR.stop_event.set()
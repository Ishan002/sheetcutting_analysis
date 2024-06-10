import cv2

class videosave:
    def __init__(self, src, vidname,resize=1):
        self.cap = cv2.VideoCapture(src)
        self.width = int((self.cap.get(3))*resize)
        self.height = int((self.cap.get(4))*resize)
        self.fps = int(self.cap.get(5))
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.vid = cv2.VideoWriter(vidname, self.fourcc ,self.fps , (self.width, self.height))

    def addframe(self, frame):
        self.vid.write(frame)

    def releaseAll(self):
        self.vid.release()
        self.cap.release()


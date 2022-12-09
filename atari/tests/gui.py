import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import time

class Feedback:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_sources = video_source

        self.numerical = tk.Frame(self.window)
        self.audio = tk.Frame(self.window)
        self.video = tk.Frame(self.window)

        self.numerical.pack(side=tk.TOP)
        self.audio.pack()
        self.video.pack()

        # numerical entry
        self.label_numerical = tk.Label(self.window, text="Numerical Feedback")
        self.label_numerical.pack(in_=self.numerical, side=tk.LEFT)
        self.entry_numerical = tk.Entry(self.window)
        self.entry_numerical.pack(in_=self.numerical, side=tk.LEFT)


        self.vid = Webcam(video_source)
        self.canvas = tk.Canvas(window, width=self.vid.width+100, height=self.vid.height+100)
        self.canvas.pack(in_=self.video)

        self.btn_snapshot = tk.Button(window,text="Snapshot",width=50, command=self.snapshot)
        self.btn_snapshot.pack(in_=self.video, anchor=tk.CENTER, expand=True)

        self.delay = 15
        self.update()
        self.window.mainloop()

    def update(self):
        ret, frame = self.vid.get_frame()
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(30, 30, image=self.photo,anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def snapshot(self):
        ret, frame = self.vid.get_frame()
        if ret:
            cv2.imwrite("frame-"+time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    

# class Voice:
#     def __init__(self, )

class Webcam:
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
        self.window.mainloop()

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return (ret, None)



def main():
    Feedback(tk.Tk(), "Tkinter and OpenCV")

if __name__=='__main__':
    main()

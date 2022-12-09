from tkinter import *
import cv2
from PIL import Image, ImageTk

def main():
    window = Tk()
    window.geometry("700x350")
    
    label = Label(window)
    label.grid(row=0, column=0)
    cap = cv2.VideoCapture(0)
    show_frames()
    window.mainloop()

# Define function to show frame
def show_frames():

    # Get the latest frame and convert into Image
    cv2image= cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)

    # Convert image to PhotoImage
    imgtk = ImageTk.PhotoImage(image = img)
    label.imgtk = imgtk
    label.configure(image=imgtk)

    # Repeat after an interval to capture continiously
label.after(20, show_frames)

if __name__=='__main__':
    main()
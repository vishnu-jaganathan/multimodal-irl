import gym
from multiprocessing import Process, Queue
from multiprocessing import active_children
import sys
from time import sleep
import keyboard

KEY_LABEL = 0

def keydown(key):
    global KEY_LABEL

    if key == "0":
        KEY_LABEL = 4
    elif key == "9":
        KEY_LABEL = 3
    elif key == "8":
        KEY_LABEL = 2
    elif key == "7":
        KEY_LABEL = 1
    elif key == "6":
        KEY_LABEL = 0
    elif key == "5":
        KEY_LABEL = -1
    elif key == "4":
        KEY_LABEL = -2
    elif key == "3":
        KEY_LABEL = -3
    elif key == "2":
        KEY_LABEL = -4
    elif key == "1":
        KEY_LABEL = -5
    elif key == "down":
        sys.exit()
    
    while keyboard.is_pressed(key):
        pass


def func1(key_queue):
    global KEY_LABEL

    print ('start func1')
    while True:
        if keyboard.read_key() in {"1","2","3","4","5","6","7","8","9","0","down"}:
            keydown(keyboard.read_key())
            key_queue.put(KEY_LABEL)
            print("put", KEY_LABEL, "in")

    print ('end func1')

def func2(key_queue):
    global KEY_LABEL

    print ('start func2')
    i = 0
    while True:
        sleep(2)
        for _ in range(key_queue.qsize()):
            print(key_queue.get())
        print(i)
        i += 1
    print ('end func2')

def main():

    key_queue = Queue()
    p1 = Process(target=func1, args=(key_queue,))
    p1.start()
    p2 = Process(target=func2, args=(key_queue,))
    p2.start()

    while True:
        if len(active_children()) == 1:
            print("here")
            key_queue.close()
            key_queue.join_thread()
            
            p1.terminate()
            p1.join()
            p2.terminate()
            p2.join()
        if len(active_children()) == 0:
            break

if __name__=='__main__':
    main()

from time import sleep
import pygame
from pygame.locals import *
import sys

KEY_LABEL = 0.0

def keydown(event):
    global KEY_LABEL

    if event.key == K_0:
        KEY_LABEL = 4
    elif event.key == K_9:
        KEY_LABEL = 3
    elif event.key == K_8:
        KEY_LABEL = 2
    elif event.key == K_7:
        KEY_LABEL = 1
    elif event.key == K_6:
        KEY_LABEL = 0
    elif event.key == K_5:
        KEY_LABEL = -1
    elif event.key == K_4:
        KEY_LABEL = -2
    elif event.key == K_3:
        KEY_LABEL = -3
    elif event.key == K_2:
        KEY_LABEL = -4
    elif event.key == K_1:
        KEY_LABEL = -5
    elif event.key == K_DOWN:
        sys.exit()

def keyup(event):
    global KEY_LABEL
    KEY_LABEL = 0

def main():
    
        print(KEY_LABEL)

if __name__ == "__main__":
    main()
import pygame as pg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pg.init()
js=pg.joystick.Joystick(0)
js.init()

img2=mpimg.imread('Joy_img_used/car.png')
img1= mpimg.imread('Joy_img_used/bike.png')
img = img1
cx,cy=0.0,0.0


plt.ion()
r=True
c = 0
while r:
    for event in pg.event.get():
        if js.get_button(0):
            r=False
        elif js.get_button(11):
            img = img2  
            c+=1
            if(c == 2):
                img = img1  
                c=0
            
    x=js.get_axis(0)
    y=-js.get_axis(1) 
    print(js.get_axis(0),js.get_axis(1))

    speed = 0.05
    cx += x * speed
    cy += y * speed
    cx = max(-1, min(1, cx))
    cy = max(-1, min(1, cy))

    plt.clf()
    plt.subplot()
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title('joystick Position as Joystick Axes')
    plt.imshow(img, extent=[cx-0.1,cx+0.1,cy-0.1,cy+0.1], aspect='auto')
    plt.grid(True)
    plt.pause(0.01)
        
pg.quit()

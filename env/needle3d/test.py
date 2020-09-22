# OpenGL test
import pygame as pg
from pygame.locals import *

from libs.libegl import EGLContext
from OpenGL.GL import *
from OpenGL.GLU import *

from PIL import Image
import numpy as np


cubeVertices = ((1,1,1),(1,1,-1),(1,-1,-1),(1,-1,1),(-1,1,1),(-1,-1,-1),(-1,-1,1),(-1,1,-1))
cubeEdges = ((0,1),(0,3),(0,4),(1,2),(1,7),(2,5),(2,3),(3,6),(4,6),(4,7),(5,6),(5,7))
cubeQuads = ((0,3,6,4),(2,5,6,3),(1,2,5,7),(1,0,4,7),(7,4,6,5),(2,3,0,1))

def wireCube():
    glBegin(GL_LINES)
    for cubeEdge in cubeEdges:
        for cubeVertex in cubeEdge:
            glVertex3fv(cubeVertices[cubeVertex])
    glEnd()

def solidCube():
    glBegin(GL_QUADS)
    for cubeQuad in cubeQuads:
        for cubeVertex in cubeQuad:
            glVertex3fv(cubeVertices[cubeVertex])
    glEnd()

def write_png(filename, width, height):
    img_buf = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    # GL_RGB => 3 components per pixel
    img = np.frombuffer(img_buf, np.uint8).reshape(height, width, 3)[::-1]
    im = Image.fromarray(img)
    im.save(filename)
    print("image generated: " + filename)

def main():
    #pg.init()
    #display = (1680, 1050)
    #pg.display.set_mode(display, DOUBLEBUF|OPENGL)
    with EGLContext() as ctx:
        display = (800, 600)
        if not ctx.initialize(*display):
            print("Failed to initialize EGL context")
            return

        gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

        glTranslatef(0.0, 0.0, -5)

        for i in range(10):
            #for event in pg.event.get():
            #    if event.type == pg.QUIT:
                    #pg.quit()
            #        quit()

            glRotatef(1, 1, 1, 1)
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
            #solidCube()
            wireCube()
            #pg.display.flip()
            #pg.time.wait(10)
            write_png("test" + str(i) + ".png", display[0], display[1])


if __name__ == "__main__":
    main()

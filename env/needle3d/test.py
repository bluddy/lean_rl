# OpenGL test
#import pygame as pg
#from pygame.locals import *

import weakref

from libs.libegl import EGLContext
import OpenGL.GL as gl
import OpenGL.GL.shaders as gl_shaders
from OpenGL.GLU import *

from PIL import Image
import numpy as np
import glm


vertex_shader = """
#version 330

layout (location=0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
   gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

fragment_shader = """
#version 330
out vec4 FragColor;

uniform vec4 ourColor;

void main()
{
   FragColor = ourColor;
}
"""

vertices = np.array(
           [ 0.6,  0.6, 0.0,
            -0.6,  0.6, 0.0,
             0.0, -0.6, 0.0,
           ],
           dtype=np.float32)
indices = np.array(
            [0, 1, 2],
            dtype=np.uint32)

class Shader(object):
    def __init__(self, vertex_shader, frag_shdaer):
        self.shader = gl_shaders.compileProgram(
            gl_shaders.compileShader(vertex_shader, gl.GL_VERTEX_SHADER),
            gl_shaders.compileShader(fragment_shader, gl.GL_FRAGMENT_SHADER)
        )
        self.color_loc = gl.glGetUniformLocation(self.shader, 'ourColor')
        self.model_loc = gl.glGetUniformLocation(self.shader, 'model')
        self.view_loc = gl.glGetUniformLocation(self.shader, 'view')
        self.proj_loc = gl.glGetUniformLocation(self.shader, 'projection')

    def use(self, on:bool):
        if on:
            gl.glUseProgram(self.shader)
        else:
            gl.glUseProgram(0)

    def set_color(self, color):
        gl.glUniform4f(self.color_loc, *color)

    def set_model(self, model_mat):
        gl.glUniformMatrix4fv(self.model_loc, 1, gl.GL_FALSE, glm.value_ptr(model_mat))

    def set_view(self, model_mat):
        gl.glUniformMatrix4fv(self.view_loc, 1, gl.GL_FALSE, glm.value_ptr(model_mat))

    def set_proj(self, model_mat):
        gl.glUniformMatrix4fv(self.proj_loc, 1, gl.GL_FALSE, glm.value_ptr(model_mat))


class OpenGLObject(object):
    def __init__(self, renderer, shader, vao, color=(0.8, 0.5, 0.0, 1.0)):
        self.shader=shader
        self.color=color
        self.vao=vao
        self.renderer = weakref.ref(renderer)
        self.model_mat = glm.mat4()

    def draw(self):

        self.shader.use(True)

        # Set the color
        self.shader.set_color(self.color)

        self.shader.set_model(self.model_mat)
        self.shader.set_view(self.renderer().view_mat)
        self.shader.set_proj(self.renderer().proj_mat)

        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        gl.glBindVertexArray(0)

        self.shader.use(False)

class OpenGLRenderer(object):
    def __init__(self, ortho=True):
        self.ortho = ortho
        self.camera_loc = glm.vec3(0.0, 0.0, 1.0)
        self.camera_lookat = glm.vec3(0.0, 0.0, 0.0)
        self.camera_up = glm.vec3(0.0, 1.0, 0.0)
        self.view_mat = glm.lookAt(self.camera_loc, self.camera_lookat, self.camera_up)
        self.proj_mat = glm.ortho(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0)

        gl.glEnable(gl.GL_DEPTH_TEST)

    def create_object(self, shader, vertices, indices):
        # Create a new VAO (Vertex Array Object) and bind it
        vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao)

        # Generate buffers to hold our vertices
        vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)

        # Send the data over to the buffer
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)

        # Create buffer for indices
        ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)

        # Describe the position data layout in the buffer
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, 3*4, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)

        # Unbind the VAO first (Important)
        gl.glBindVertexArray(0)

        # Unbind other stuff
        gl.glDisableVertexAttribArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

        return OpenGLObject(self, shader=shader, vao=vao)

    def start_draw(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)


def write_png(filename, width, height):
    img_buf = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
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

        renderer = OpenGLRenderer()

        shader = Shader(vertex_shader, fragment_shader)

        obj = renderer.create_object(shader, vertices, indices)

        renderer.start_draw()

        for i in range(10):
            obj.draw()
            write_png("test" + str(i) + ".png", display[0], display[1])

if __name__ == "__main__":
    main()

# OpenGL test
import pygame as pg
from pygame.locals import *

from libs.libegl import EGLContext
import OpenGL.GL as gl
import OpenGL.GL.shaders as shaders
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

def create_object(shader):

def create_object():
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

    return vao

def draw(shader, vertex_array_object):
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    gl.glUseProgram(shader)

    # Set the color
    colorLoc = gl.glGetUniformLocation(shader, 'ourColor')
    gl.glUniform4f(colorLoc, 0.8, 0.5, 0.0, 1.0)

    # Set model matrix
    model_mat = glm.mat4()
    modelLoc = gl.glGetUniformLocation(shader, 'model')
    gl.glUniformMatrix4fv(modelLoc, 1, gl.GL_FALSE, glm.value_ptr(model_mat))

    view = glm.lookAt(
            glm.vec3(0.0, 0.0, 1.0), #location
            glm.vec3(0.0, 0.0, 0.0), #lookat
            glm.vec3(0.0, 1.0, 0.0), #up
            )
    viewLoc = gl.glGetUniformLocation(shader, 'view')
    gl.glUniformMatrix4fv(viewLoc, 1, gl.GL_FALSE, glm.value_ptr(view))

    #proj_mat = glm.ortho(0.0, 800.0, 0.0, 600.0, 0.1, 100.0)
    proj_mat = glm.ortho(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0)
    projLoc = gl.glGetUniformLocation(shader, 'projection')
    gl.glUniformMatrix4fv(projLoc, 1, gl.GL_FALSE, glm.value_ptr(proj_mat))

    gl.glBindVertexArray(vertex_array_object)
    gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

    gl.glBindVertexArray(0)
    gl.glUseProgram(0)

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

        gl.glEnable(gl.GL_DEPTH_TEST)

        shader = shaders.compileProgram(
            shaders.compileShader(vertex_shader, gl.GL_VERTEX_SHADER),
            shaders.compileShader(fragment_shader, gl.GL_FRAGMENT_SHADER)
        )

        vao = create_object()

        for i in range(10):
            draw(shader, vao)
            write_png("test" + str(i) + ".png", display[0], display[1])

if __name__ == "__main__":
    main()

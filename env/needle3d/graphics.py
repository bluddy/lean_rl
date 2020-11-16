# OpenGL test
#import pygame as pg
#from pygame.locals import *

import weakref

from libs.libegl import EGLContext
import OpenGL
OpenGL.ERROR_CHECKING=False
OpenGL.ERROR_LOGGING=False
import OpenGL.GL as gl
import OpenGL.GL.shaders as gl_shaders

from PIL import Image
import numpy as np
import glm
import math
import ctypes

DEFAULT=0
LIGHTING=1


class Shader(object):
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

    uniform vec4 objectColor;

    void main()
    {
    FragColor = objectColor;
    }
    """

    def __init__(self):
        self.shader = gl_shaders.compileProgram(
            gl_shaders.compileShader(self.vertex_shader, gl.GL_VERTEX_SHADER),
            gl_shaders.compileShader(self.fragment_shader, gl.GL_FRAGMENT_SHADER)
        )
        self.obj_color_loc = gl.glGetUniformLocation(self.shader, 'objectColor')

        self.model_loc = gl.glGetUniformLocation(self.shader, 'model')
        self.view_loc = gl.glGetUniformLocation(self.shader, 'view')
        self.proj_loc = gl.glGetUniformLocation(self.shader, 'projection')

    def use(self, on:bool):
        if on:
            gl.glUseProgram(self.shader)
        else:
            gl.glUseProgram(0)

    def set_object_color(self, color):
        gl.glUniform4f(self.obj_color_loc, *color)

    def set_model(self, model_mat):
        gl.glUniformMatrix4fv(self.model_loc, 1, gl.GL_FALSE, glm.value_ptr(model_mat))

    def set_view(self, model_mat):
        gl.glUniformMatrix4fv(self.view_loc, 1, gl.GL_FALSE, glm.value_ptr(model_mat))

    def set_proj(self, model_mat):
        gl.glUniformMatrix4fv(self.proj_loc, 1, gl.GL_FALSE, glm.value_ptr(model_mat))


class LightingShader(object):

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

    uniform float ambientStrength;
    uniform vec4 objectColor;
    uniform vec3 ambientLightColor;

    void main()
    {
    vec3 ambient = ambientStrength * ambientLightColor;
    FragColor = vec4(ambient, 1.0) * objectColor;
    }
    """
    def __init__(self):
        self.shader = gl_shaders.compileProgram(
            gl_shaders.compileShader(self.vertex_shader, gl.GL_VERTEX_SHADER),
            gl_shaders.compileShader(self.fragment_shader, gl.GL_FRAGMENT_SHADER)
        )
        self.ambient_str_loc = gl.glGetUniformLocation(self.shader, 'ambientStrength')
        self.ambient_color_loc = gl.glGetUniformLocation(self.shader, 'ambientLightColor')
        self.obj_color_loc = gl.glGetUniformLocation(self.shader, 'objectColor')

        self.model_loc = gl.glGetUniformLocation(self.shader, 'model')
        self.view_loc = gl.glGetUniformLocation(self.shader, 'view')
        self.proj_loc = gl.glGetUniformLocation(self.shader, 'projection')

    def set_default_values(self):
        self.set_ambient_strength(0.2)
        self.set_ambient_color((1.0, 1.0, 1.0))

    def use(self, on:bool):
        if on:
            gl.glUseProgram(self.shader)
            self.set_default_values()
        else:
            gl.glUseProgram(0)

    def set_object_color(self, color):
        gl.glUniform4f(self.obj_color_loc, *color)

    def set_ambient_color(self, color):
        gl.glUniform3f(self.ambient_color_loc, *color)

    def set_ambient_strength(self, x):
        gl.glUniform1f(self.ambient_str_loc, x)

    def set_model(self, model_mat):
        gl.glUniformMatrix4fv(self.model_loc, 1, gl.GL_FALSE, glm.value_ptr(model_mat))

    def set_view(self, model_mat):
        gl.glUniformMatrix4fv(self.view_loc, 1, gl.GL_FALSE, glm.value_ptr(model_mat))

    def set_proj(self, model_mat):
        gl.glUniformMatrix4fv(self.proj_loc, 1, gl.GL_FALSE, glm.value_ptr(model_mat))

LINES=0
TRIANGLES=1

class OpenGLObject(object):
    def __init__(self, renderer, shader, vao, buffers, vertices, indices,
            color=(0.8, 0.5, 0.0, 1.0), primitive=TRIANGLES):
        self.shader=shader
        self.color=color
        self.vao=vao
        self.buffers=buffers
        self.vertices = vertices
        self.indices = indices
        self.renderer = weakref.ref(renderer)
        self.model = glm.mat4()
        self.primitive = primitive

    def draw(self):
        self.shader.use(True)

        # Set the color
        self.shader.set_object_color(self.color)

        self.shader.set_model(self.model)
        self.shader.set_view(self.renderer().view_mat)
        self.shader.set_proj(self.renderer().proj_mat)

        gl.glBindVertexArray(self.vao)
        if self.primitive == TRIANGLES:
            gl.glDrawElements(gl.GL_TRIANGLES, len(self.indices), gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))
        elif self.primitive == LINES:
            gl.glDrawArrays(gl.GL_LINES, 0, len(self.vertices))
        else:
            raise ValueError("wrong primitive")
        gl.glBindVertexArray(0)

        self.shader.use(False)

    def reset(self):
        self.model = glm.mat4()

    def rotate(self, angle, vec=(0., 0., -1.)):
        self.model = glm.rotate(self.model, angle, glm.vec3(vec))

    def translate(self, vec):
        self.model = glm.translate(self.model, glm.vec3(vec))

    def scale(self, vec):
        self.model = glm.scale(self.model, glm.vec3(vec))

    def set_color(self, vec):
        self.color = vec

    def __del__(self):
        gl.glDeleteBuffers(len(self.buffers), self.buffers)

class OpenGLRenderer(object):
    def __init__(self, res=(800,600), ortho=True, bg_color=(0., 0., 0.)):
        self.res = res
        self.camera_loc = glm.vec3(0.0, 0.0, 1.0)
        self.camera_lookat = glm.vec3(0.0, 0.0, 0.0)
        self.camera_up = glm.vec3(0.0, 1.0, 0.0)
        self.update_view_matrix()
        self.proj_mat = glm.ortho(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0)

        self.ctx =  EGLContext()

        if not self.ctx.initialize(*self.res):
            print("Failed to initialize EGL context")
            return

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glLineWidth(10)
        gl.glClearColor(*bg_color, 1.)

        self.shaders = {}
        self.shaders[DEFAULT] = Shader()
        self.shaders[LIGHTING] = LightingShader()

    def get_width(self):
        return self.res[0]
    def get_height(self):
        return self.res[1]

    def set_ortho(self, left, right, bottom, top):
        self.proj_mat = glm.ortho(left, right, bottom, top, -0.1, 100.0)

    def set_perspective(self, fov=45):
        self.proj_mat = glm.perspectiveFov(glm.radians(fov), float(self.res[0]), float(self.res[1]), -0.1, 100.)

    def set_camera_loc(self, vec):
        self.camera_loc = glm.vec3(vec)

    def set_camera_up(self, vec):
        self.camera_up = glm.vec3(vec)

    def move_camera(self, vec):
        self.camera_loc += glm.vec3(vec)

    def set_camera_lookat(self, vec):
        self.camera_lookat = glm.vec3(vec)

    def set_camera_up(self, vec):
        self.camera_up = glm.vec3(vec)

    def update_view_matrix(self):
        self.view_mat = glm.lookAt(self.camera_loc, self.camera_lookat, self.camera_up)

    def translate_camera(self, vec):
        self.view_mat = glm.translate(self.view_mat, glm.vec3(vec))

    def get_img(self):
        gl.glFinish()
        img_buf = gl.glReadPixels(0, 0, self.res[0], self.res[1], gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        img = np.frombuffer(img_buf, np.uint8).reshape(self.res[1], self.res[0], 3)[::-1]
        return img

    def create_object(self, vertices, indices=None, shader=DEFAULT, primitive=TRIANGLES):
        # Create a new VAO (Vertex Array Object) and bind it
        vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao)

        # Generate buffers to hold our vertices
        vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)

        # Describe the position data layout in the buffer
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, 0, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)

        buffers=[vbo, vao]

        # Create buffer for indices if needed
        if indices is not None:
            ebo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)

            buffers.insert(0, ebo)

        # Unbind the VAO first (important)
        gl.glBindVertexArray(0)
        # Unbind other stuff
        gl.glDisableVertexAttribArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

        shader = self.shaders[shader]

        return OpenGLObject(self, shader=shader, vao=vao, buffers=np.array(buffers),
                vertices=vertices, indices=indices, primitive=primitive)

    def create_rectangle(self, shader=DEFAULT):
        vertices = np.array(
                [ 0.5, 0.5, 0.,
                  -0.5, 0.5, 0.,
                  0.5, -0.5, 0.,
                  -0.5, -0.5, 0.,
                ], dtype=np.float32)
        # CCW
        indices = np.array(
                [0, 1, 2,
                 2, 1, 3,
                 ], dtype=np.uint32)
        return self.create_object(vertices, indices, shader=shader)

    def create_cube(self, shader=DEFAULT):
        #1 -----0
        #|\     |\
        #| \    | \
        #|  5---+--4
        #|  |   |  |
        #3--+---2  |
        # \ |    \ |
        #  \|     \|
        #   7------6
        p = 0.5
        n = -0.5
        vertices = np.array(
                [ p, p, p, # 0
                  n, p, p, # 1
                  p, n, p, # 2
                  n, n, p, # 3
                  p, p, n, # 4
                  n, p, n, # 5
                  p, n, n, # 6
                  n, n, n, # 7
                ], dtype=np.float32)
        # CCW
        indices = np.array(
                [0, 4, 6, # right side
                 6, 2, 0,
                 5, 1, 3, # left side
                 3, 7, 5,
                 1, 0, 2, # back side
                 2, 3, 1,
                 4, 5, 7, # front
                 7, 6, 4,
                 0, 1, 5, # top
                 5, 4, 0,
                 3, 2, 6, # bottom
                 6, 7, 3
                 ], dtype=np.uint32)
        return self.create_object(vertices, indices, shader=shader)

    def create_triangle(self, shader=DEFAULT):
        vertices = np.array(
                [ 0.5,  0.5, 0.,
                  -0.5,  0.5, 0.,
                  0., -0.5, 0.,
                ], dtype=np.float32)
        indices = np.array(
                [0, 1, 2],
                dtype=np.uint32)
        return self.create_object(vertices, indices, shader=shader)

    def create_wireframe_rec(self, shader=DEFAULT):
        vertices = np.array(
                [ 0.5, 0.5, 0.,
                 -0.5, 0.5, 0.,
                 -0.5, 0.5, 0.,
                 -0.5, -0.5, 0.,
                 -0.5, -0.5, 0.,
                 0.5, -0.5, 0.,
                 0.5, -0.5, 0.,
                 0.5, 0.5, 0.
                 ], np.float32)
        return self.create_object(vertices, primitive=LINES, shader=shader)

    def start_draw(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    def close(self):
        if self.ctx is not None:
            self.ctx.release()
            self.ctx = None

    def __del__(self):
        self.close()

def write_png(filename, renderer):
    img = renderer.get_img()
    im = Image.fromarray(img)
    im.save(filename)
    print("image generated: " + filename)

def main():

    renderer = OpenGLRenderer(bg_color=(0.5, 0.75, 0.))

    obj = renderer.create_rectangle()
    obj.rotate(math.pi/4)

    renderer.start_draw()

    for i in range(10):
        obj.draw()
        write_png("test" + str(i) + ".png", renderer)

if __name__ == "__main__":
    main()

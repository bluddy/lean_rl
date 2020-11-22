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
import ctypes as ct

NONE=0
DEFAULT=1
LIGHTING=2

sizeof_float = ct.sizeof(ct.c_float)

def arr(x):
    return np.array(x, dtype=np.float32)

class Shader(object):
    vertex_shader = """
    #version 330

    layout (location=0) in vec3 aPos;

    uniform mat4 model;
    uniform mat4 normal_matrix;
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

    def __init__(self, renderer):
        self.id = DEFAULT
        self.renderer = renderer
        self.shader = gl_shaders.compileProgram(
            gl_shaders.compileShader(self.vertex_shader, gl.GL_VERTEX_SHADER),
            gl_shaders.compileShader(self.fragment_shader, gl.GL_FRAGMENT_SHADER)
        )
        self.obj_color_loc = gl.glGetUniformLocation(self.shader, 'objectColor')

        self.model_loc = gl.glGetUniformLocation(self.shader, 'model')
        self.normal_matrix_loc = gl.glGetUniformLocation(self.shader, 'normal_matrix')
        self.view_loc = gl.glGetUniformLocation(self.shader, 'view')
        self.proj_loc = gl.glGetUniformLocation(self.shader, 'projection')

    def use(self):
        gl.glUseProgram(self.shader)
        self.renderer.active_shader = self.id

    def set_object_color(self, color):
        gl.glUniform4f(self.obj_color_loc, *color)

    def set_model(self, model_mat):
        gl.glUniformMatrix4fv(self.model_loc, 1, gl.GL_FALSE, glm.value_ptr(model_mat))

    def set_normal_matrix(self, mat):
        gl.glUniformMatrix4fv(self.normal_matrix_loc, 1, gl.GL_FALSE, glm.value_ptr(mat))

    def set_view(self, mat):
        gl.glUniformMatrix4fv(self.view_loc, 1, gl.GL_FALSE, glm.value_ptr(mat))

    def set_proj(self, mat):
        gl.glUniformMatrix4fv(self.proj_loc, 1, gl.GL_FALSE, glm.value_ptr(mat))


class LightingShader(object):

    vertex_shader = """
    #version 330

    layout (location=0) in vec3 aPos;
    layout (location=1) in vec3 aNormal;

    uniform mat4 model;
    uniform mat4 normal_matrix; // For correcting normals
    uniform mat4 view;
    uniform mat4 projection;

    out vec3 Normal;
    out vec3 FragPos;

    void main()
    {
        gl_Position = projection * view * model * vec4(aPos, 1.0);

        // Calculate world pos of vertex (to get interpolated fragment pos for triangle)
        FragPos = vec3(model * vec4(aPos, 1.0));

        Normal = mat3(normal_matrix) * aNormal; // Passthrough
    }
    """

    fragment_shader = """

    #version 330

    out vec4 FragColor;

    in vec3 Normal;
    in vec3 FragPos;

    uniform float ambientStrength;
    uniform vec4 objectColor;
    uniform vec3 lightColor;
    uniform vec3 lightPos;

    void main()
    {
        // Ambient light
        vec3 ambient = ambientStrength * lightColor;

        // Calculate diffuse light
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * lightColor;

        // debug FragColor = vec4(norm, 1.0);
        FragColor = vec4(ambient + diffuse, 1.0) * objectColor;
    }
    """

    def __init__(self, renderer):
        self.id = LIGHTING
        self.renderer = renderer
        self.shader = gl_shaders.compileProgram(
            gl_shaders.compileShader(self.vertex_shader, gl.GL_VERTEX_SHADER),
            gl_shaders.compileShader(self.fragment_shader, gl.GL_FRAGMENT_SHADER)
        )
        self.ambient_str_loc = gl.glGetUniformLocation(self.shader, 'ambientStrength')
        self.light_color_loc = gl.glGetUniformLocation(self.shader, 'lightColor')
        self.obj_color_loc = gl.glGetUniformLocation(self.shader, 'objectColor')
        self.light_pos_loc = gl.glGetUniformLocation(self.shader, 'lightPos')

        self.model_loc = gl.glGetUniformLocation(self.shader, 'model')
        self.normal_matrix_loc = gl.glGetUniformLocation(self.shader, 'normal_matrix')
        self.view_loc = gl.glGetUniformLocation(self.shader, 'view')
        self.proj_loc = gl.glGetUniformLocation(self.shader, 'projection')

        self.use() # must have for init uniforms

        self.set_ambient_strength(0.2)
        self.set_light_color((1.0, 1.0, 1.0))
        self.set_light_pos((200,200,200))

    def use(self):
        gl.glUseProgram(self.shader)
        self.renderer.active_shader = self.id

    # Determined by object
    def set_object_color(self, color):
        gl.glUniform4f(self.obj_color_loc, *color)

    def set_model(self, model_mat):
        gl.glUniformMatrix4fv(self.model_loc, 1, gl.GL_FALSE, glm.value_ptr(model_mat))

    def set_normal_matrix(self, mat):
        gl.glUniformMatrix4fv(self.normal_matrix_loc, 1, gl.GL_FALSE, glm.value_ptr(mat))

    # Determined by renderer
    def set_light_color(self, color):
        gl.glUniform3f(self.light_color_loc, *color)

    def set_ambient_strength(self, x):
        gl.glUniform1f(self.ambient_str_loc, x)

    def set_light_pos(self, vec):
        gl.glUniform3f(self.light_pos_loc, *vec)

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
        self.renderer = renderer
        self.model = glm.mat4()
        self.normal_matrix = glm.inverseTranspose(self.model)
        self.model_dirty = False
        self.primitive = primitive

    def draw(self):
        self.shader.use()

        # Set the color
        self.shader.set_object_color(self.color)

        if self.model_dirty:
            self.normal_matrix = glm.inverseTranspose(self.model)
            self.model_dirty = False
        self.shader.set_model(self.model)
        self.shader.set_normal_matrix(self.normal_matrix)
        self.shader.set_view(self.renderer.view_mat)
        self.shader.set_proj(self.renderer.proj_mat)

        gl.glBindVertexArray(self.vao)
        if self.primitive == TRIANGLES:
            if self.indices is not None:
                gl.glDrawElements(gl.GL_TRIANGLES, len(self.indices), gl.GL_UNSIGNED_INT, ct.c_void_p(0))
            else:
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(self.vertices))

        elif self.primitive == LINES:
            gl.glDrawArrays(gl.GL_LINES, 0, len(self.vertices))
        else:
            raise ValueError("wrong primitive")
        gl.glBindVertexArray(0)

    def reset(self):
        self.model = glm.mat4()
        self.model_dirty = True

    def rotate(self, angle, vec=(0., 0., -1.)):
        self.model = glm.rotate(self.model, angle, glm.vec3(vec))
        self.model_dirty = True

    def translate(self, vec):
        self.model = glm.translate(self.model, glm.vec3(vec))
        self.model_dirty = True

    def scale(self, vec):
        self.model = glm.scale(self.model, glm.vec3(vec))
        self.model_dirty = True

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
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glLineWidth(10)
        gl.glClearColor(*bg_color, 1.)
        gl.glFrontFace(gl.GL_CCW)

        self.shaders = {}
        self.shaders[DEFAULT] = Shader(self)
        self.shaders[LIGHTING] = LightingShader(self)

        self.active_shader = NONE

    def get_width(self):
        return self.res[0]
    def get_height(self):
        return self.res[1]

    def set_ortho(self, left, right, bottom, top):
        self.proj_mat = glm.ortho(left, right, bottom, top, -0.1, 100.0)

    def set_perspective(self, fov=45):
        self.proj_mat = glm.perspectiveFov(glm.radians(fov), float(self.res[0]), float(self.res[1]), 1., 5000.)

    def set_camera_loc(self, vec):
        self.camera_loc = glm.vec3(vec)

    def set_camera_up(self, vec):
        self.camera_up = glm.vec3(vec)

    def _adjust_lighting_shader(self, fun):
        '''
        Pass in a function to carry out on the lighting shader
        Must be temporarily changed to active shader
        '''
        old_shader_id = self.active_shader
        shader = self.shaders[LIGHTING]
        shader.use()
        fun(shader)
        self.shaders[old_shader_id].use()

    def set_light_color(self, color):
        self._adjust_lighting_shader(lambda shader: shader.set_light_color(color))

    def set_ambient_strength(self, x):
        self._adjust_lighting_shader(lambda shader: shader.set_ambient_strength(x))

    def set_light_pos(self, vec):
        self._adjust_lighting_shader(lambda shader: shader.set_light_pos(vec))

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

    def create_object(self, vertices, indices=None, shader=DEFAULT, primitive=TRIANGLES, normals=True):
        '''
        stride: how far to move between vertices
        '''
        # Create a new VAO (Vertex Array Object) and bind it
        vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao)

        # Generate buffers to hold our vertices
        vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)

        # Describe the position data layout in the buffer
        stride = 3 * sizeof_float
        if normals:
            stride = 6 * sizeof_float

        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, stride, ct.c_void_p(0))
        gl.glEnableVertexAttribArray(0)

        if normals:
            gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, False, stride, ct.c_void_p(3 * sizeof_float))
            gl.glEnableVertexAttribArray(1)

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
                [ 0.5, 0.5, 0., 0., 0., 1.,
                  -0.5, 0.5, 0., 0., 0., 1.,
                  0.5, -0.5, 0., 0., 0., 1.,
                  -0.5, -0.5, 0., 0., 0., 1.,
                ], dtype=np.float32)
        # CCW
        indices = np.array(
                [0, 1, 2,
                 2, 1, 3,
                 ], dtype=np.uint32)
        return self.create_object(vertices, indices, shader=shader)

    def create_cube(self, shader=DEFAULT):
        # Into the screen is neg
        #1 -----0
        #|\   F |\
        #| \    | \
        #|  5D--+--4
        #|A |   | C|
        #3--+--B2  |
        # \ | E  \ |
        #  \|     \|
        #   7------6
        p = 0.5
        n = -0.5
        v0 = [p, p, n]
        v1 = [n, p, n]
        v2 = [p, n, n]
        v3 = [n, n, n]
        v4 = [p, p, p]
        v5 = [n, p, p]
        v6 = [p, n, p]
        v7 = [n, n, p]

        nA = [-1, 0, 0]
        nB = [0, 0, 1]
        nC = [1, 0, 0]
        nD = [0, 0, -1]
        nE = [0, -1, 0]
        nF = [0, 1, 0]

        vertices = [
                *v0, *nF, *v2, *nF, *v3, *nF,
                *v3, *nF, *v1, *nF, *v0, *nF,
                *v1, *nA, *v3, *nA, *v7, *nA,
                *v7, *nA, *v5, *nA, *v1, *nA,
                *v5, *nB, *v7, *nB, *v6, *nB,
                *v6, *nB, *v4, *nB, *v5, *nB,
                *v4, *nC, *v6, *nC, *v2, *nC,
                *v2, *nC, *v0, *nC, *v4, *nC,
                *v0, *nD, *v2, *nD, *v3, *nD,
                *v3, *nD, *v1, *nD, *v0, *nD,
                *v2, *nE, *v6, *nE, *v7, *nE,
                *v7, *nE, *v3, *nE, *v2, *nE,
                *v0, *nF, *v1, *nF, *v5, *nF,
                *v5, *nF, *v4, *nF, *v0, *nF,
        ]
        vertices = np.array(vertices, dtype=np.float32)
        return self.create_object(vertices, shader=shader)
    #TODO: format with normals

    def create_triangle(self, shader=DEFAULT):
        vertices = np.array(
                [ 0.5,  0.5, 0., 0., 0., 1.,
                  -0.5,  0.5, 0., 0., 0., 1.,
                  0., -0.5, 0., 0., 0., 1.,
                ], dtype=np.float32)
        indices = np.array(
                [0, 1, 2],
                dtype=np.uint32)
        return self.create_object(vertices, indices, shader=shader)

    def calc_normals(self, v0, v1, v2):
        ''' Calculate normals for counter-clockwise vertices '''
        vec1 = v1 - v0
        vec2 = v2 - v1
        normal = np.cross(vec1, vec2)
        normal /= np.linalg.norm(normal)
        return normal

    def add_normals(self, vertices):
        ''' Add normals to vertex list '''

        out = []
        vs = []
        for i, v in enumerate(vertices):
            if i and i % 3 == 0:
                n = self.calc_normals(*vs)
                out.extend([vs[0], n, vs[1], n, vs[2], n])
                vs = []

            vs.append(v)
        return out

    def create_pyramid(self, shader=DEFAULT):
        #                          0
        #      0                  / \
        #    / | \               /   \
        #   /--4--\             /     \
        #  1-- | --3           1-------2
        #    --2-
        v0 = arr([0, 0, 0.5])
        v1 = arr([-0.5, -0.5, -0.5])
        v2 = arr([0.5, -0.5, -0.5])
        v3 = arr([0.5, 0.5, -0.5])
        v4 = arr([-0.5, 0.5, -0.5])
        vertices = [
          v0, v1, v2,
          v0, v2, v3,
          v0, v3, v4,
          v0, v4, v1,
          v1, v4, v2,
          v4, v3, v2,
          ]
        vertices = self.add_normals(vertices)
        vertices = np.array(vertices, dtype=np.float32)
        return self.create_object(vertices, shader=shader)

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
        return self.create_object(vertices, primitive=LINES, shader=shader, normals=False)

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

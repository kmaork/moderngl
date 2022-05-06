import moderngl
import numpy as np
import zengl
from objloader import Obj
from pyrr import Matrix44

from ported._example import Example


def kernel(s):
    x = np.arange(-s, s + 1)
    y = np.exp(-x * x / (s * s / 4))
    y /= y.sum()
    v = ', '.join(f'{t:.8f}' for t in y)
    # from matplotlib import pyplot as plt
    # plt.plot(x, y)
    # plt.show()
    return f'const int N = {s * 2 + 1};\nfloat coeff[N] = float[]({v});'


def zengl_pipeline():
    ctx = zengl.context()
    ctx.includes['kernel'] = kernel(19)

    size = (512, 512)
    image = ctx.image(size, 'rgba8unorm')
    depth = ctx.image(size, 'depth24plus')

    temp = ctx.image(size, 'rgba8unorm')
    output = ctx.image(size, 'rgba8unorm')

    image.clear_value = (0.2, 0.2, 0.2, 1.0)

    model = Obj.open('examples/data/sitting_dummy.obj').pack('vx vy vz nx ny nz')
    vertex_buffer = ctx.buffer(model)

    uniform_buffer = ctx.buffer(size=80)

    monkey = ctx.pipeline(
        vertex_shader='''
            #version 330

            layout (std140) uniform Common {
                mat4 mvp;
            };

            layout (location = 0) in vec3 in_vert;
            layout (location = 1) in vec3 in_norm;

            out vec3 v_norm;

            void main() {
                gl_Position = mvp * vec4(in_vert, 1.0);
                v_norm = in_norm;
            }
        ''',
        fragment_shader='''
            #version 330

            in vec3 v_norm;

            layout (location = 0) out vec4 out_color;

            void main() {
                vec3 light = vec3(-140.0, -300.0, 350.0);
                float lum = dot(normalize(light), normalize(v_norm)) * 0.7 + 0.3;
                out_color = vec4(lum, lum, lum, 1.0);
            }
        ''',
        layout=[
            {
                'name': 'Common',
                'binding': 0,
            },
        ],
        resources=[
            {
                'type': 'uniform_buffer',
                'binding': 0,
                'buffer': uniform_buffer,
            },
        ],
        framebuffer=[image, depth],
        topology='triangles',
        cull_face='back',
        vertex_buffers=zengl.bind(vertex_buffer, '3f 3f', 0, 1),
        vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f'),
    )

    blur_x = ctx.pipeline(
        vertex_shader='''
            #version 330

            vec2 positions[3] = vec2[](
                vec2(-1.0, -1.0),
                vec2(3.0, -1.0),
                vec2(-1.0, 3.0)
            );

            void main() {
                gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
            }
        ''',
        fragment_shader='''
            #version 330

            uniform sampler2D Texture;

            layout (location = 0) out vec4 out_color;

            #include "kernel"

            void main() {
                vec3 color = vec3(0.0, 0.0, 0.0);
                for (int i = 0; i < N; ++i) {
                    color += texelFetch(Texture, ivec2(gl_FragCoord.xy) + ivec2(i - N / 2, 0), 0).rgb * coeff[i];
                }
                out_color = vec4(color, 1.0);
            }
        ''',
        layout=[
            {
                'name': 'Texture',
                'binding': 0,
            },
        ],
        resources=[
            {
                'type': 'sampler',
                'binding': 0,
                'image': image,
            },
        ],
        framebuffer=[temp],
        topology='triangles',
        vertex_count=3,
    )

    blur_y = ctx.pipeline(
        vertex_shader='''
            #version 330

            vec2 positions[3] = vec2[](
                vec2(-1.0, -1.0),
                vec2(3.0, -1.0),
                vec2(-1.0, 3.0)
            );

            void main() {
                gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
            }
        ''',
        fragment_shader='''
            #version 330

            uniform sampler2D Texture;

            layout (location = 0) out vec4 out_color;

            #include "kernel"

            void main() {
                vec3 color = vec3(0.0, 0.0, 0.0);
                for (int i = 0; i < N; ++i) {
                    color += texelFetch(Texture, ivec2(gl_FragCoord.xy) + ivec2(0, i - N / 2), 0).rgb * coeff[i];
                }
                out_color = vec4(color, 1.0);
            }
        ''',
        layout=[
            {
                'name': 'Texture',
                'binding': 0,
            },
        ],
        resources=[
            {
                'type': 'sampler',
                'binding': 0,
                'image': temp,
            },
        ],
        framebuffer=[output],
        topology='triangles',
        vertex_count=3,
    )


    def render():
        ctx.reset()
        camera = zengl.camera((-85, -180, 140), (0.0, 0.0, 65.0), aspect=1.0, fov=45.0)
        uniform_buffer.write(camera)
        image.clear()
        depth.clear()
        monkey.render()
        blur_x.render()
        blur_y.render()

    texture = zengl.inspect(output)['texture']
    return (*size, texture, render)


class CrateExample(Example):
    title = "Crate"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330

                uniform mat4 Mvp;

                in vec3 in_position;
                in vec3 in_normal;
                in vec2 in_texcoord_0;

                out vec3 v_vert;
                out vec3 v_norm;
                out vec2 v_text;

                void main() {
                    gl_Position = Mvp * vec4(in_position, 1.0);
                    v_vert = in_position;
                    v_norm = in_normal;
                    v_text = in_texcoord_0;
                }
            ''',
            fragment_shader='''
                #version 330

                uniform vec3 Light;
                uniform sampler2D Texture;

                in vec3 v_vert;
                in vec3 v_norm;
                in vec2 v_text;

                out vec4 f_color;

                void main() {
                    float lum = clamp(dot(normalize(Light - v_vert), normalize(v_norm)), 0.0, 1.0) * 0.8 + 0.2;
                    f_color = vec4(texture(Texture, v_text).rgb * lum, 1.0);
                }
            ''',
        )

        self.mvp = self.prog['Mvp']
        self.light = self.prog['Light']

        self.scene = self.load_scene('crate.obj')
        self.vao = self.scene.root_nodes[0].mesh.vao.instance(self.prog)

        width, height, texture_obj, self.render_to_texture = zengl_pipeline()
        self.texture = self.ctx.external_texture(texture_obj, (width, height), 4, 0, 'f1')
        self.viewport = self.ctx.viewport

    def render(self, time, frame_time):
        self.render_to_texture()

        angle = time
        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.viewport = self.viewport
        self.ctx.enable(moderngl.DEPTH_TEST)

        camera_pos = (np.cos(angle) * 3.0, np.sin(angle) * 3.0, 2.0)

        proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 100.0)
        lookat = Matrix44.look_at(
            camera_pos,
            (0.0, 0.0, 0.5),
            (0.0, 0.0, 1.0),
        )

        self.mvp.write((proj * lookat).astype('f4'))
        self.light.value = camera_pos
        self.texture.use()
        self.vao.render()


if __name__ == '__main__':
    CrateExample.run()

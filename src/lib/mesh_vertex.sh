#version 120

attribute vec3 vertex;
attribute vec3 color;
uniform mat4 PVM;

varying vec3 v_color;

void main() {
    v_color = color;
    gl_Position = PVM * vec4(vertex, 1.0);
}

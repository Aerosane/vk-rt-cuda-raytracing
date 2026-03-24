#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 hitColor;

void main() {
    // Sky gradient on miss
    hitColor = vec3(0.1, 0.1, 0.3);
}

import time
import numpy as np
import pygame
import moderngl
import librosa
from pathlib import Path

WIDTH = 1280
HEIGHT = 720
FPS = 60

AUDIO_DIR = Path("audio")

VERT_SHADER = """
#version 330
in vec2 in_vert;
out vec2 uv;

void main() {
    uv = in_vert * 0.5 + 0.5;
    gl_Position = vec4(in_vert,0.0,1.0);
}
"""

FRAG_SHADER = """
#version 330

uniform vec2 resolution;
uniform float time;
uniform float bass;
uniform float mid;
uniform float treble;
uniform float rms;

in vec2 uv;
out vec4 fragColor;

float hash(vec2 p){
    p = fract(p*vec2(234.34,435.345));
    p += dot(p,p+34.23);
    return fract(p.x*p.y);
}

float noise(vec2 p){
    vec2 i = floor(p);
    vec2 f = fract(p);

    float a = hash(i);
    float b = hash(i+vec2(1,0));
    float c = hash(i+vec2(0,1));
    float d = hash(i+vec2(1,1));

    vec2 u = f*f*(3.0-2.0*f);

    return mix(a,b,u.x) +
           (c-a)*u.y*(1.0-u.x) +
           (d-b)*u.x*u.y;
}

void main(){

    vec2 p = uv*2.0-1.0;
    p.x *= resolution.x/resolution.y;

    float r = length(p);
    float a = atan(p.y,p.x);

    float wave = sin(a*8.0 + time*1.5 + mid*6.0);
    float ring = smoothstep(0.35 + bass*0.2,0.0,abs(r-(0.35+wave*0.05)));

    float fog = noise(p*3.0 + time*0.1);
    float swirl = sin(a*4.0 + fog*3.0 + time);

    vec3 col = vec3(0.0);

    col += vec3(0.6,0.3,1.0)*ring*(0.4+bass*1.5);
    col += vec3(0.1,0.4,1.0)*(fog*0.4+mid);
    col += vec3(1.0,0.9,0.6)*pow(max(0.0,swirl),4.0)*treble;

    col *= 1.0 + rms*1.5;

    float vignette = smoothstep(1.4,0.3,r);
    col *= vignette;

    fragColor = vec4(col,1.0);
}
"""


def analyze_audio(path):

    y, sr = librosa.load(path, sr=None)

    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    bass_band = (freqs < 200)
    mid_band = (freqs >= 200) & (freqs < 2000)
    treble_band = (freqs >= 2000)

    bass = stft[bass_band].mean(axis=0)
    mid = stft[mid_band].mean(axis=0)
    treble = stft[treble_band].mean(axis=0)

    rms = librosa.feature.rms(y=y)[0]

    length = min(len(bass),len(rms))

    bass = bass[:length]
    mid = mid[:length]
    treble = treble[:length]
    rms = rms[:length]

    bass /= bass.max()
    mid /= mid.max()
    treble /= treble.max()
    rms /= rms.max()

    return bass,mid,treble,rms


def main():

    audio_files = list(AUDIO_DIR.glob("*"))
    if not audio_files:
        print("No audio files in /audio folder")
        return

    track = audio_files[0]

    print("Analyzing:",track.name)

    bass,mid,treble,rms = analyze_audio(track)

    pygame.init()
    pygame.mixer.init()

    pygame.display.set_mode((WIDTH,HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)

    ctx = moderngl.create_context()
    prog = ctx.program(vertex_shader=VERT_SHADER, fragment_shader=FRAG_SHADER)

    vertices = np.array([
        -1,-1,
        1,-1,
        -1,1,
        1,1
    ],dtype="f4")

    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.simple_vertex_array(prog,vbo,"in_vert")

    prog["resolution"].value = (WIDTH,HEIGHT)

    pygame.mixer.music.load(track)
    pygame.mixer.music.play()

    start = time.time()

    clock = pygame.time.Clock()

    frame = 0
    running = True

    while running:

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running=False

        frame = min(frame+1,len(bass)-1)

        prog["time"].value = time.time()-start
        prog["bass"].value = float(bass[frame])
        prog["mid"].value = float(mid[frame])
        prog["treble"].value = float(treble[frame])
        prog["rms"].value = float(rms[frame])

        ctx.clear(0,0,0)
        vao.render(moderngl.TRIANGLE_STRIP)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
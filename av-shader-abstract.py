import time
from pathlib import Path

import librosa
import moderngl
import numpy as np
import pygame


WIDTH = 1280
HEIGHT = 720
FPS = 60

AUDIO_DIR = Path("audio")
SUPPORTED_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 64


def list_audio_files(audio_dir: Path) -> list[Path]:
    if not audio_dir.exists():
        raise FileNotFoundError(f"Missing folder: {audio_dir.resolve()}")
    files = [p for p in audio_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS]
    files.sort()
    if not files:
        raise FileNotFoundError(f"No supported audio files found in {audio_dir.resolve()}")
    return files


def normalize_feature(x: np.ndarray, power: float = 1.0) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32))
    lo = np.percentile(x, 5)
    hi = np.percentile(x, 95)
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return np.power(y, power).astype(np.float32)


def smooth_feature(x: np.ndarray, alpha: float = 0.18) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    if len(x) == 0:
        return out
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = out[i - 1] * (1.0 - alpha) + x[i] * alpha
    return out


def analyze_audio(path: Path, fps: int = FPS) -> dict:
    print(f"Analyzing: {path.name}")
    y, sr = librosa.load(path.as_posix(), sr=None, mono=True)
    if y.size == 0:
        raise ValueError("Audio file appears empty.")

    duration = librosa.get_duration(y=y, sr=sr)

    stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    power_spec = stft ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    spectral_centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)

    bass_mask = (freqs >= 20) & (freqs < 180)
    mid_mask = (freqs >= 180) & (freqs < 2200)
    treble_mask = (freqs >= 2200) & (freqs < 10000)

    bass = power_spec[bass_mask].mean(axis=0) if np.any(bass_mask) else np.zeros(stft.shape[1])
    mid = power_spec[mid_mask].mean(axis=0) if np.any(mid_mask) else np.zeros(stft.shape[1])
    treble = power_spec[treble_mask].mean(axis=0) if np.any(treble_mask) else np.zeros(stft.shape[1])

    mel = librosa.feature.melspectrogram(
        S=power_spec,
        sr=sr,
        n_mels=N_MELS,
        fmax=min(sr // 2, 16000),
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    frame_times = librosa.frames_to_time(np.arange(stft.shape[1]), sr=sr, hop_length=HOP_LENGTH)
    mel_times = librosa.frames_to_time(np.arange(mel_db.shape[1]), sr=sr, hop_length=HOP_LENGTH)

    target_count = max(1, int(duration * fps))
    target_times = np.linspace(0.0, duration, target_count, endpoint=False)

    def interp_feature(values: np.ndarray) -> np.ndarray:
        return np.interp(target_times, frame_times, values).astype(np.float32)

    rms_f = smooth_feature(normalize_feature(interp_feature(rms), power=0.85), alpha=0.22)
    bass_f = smooth_feature(normalize_feature(interp_feature(bass), power=0.70), alpha=0.18)
    mid_f = smooth_feature(normalize_feature(interp_feature(mid), power=0.85), alpha=0.18)
    treble_f = smooth_feature(normalize_feature(interp_feature(treble), power=0.95), alpha=0.20)
    centroid_f = smooth_feature(normalize_feature(interp_feature(spectral_centroid), power=1.0), alpha=0.18)
    onset_f = smooth_feature(normalize_feature(interp_feature(onset_env), power=0.60), alpha=0.28)

    spectrum_rows = []
    for i in range(N_MELS):
        band = np.interp(target_times, mel_times, mel_db[i]).astype(np.float32)
        band = normalize_feature(band, power=0.8)
        spectrum_rows.append(band)
    spectrum = np.stack(spectrum_rows, axis=1).astype(np.float32)

    return {
        "duration": duration,
        "times": target_times,
        "rms": rms_f,
        "bass": bass_f,
        "mid": mid_f,
        "treble": treble_f,
        "centroid": centroid_f,
        "onset": onset_f,
        "spectrum": spectrum,
    }


def get_frame_index(playback_seconds: float, total_frames: int) -> int:
    idx = int(playback_seconds * FPS)
    return max(0, min(total_frames - 1, idx))


def set_uniform_safe(program: moderngl.Program, name: str, value):
    if name in program:
        program[name].value = value


VERTEX_SHADER = """
#version 330

in vec2 in_vert;
out vec2 v_uv;

void main() {
    v_uv = in_vert * 0.5 + 0.5;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

SCENE_FRAGMENT_SHADER = """
#version 330

uniform vec2 u_resolution;
uniform float u_time;
uniform float u_rms;
uniform float u_bass;
uniform float u_mid;
uniform float u_treble;
uniform float u_centroid;
uniform float u_onset;
uniform float u_progress;

uniform sampler2D u_prev_frame;
uniform sampler2D u_spectrum_tex;

in vec2 v_uv;
out vec4 fragColor;

const float PI = 3.14159265359;

float hash(vec2 p) {
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.23);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);

    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x)
         + (c - a) * u.y * (1.0 - u.x)
         + (d - b) * u.x * u.y;
}

float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    mat2 m = mat2(1.6, 1.2, -1.2, 1.6);
    for (int i = 0; i < 5; i++) {
        v += a * noise(p);
        p = m * p;
        a *= 0.5;
    }
    return v;
}

vec2 rotate2d(vec2 p, float a) {
    float s = sin(a);
    float c = cos(a);
    return mat2(c, -s, s, c) * p;
}

float sdCircle(vec2 p, float r) {
    return length(p) - r;
}

float sdBox(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

float sdDiamond(vec2 p, float s) {
    p = abs(p);
    return (p.x + p.y - s) * 0.70710678;
}

float stroke(float d, float w) {
    return smoothstep(w, 0.0, abs(d));
}

float fillShape(float d) {
    return smoothstep(0.0, -0.01, d);
}

float spectrumSample(float x) {
    return texture(u_spectrum_tex, vec2(clamp(x, 0.0, 1.0), 0.5)).r;
}

vec3 paletteA(float t) {
    vec3 a = vec3(0.05, 0.04, 0.09);
    vec3 b = vec3(0.30, 0.23, 0.52);
    vec3 c = vec3(0.90, 0.55, 0.25);
    vec3 d = vec3(0.20, 0.60, 0.95);
    return a + b * cos(6.28318 * (c * t + d));
}

vec3 paletteB(float t) {
    vec3 a = vec3(0.08, 0.02, 0.03);
    vec3 b = vec3(0.85, 0.12, 0.28);
    vec3 c = vec3(0.95, 0.82, 0.18);
    vec3 d = vec3(0.10, 0.12, 0.65);
    return a + 0.35 * cos(6.28318 * (c * t + d));
}

vec3 renderField(vec2 uv) {
    vec2 p = uv * 2.0 - 1.0;
    p.x *= u_resolution.x / u_resolution.y;

    float t = u_time;
    float rms = u_rms;
    float bass = u_bass;
    float mid = u_mid;
    float treble = u_treble;
    float onset = u_onset;
    float centroidVal = u_centroid;

    float r = length(p);
    float ang = atan(p.y, p.x);

    float specArc = spectrumSample((ang + PI) / (2.0 * PI));
    float specRad = spectrumSample(clamp(r, 0.0, 1.0));
    float specPulse = spectrumSample(fract(ang * 0.1 + r * 0.75));

    vec2 q = p;
    q = rotate2d(q, t * 0.08 + bass * 0.22);
    q.x = abs(q.x);

    float field = fbm(p * (2.0 + mid * 2.0) + vec2(t * 0.08, -t * 0.05));
    float field2 = fbm(rotate2d(p, 0.7) * (4.0 + treble * 4.0) - vec2(t * 0.10, t * 0.04));

    float beat = smoothstep(0.18, 0.95, onset * 2.1);
    float pulse = 1.0 + bass * 0.35 + beat * 0.7 + rms * 0.45;

    float outerRing = stroke(sdCircle(p, 0.62 + bass * 0.06 + sin(t * 0.6) * 0.01), 0.012 + specRad * 0.030);
    float innerRing = stroke(sdCircle(p, 0.36 + sin(t * 0.9 + specArc * 5.0) * 0.015), 0.010 + mid * 0.018);
    float coreRing = stroke(sdCircle(p, 0.14 + bass * 0.05), 0.018 + beat * 0.03);

    vec2 shardP = rotate2d(q, PI * 0.25);
    float shard1 = fillShape(max(sdDiamond(shardP - vec2(0.0, 0.26), 0.24 + specArc * 0.10), -sdCircle(shardP, 0.85)));
    float shard2 = fillShape(sdDiamond(rotate2d(q, -PI * 0.12) - vec2(0.0, -0.12), 0.18 + treble * 0.08));
    float shardCut = fillShape(-sdBox(rotate2d(q, 0.18), vec2(0.08 + bass * 0.06, 0.90)));
    float shards = max(shard1 * (1.0 - shardCut), shard2 * 0.65);

    float spine = 1.0 - smoothstep(0.012, 0.035 + bass * 0.03, abs(p.x));
    float spineCuts = 1.0 - fillShape(sdBox(p - vec2(0.0, 0.52), vec2(0.03, 0.11)));
    spine *= spineCuts;

    float halo = 0.018 / (r * r + 0.025);

    vec2 orbP1 = p - vec2(cos(t * 0.55 + specArc * 4.0), sin(t * 0.55 + specArc * 4.0)) * (0.24 + bass * 0.12);
    vec2 orbP2 = p - vec2(cos(-t * 0.42 + 2.2), sin(-t * 0.42 + 2.2)) * (0.44 + mid * 0.08);
    float orbit1 = stroke(sdCircle(orbP1, 0.04 + specPulse * 0.03), 0.014);
    float orbit2 = fillShape(sdCircle(orbP2, 0.05 + treble * 0.02));

    float crossA = stroke(sdSegment(p, vec2(-0.42, 0.0), vec2(0.42, 0.0)), 0.010 + specArc * 0.02);
    float crossB = stroke(sdSegment(p, vec2(0.0, -0.55), vec2(0.0, 0.55)), 0.008 + bass * 0.02);

    float radialBands = smoothstep(0.78, 0.80, sin(ang * (8.0 + centroidVal * 8.0) + t * 0.7 + specArc * 8.0));
    radialBands *= smoothstep(0.85, 0.25, r);

    float maskFace = fillShape(sdDiamond(p * vec2(1.0, 1.2), 0.78));
    float eyeL = 1.0 - fillShape(sdCircle(p - vec2(-0.18, 0.06), 0.08 + treble * 0.02));
    float eyeR = 1.0 - fillShape(sdCircle(p - vec2(0.18, 0.06), 0.08 + treble * 0.02));
    float mouth = stroke(sdSegment(p, vec2(-0.14, -0.18), vec2(0.14, -0.18)), 0.018 + bass * 0.02);
    float faceGhost = maskFace * eyeL * eyeR * mouth * 0.28;

    float textureCuts = smoothstep(0.45, 0.85, field + field2 * 0.35 + specRad * 0.3);
    float darkCuts = smoothstep(0.62, 0.90, field2 + specPulse * 0.6);

    vec3 base = paletteA(0.08 + u_progress * 0.1 + field * 0.15);
    vec3 hot = paletteB(0.20 + specArc * 0.18 + beat * 0.08);
    vec3 cold = vec3(0.12, 0.45, 0.95);
    vec3 warm = vec3(1.0, 0.55, 0.20);
    vec3 ivory = vec3(0.96, 0.92, 0.86);
    vec3 voidTone = vec3(0.02, 0.01, 0.04);

    vec3 color = vec3(0.0);

    color += base * (0.18 + halo * 2.2);
    color += warm * outerRing * (0.8 + bass * 1.4);
    color += cold * innerRing * (0.9 + mid * 1.2);
    color += ivory * coreRing * (0.5 + beat * 1.2);
    color += mix(cold, warm, specArc) * shards * (0.55 + pulse);
    color += vec3(0.85, 0.15, 0.25) * spine * (0.35 + bass * 1.4);
    color += ivory * orbit1 * (0.25 + treble * 1.0);
    color += warm * orbit2 * (0.18 + bass * 0.7);
    color += mix(warm, cold, specPulse) * crossA * (0.16 + mid * 0.75);
    color += mix(ivory, warm, 0.4) * crossB * (0.14 + bass * 0.9);
    color += mix(cold, warm, field) * radialBands * (0.2 + treble * 0.8);
    color += vec3(0.45, 0.18, 0.72) * faceGhost * (0.6 + beat * 0.8);

    color *= mix(1.0, 0.55, darkCuts * 0.6);
    color += textureCuts * 0.08 * mix(cold, warm, specArc);

    float vignette = smoothstep(1.35, 0.20, r);
    color *= vignette;

    color = max(color, vec3(0.0));
    color += halo * 0.15 * ivory;

    return color;
}

void main() {
    vec2 uv = v_uv;

    vec2 prevUv = uv;
    prevUv += (uv - 0.5) * (-0.003 - u_bass * 0.006);
    prevUv += vec2(
        sin(u_time * 0.21 + uv.y * 6.0),
        cos(u_time * 0.19 + uv.x * 5.0)
    ) * 0.0012 * (0.4 + u_mid);

    vec3 prev = texture(u_prev_frame, prevUv).rgb;
    vec3 current = renderField(uv);

    float feedback = 0.94 - u_onset * 0.06;
    vec3 color = current + prev * feedback * 0.65;

    fragColor = vec4(color, 1.0);
}
"""

POST_FRAGMENT_SHADER = """
#version 330

uniform vec2 u_resolution;
uniform sampler2D u_scene_tex;
uniform float u_rms;
uniform float u_bass;
uniform float u_onset;

in vec2 v_uv;
out vec4 fragColor;

vec3 tone_map(vec3 x) {
    return x / (1.0 + x);
}

void main() {
    vec2 uv = v_uv;
    vec2 px = 1.0 / u_resolution;

    vec3 c = texture(u_scene_tex, uv).rgb;

    vec3 blur = vec3(0.0);
    blur += texture(u_scene_tex, uv + px * vec2( 2.0,  0.0)).rgb;
    blur += texture(u_scene_tex, uv + px * vec2(-2.0,  0.0)).rgb;
    blur += texture(u_scene_tex, uv + px * vec2( 0.0,  2.0)).rgb;
    blur += texture(u_scene_tex, uv + px * vec2( 0.0, -2.0)).rgb;
    blur += texture(u_scene_tex, uv + px * vec2( 2.0,  2.0)).rgb;
    blur += texture(u_scene_tex, uv + px * vec2(-2.0,  2.0)).rgb;
    blur += texture(u_scene_tex, uv + px * vec2( 2.0, -2.0)).rgb;
    blur += texture(u_scene_tex, uv + px * vec2(-2.0, -2.0)).rgb;
    blur *= 0.125;

    float bloom = 0.08 + u_rms * 0.18 + u_onset * 0.12;
    vec3 color = c + blur * bloom;

    float d = distance(uv, vec2(0.5));
    float vignette = smoothstep(0.95, 0.18, d);
    color *= vignette;

    color *= 1.0 + u_bass * 0.08;
    color = tone_map(color);
    color = pow(color, vec3(0.94));

    fragColor = vec4(color, 1.0);
}
"""


def main():
    audio_files = list_audio_files(AUDIO_DIR)
    current_index = 0

    pygame.init()
    pygame.mixer.init()
    pygame.display.set_caption("Abstract Totemic Audio Shader")
    pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)

    ctx = moderngl.create_context()
    ctx.enable(moderngl.BLEND)

    quad = np.array(
        [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        dtype="f4",
    )
    vbo = ctx.buffer(quad.tobytes())

    scene_prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=SCENE_FRAGMENT_SHADER)
    post_prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=POST_FRAGMENT_SHADER)

    scene_vao = ctx.simple_vertex_array(scene_prog, vbo, "in_vert")
    post_vao = ctx.simple_vertex_array(post_prog, vbo, "in_vert")

    set_uniform_safe(scene_prog, "u_resolution", (WIDTH, HEIGHT))
    set_uniform_safe(post_prog, "u_resolution", (WIDTH, HEIGHT))

    tex_a = ctx.texture((WIDTH, HEIGHT), 4, dtype="f1")
    tex_b = ctx.texture((WIDTH, HEIGHT), 4, dtype="f1")
    tex_a.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex_b.filter = (moderngl.LINEAR, moderngl.LINEAR)

    fbo_a = ctx.framebuffer(color_attachments=[tex_a])
    fbo_b = ctx.framebuffer(color_attachments=[tex_b])

    zero_data = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    tex_a.write(zero_data.tobytes())
    tex_b.write(zero_data.tobytes())

    spectrum_tex = ctx.texture((N_MELS, 1), 1, dtype="f4")
    spectrum_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    spectrum_tex.repeat_x = False
    spectrum_tex.repeat_y = False

    current_audio = audio_files[current_index]
    features = analyze_audio(current_audio)

    pygame.mixer.music.load(current_audio.as_posix())
    pygame.mixer.music.play()

    start_wall_time = time.time()
    paused = False
    pause_started = 0.0
    paused_accum = 0.0

    clock = pygame.time.Clock()
    ping = True

    print("Controls:")
    print("  SPACE = pause/resume")
    print("  RIGHT = next track")
    print("  LEFT  = previous track")
    print("  ESC   = quit")
    print(f"Playing: {current_audio.name}")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    if paused:
                        pygame.mixer.music.unpause()
                        paused = False
                        paused_accum += time.time() - pause_started
                    else:
                        pygame.mixer.music.pause()
                        paused = True
                        pause_started = time.time()

                elif event.key in (pygame.K_RIGHT, pygame.K_LEFT):
                    if event.key == pygame.K_RIGHT:
                        current_index = (current_index + 1) % len(audio_files)
                    else:
                        current_index = (current_index - 1) % len(audio_files)

                    current_audio = audio_files[current_index]
                    features = analyze_audio(current_audio)
                    pygame.mixer.music.load(current_audio.as_posix())
                    pygame.mixer.music.play()

                    start_wall_time = time.time()
                    paused = False
                    pause_started = 0.0
                    paused_accum = 0.0

                    tex_a.write(zero_data.tobytes())
                    tex_b.write(zero_data.tobytes())

                    print(f"Playing: {current_audio.name}")

        if paused:
            playback_time = max(0.0, pause_started - start_wall_time - paused_accum)
        else:
            playback_time = max(0.0, time.time() - start_wall_time - paused_accum)

        if playback_time >= features["duration"] and not paused:
            current_index = (current_index + 1) % len(audio_files)
            current_audio = audio_files[current_index]
            features = analyze_audio(current_audio)
            pygame.mixer.music.load(current_audio.as_posix())
            pygame.mixer.music.play()

            start_wall_time = time.time()
            paused = False
            pause_started = 0.0
            paused_accum = 0.0

            tex_a.write(zero_data.tobytes())
            tex_b.write(zero_data.tobytes())

            print(f"Playing: {current_audio.name}")
            playback_time = 0.0

        idx = get_frame_index(playback_time, len(features["times"]))

        rms = float(features["rms"][idx])
        bass = float(features["bass"][idx])
        mid = float(features["mid"][idx])
        treble = float(features["treble"][idx])
        centroid_val = float(features["centroid"][idx])
        onset = float(features["onset"][idx])
        progress = float(playback_time / max(features["duration"], 1e-6))

        spectrum_row = np.clip(features["spectrum"][idx], 0.0, 1.0).astype("f4")
        spectrum_tex.write(spectrum_row.tobytes())

        read_tex = tex_a if ping else tex_b
        write_fbo = fbo_b if ping else fbo_a

        read_tex.use(location=0)
        spectrum_tex.use(location=1)

        set_uniform_safe(scene_prog, "u_prev_frame", 0)
        set_uniform_safe(scene_prog, "u_spectrum_tex", 1)
        set_uniform_safe(scene_prog, "u_time", playback_time)
        set_uniform_safe(scene_prog, "u_rms", rms)
        set_uniform_safe(scene_prog, "u_bass", bass)
        set_uniform_safe(scene_prog, "u_mid", mid)
        set_uniform_safe(scene_prog, "u_treble", treble)
        set_uniform_safe(scene_prog, "u_centroid", centroid_val)
        set_uniform_safe(scene_prog, "u_onset", onset)
        set_uniform_safe(scene_prog, "u_progress", progress)

        write_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        scene_vao.render(moderngl.TRIANGLE_STRIP)

        screen_tex = tex_b if ping else tex_a
        screen_tex.use(location=0)

        set_uniform_safe(post_prog, "u_scene_tex", 0)
        set_uniform_safe(post_prog, "u_rms", rms)
        set_uniform_safe(post_prog, "u_bass", bass)
        set_uniform_safe(post_prog, "u_onset", onset)

        ctx.screen.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        post_vao.render(moderngl.TRIANGLE_STRIP)

        pygame.display.flip()
        clock.tick(FPS)
        ping = not ping

    pygame.mixer.music.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
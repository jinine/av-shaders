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
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 78.233);
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
    mat2 m = mat2(1.7, 1.2, -1.2, 1.7);

    for (int i = 0; i < 6; i++) {
        v += a * noise(p);
        p = m * p;
        a *= 0.5;
    }
    return v;
}

vec3 palette(float t) {
    vec3 a = vec3(0.06, 0.05, 0.10);
    vec3 b = vec3(0.28, 0.20, 0.44);
    vec3 c = vec3(0.82, 0.52, 0.36);
    vec3 d = vec3(0.12, 0.26, 0.74);
    return a + b * cos(6.28318 * (c * t + d));
}

float spectrum_sample(float x) {
    return texture(u_spectrum_tex, vec2(clamp(x, 0.0, 1.0), 0.5)).r;
}

vec3 render_field(vec2 uv) {
    vec2 p = uv * 2.0 - 1.0;
    p.x *= u_resolution.x / u_resolution.y;

    float t = u_time;
    float r = length(p);
    float ang = atan(p.y, p.x);

    float bass_val = u_bass;
    float mid_val = u_mid;
    float treble_val = u_treble;
    float rms_val = u_rms;
    float onset_val = u_onset;
    float centroid_val = u_centroid;

    float spectral_arc = spectrum_sample((ang + PI) / (2.0 * PI));
    float spectral_radial = spectrum_sample(clamp(r, 0.0, 1.0));
    float spectral_fine = spectrum_sample(fract(ang * 0.159 + r * 0.7 + t * 0.02));

    vec2 flow = p;
    flow += 0.12 * vec2(
        sin(t * 0.45 + p.y * (4.0 + mid_val * 4.0)),
        cos(t * 0.35 + p.x * (5.0 + treble_val * 5.0))
    ) * (0.25 + bass_val * 1.25);

    float n1 = fbm(flow * (2.2 + mid_val * 3.5) + t * 0.05);
    float n2 = fbm(flow * (5.0 + treble_val * 4.5) - t * 0.12 + n1 * 1.8);
    float n3 = fbm(flow * 10.0 + vec2(n1, n2) * 2.2);

    float spiral = sin(
        ang * (5.0 + centroid_val * 8.0)
        - r * (12.0 + bass_val * 10.0)
        + t * (0.8 + mid_val * 0.8)
        + n2 * 6.0
        + spectral_arc * 7.0
    );

    float ring_radius = 0.30 + 0.10 * spiral + bass_val * 0.14;
    float ring = smoothstep(0.18 + bass_val * 0.20, 0.0, abs(r - ring_radius));

    float core = smoothstep(0.28 + bass_val * 0.10, 0.0, r);
    float halo = 0.025 / (r * r + 0.03);

    float mist = smoothstep(0.12, 0.98, n1 * 0.8 + n2 * 0.35 + spectral_radial * 0.65);
    float strands = smoothstep(0.45, 0.92, n2 + spectral_fine * 0.9);
    float veins = pow(max(0.0, sin(ang * 10.0 + t * 1.6 + spectral_arc * 8.0 + n3 * 3.0)), 6.0);

    float beat_flash = smoothstep(0.18, 0.95, onset_val * 2.0);
    float pulse = 1.0 + beat_flash * 0.8 + bass_val * 0.35 + rms_val * 0.5;

    vec3 base = palette(0.12 + n1 * 0.22 + u_progress * 0.10);
    vec3 high = palette(0.42 + n2 * 0.16 + spectral_arc * 0.18);
    vec3 hot  = palette(0.78 + spectral_fine * 0.22 + treble_val * 0.12);

    vec3 color = vec3(0.0);
    color += base * mist * (0.65 + mid_val * 0.95);
    color += high * strands * (0.30 + spectral_arc * 1.25);
    color += hot * ring * pulse * 1.5;
    color += vec3(1.0, 0.95, 0.90) * veins * (0.08 + treble_val * 0.60);
    color += vec3(0.95, 0.97, 1.0) * halo * (0.20 + bass_val * 0.85);
    color += core * vec3(0.12, 0.16, 0.22) * pulse;

    float stars = pow(max(0.0, noise(uv * 45.0 + t * 0.012) - 0.84), 5.0);
    color += vec3(1.0, 0.96, 0.92) * stars * (0.08 + treble_val * 0.75);

    return color;
}

void main() {
    vec2 uv = v_uv;

    vec2 prev_uv = uv;
    prev_uv += (uv - 0.5) * (-0.004 - u_bass * 0.008);
    prev_uv += vec2(
        sin(u_time * 0.25 + uv.y * 10.0),
        cos(u_time * 0.23 + uv.x * 8.0)
    ) * 0.0015 * (0.5 + u_mid);

    vec3 prev = texture(u_prev_frame, prev_uv).rgb;
    vec3 current = render_field(uv);

    float feedback = 0.955 - u_onset * 0.08;
    vec3 mixed_color = current + prev * feedback;

    fragColor = vec4(mixed_color, 1.0);
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

    float bloom_amt = 0.10 + u_rms * 0.20 + u_onset * 0.15;
    vec3 color = c + blur * bloom_amt;

    float d = distance(uv, vec2(0.5));
    float vignette = smoothstep(0.92, 0.22, d);
    color *= vignette;

    color *= 1.0 + u_bass * 0.10;
    color = tone_map(color);
    color = pow(color, vec3(0.95));

    fragColor = vec4(color, 1.0);
}
"""


def main():
    audio_files = list_audio_files(AUDIO_DIR)
    current_index = 0

    pygame.init()
    pygame.mixer.init()
    pygame.display.set_caption("Cinematic Audio Shader")
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
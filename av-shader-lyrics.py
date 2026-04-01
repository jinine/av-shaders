import time
import json
import argparse
from pathlib import Path
import numpy as np
import pygame
import moderngl
import librosa
from dataclasses import dataclass

WIDTH = 1280
HEIGHT = 720
FPS = 60

AUDIO_DIR = Path("audio")
SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

N_FFT = 2048
HOP_LENGTH = 512

@dataclass
class Lyric:
    """Represents a single lyric line with timing"""
    text: str
    start_time: float
    duration: float
    
    @property
    def end_time(self) -> float:
        return self.start_time + self.duration


def list_audio_files(audio_dir: Path) -> list[Path]:
    if not audio_dir.exists():
        raise FileNotFoundError(f"Missing folder: {audio_dir.resolve()}")
    files = [p for p in audio_dir.iterdir() if p.suffix.lower() in SUPPORTED_AUDIO_EXTS]
    files.sort()
    if not files:
        raise FileNotFoundError(f"No supported audio files found in {audio_dir.resolve()}")
    return files


def load_lyrics(audio_path: Path) -> list[Lyric]:
    """Load lyrics from JSON file alongside audio file"""
    lyrics_path = audio_path.with_suffix(".json")
    if not lyrics_path.exists():
        print(f"⚠️  No lyrics file found: {lyrics_path.name}")
        print("   Create a JSON file with this format:")
        print("   {'lyrics': [{'text': 'line 1', 'start': 0.0, 'duration': 2.0}, ...]}")
        return []
    
    try:
        with open(lyrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        lyrics = [Lyric(
            text=item["text"],
            start_time=item["start"],
            duration=item["duration"]
        ) for item in data.get("lyrics", [])]
        print(f"✓ Loaded {len(lyrics)} lyrics from {lyrics_path.name}")
        return lyrics
    except Exception as e:
        print(f"✗ Error loading lyrics: {e}")
        return []


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

    bass_mask = (freqs >= 20) & (freqs < 180)
    mid_mask = (freqs >= 180) & (freqs < 2200)
    treble_mask = (freqs >= 2200) & (freqs < 10000)

    bass = power_spec[bass_mask].mean(axis=0) if np.any(bass_mask) else np.zeros(stft.shape[1])
    mid = power_spec[mid_mask].mean(axis=0) if np.any(mid_mask) else np.zeros(stft.shape[1])
    treble = power_spec[treble_mask].mean(axis=0) if np.any(treble_mask) else np.zeros(stft.shape[1])

    frame_times = librosa.frames_to_time(np.arange(stft.shape[1]), sr=sr, hop_length=HOP_LENGTH)
    target_count = max(1, int(duration * fps))
    target_times = np.linspace(0.0, duration, target_count, endpoint=False)

    def interp_feature(values: np.ndarray) -> np.ndarray:
        return np.interp(target_times, frame_times, values).astype(np.float32)

    rms_f = smooth_feature(normalize_feature(interp_feature(rms), power=0.85), alpha=0.22)
    bass_f = smooth_feature(normalize_feature(interp_feature(bass), power=0.70), alpha=0.18)
    mid_f = smooth_feature(normalize_feature(interp_feature(mid), power=0.85), alpha=0.18)
    treble_f = smooth_feature(normalize_feature(interp_feature(treble), power=0.95), alpha=0.20)

    return {
        "duration": duration,
        "times": target_times,
        "rms": rms_f,
        "bass": bass_f,
        "mid": mid_f,
        "treble": treble_f,
    }


def get_frame_index(playback_seconds: float, total_frames: int) -> int:
    idx = int(playback_seconds * FPS)
    return max(0, min(total_frames - 1, idx))


VERTEX_SHADER = """
#version 330
in vec2 in_vert;
out vec2 uv;

void main() {
    uv = in_vert * 0.5 + 0.5;
    gl_Position = vec4(in_vert, 0.0, 1.0);
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
uniform float lyric_opacity;

in vec2 uv;
out vec4 fragColor;

// Hash function for randomness
float hash(vec2 p){
    p = fract(p * vec2(234.34, 435.345));
    p += dot(p, p + 34.23);
    return fract(p.x * p.y);
}

// Noise function
float noise(vec2 p){
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    
    float ab = mix(a, b, f.x);
    float cd = mix(c, d, f.x);
    return mix(ab, cd, f.y);
}

// Fractal Brownian Motion
float fbm(vec2 p){
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    
    for(int i = 0; i < 5; i++){
        value += amplitude * noise(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return value;
}

// Voronoi-like cellular pattern
float cellular(vec2 p){
    vec2 i = floor(p);
    vec2 f = fract(p);
    
    float minDist = 1.0;
    for(int y = -1; y <= 1; y++){
        for(int x = -1; x <= 1; x++){
            vec2 neighbor = vec2(float(x), float(y));
            vec2 cell = neighbor + hash(i + neighbor) * 0.5;
            float dist = length(f - cell);
            minDist = min(minDist, dist);
        }
    }
    return minDist;
}

void main() {
    // Aspect ratio correction
    vec2 aspect = vec2(resolution.x / resolution.y, 1.0);
    
    // Background with audio reactivity
    vec2 uv_centered = (uv - 0.5) * 2.0 * aspect;
    
    // Rugged glitch effect - more intense
    float glitch = sin(time * 3.0) * 0.5 + 0.5;
    float glitch_intensity = bass * 0.15 + treble * 0.1;
    
    // Distort UV based on multiple layers of noise and audio
    vec2 distorted_uv = uv;
    distorted_uv += fbm(uv * 3.0 + time * 0.3) * 0.03 * glitch_intensity;
    distorted_uv += sin(uv.y * 10.0 + time * 2.0) * 0.015 * bass;
    distorted_uv += cos(uv.x * 8.0 + time * 1.5) * 0.015 * mid;
    distorted_uv += noise(uv * 5.0 + time * 0.5) * 0.02 * treble;
    
    // Create multiple layers of color
    float layer1 = fbm(distorted_uv * 2.0 + time * 0.2);
    float layer2 = fbm(distorted_uv * 4.0 - time * 0.15);
    float layer3 = cellular(distorted_uv * 6.0 + time * 0.1);
    
    // Base color palette - field tech aesthetic
    vec3 dark_color = vec3(0.08, 0.04, 0.15);
    vec3 mid_color = vec3(0.15, 0.08, 0.25);
    vec3 bright_color = vec3(0.25, 0.12, 0.35);
    
    // Blend layers
    vec3 bg_color = mix(dark_color, mid_color, layer1);
    bg_color = mix(bg_color, bright_color, layer2 * 0.4);
    
    // Add cellular pattern overlay for tech feel
    bg_color += layer3 * 0.15 * vec3(0.1, 0.2, 0.3);
    
    // Add bass responsiveness (red/amber tint)
    bg_color += vec3(bass * 0.6, bass * 0.3, 0.0) * 0.8;
    
    // Add mid responsiveness (green tint)
    bg_color += vec3(0.0, mid * 0.35, 0.0) * 0.6;
    
    // Add treble responsiveness (cyan/blue tint)
    bg_color += vec3(0.0, treble * 0.35, treble * 0.5) * 0.8;
    
    // Vignette effect - stronger
    float vignette = length(uv_centered) * 0.5;
    vignette = 1.0 - smoothstep(0.4, 1.8, vignette);
    bg_color *= vignette * 0.85 + 0.15;
    
    // Add scanlines for retro glitch effect - more visible
    float scanlines = sin(uv.y * 300.0 + time) * 0.5 + 0.5;
    scanlines = mix(1.0, scanlines, 0.15);
    
    // Pulse effect synced to audio
    float pulse = sin(time * 6.28 * 2.0) * 0.5 + 0.5;
    float audio_pulse = rms * pulse;
    
    bg_color += audio_pulse * vec3(0.3, 0.15, 0.2);
    
    // Apply scanlines
    bg_color *= scanlines;
    
    // Multiple glitch bars with audio sync
    for(int i = 0; i < 3; i++){
        float bar_y = uv.y + float(i) * 0.2;
        float glitch_bar = hash(vec2(floor(bar_y * 20.0), time + float(i))) * glitch_intensity;
        if(mod(bar_y * 20.0, 4.0) < 1.5){
            float bar_color = glitch_bar * (0.7 + bass * 0.3);
            bg_color += vec3(1.0, 0.2, 0.5) * bar_color * 0.2;
        }
    }
    
    // Digital lines effect - horizontal
    float line_pattern = sin(uv.y * 50.0) * 0.5 + 0.5;
    line_pattern = smoothstep(0.4, 0.5, line_pattern);
    bg_color += line_pattern * vec3(0.0, 0.3, 0.5) * (0.3 + treble * 0.3);
    
    // Digital lines effect - vertical (audio reactive)
    float v_line_pattern = sin(uv.x * 40.0 + time * bass * 2.0) * 0.5 + 0.5;
    v_line_pattern = smoothstep(0.3, 0.5, v_line_pattern);
    bg_color += v_line_pattern * vec3(0.5, 0.0, 0.3) * (0.2 + mid * 0.4);
    
    // Lyric text area highlight - more prominent glow
    vec2 text_center = vec2(0.5, 0.5);
    float text_dist = distance(uv, text_center);
    
    // Create stronger glow around text area
    float text_glow = exp(-text_dist * 1.5) * lyric_opacity * 0.5;
    bg_color += vec3(0.2, 0.6, 1.0) * text_glow;
    
    // Add glow corona when lyric is visible
    if(lyric_opacity > 0.1){
        float corona = sin(text_dist * 10.0 - time * 3.0) * 0.5 + 0.5;
        corona *= exp(-text_dist * 1.2);
        bg_color += vec3(0.3, 0.8, 1.0) * corona * lyric_opacity * 0.3;
    }
    
    // Enhance colors a bit more
    bg_color = pow(bg_color, vec3(0.95));
    
    fragColor = vec4(bg_color, 1.0);
}
"""


def create_text_texture(text: str, font_size: int = 96) -> pygame.Surface:
    """Render text to a pygame surface with transparent background"""
    font = pygame.font.Font(None, font_size)
    text_surface = font.render(text, True, (255, 255, 255))
    # Create a transparent surface
    final_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT - 100))
    final_surface.blit(text_surface, text_rect)
    return final_surface


def render_text_to_texture(text: str, font_size: int = 96) -> pygame.Surface:
    """Render text to a pygame surface"""
    font = pygame.font.Font(None, font_size)
    surface = font.render(text, True, (255, 255, 255))
    return surface


def main(render_mode: bool = False):
    # Setup Pygame and ModernGL
    pygame.init()
    pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Audio Visualizer - Lyrics Mode")
    
    ctx = moderngl.create_context()
    ctx.enable(moderngl.BLEND)
    
    # Create quad for fullscreen rendering
    quad_buffer = ctx.buffer(data=np.array(
        [[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype='f4'
    ))
    
    program = ctx.program(
        vertex_shader=VERTEX_SHADER,
        fragment_shader=FRAG_SHADER,
    )
    
    vao = ctx.vertex_array(program, [(quad_buffer, '2f', 'in_vert')])
    
    # Create text shader for overlaying lyrics
    TEXT_VERTEX_SHADER = """
    #version 330
    in vec2 in_vert;
    out vec2 uv;
    
    void main() {
        uv = in_vert * 0.5 + 0.5;
        gl_Position = vec4(in_vert, 0.0, 1.0);
    }
    """
    
    TEXT_FRAG_SHADER = """
    #version 330
    uniform sampler2D tex;
    in vec2 uv;
    out vec4 fragColor;
    
    void main() {
        fragColor = texture(tex, uv);
    }
    """
    
    text_program = ctx.program(
        vertex_shader=TEXT_VERTEX_SHADER,
        fragment_shader=TEXT_FRAG_SHADER,
    )
    text_vao = ctx.vertex_array(text_program, [(quad_buffer, '2f', 'in_vert')])
    
    # Get audio file
    try:
        audio_files = list_audio_files(AUDIO_DIR)
        audio_path = audio_files[0]
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Load audio and lyrics
    audio_data = analyze_audio(audio_path)
    lyrics = load_lyrics(audio_path)
    
    # Load audio for playback
    import sounddevice as sd
    y, sr = librosa.load(audio_path.as_posix(), sr=None, mono=True)
    
    if render_mode:
        print("\n🎬 Rendering mode - generating video file...")
        frames = []
        import cv2
    else:
        # Setup playback
        stream = sd.OutputStream(channels=1, samplerate=sr, blocksize=2048)
        stream.start()
    
    playback_start = None
    
    # Play audio in a separate thread
    def play_audio():
        sd.play(y, sr, blocking=True)
    
    import threading
    
    if render_mode:
        # In render mode, generate frames based on time
        total_frames = int(audio_data['duration'] * FPS)
        playback_start = 0.0
    else:
        # In interactive mode, play audio
        audio_thread = threading.Thread(target=play_audio, daemon=True)
        audio_thread.start()
        playback_start = time.time()
    
    # Main loop
    clock = pygame.time.Clock()
    running = True
    
    print("\n🎵 Lyric Video Shader Started")
    print(f"   Track: {audio_path.name}")
    print(f"   Duration: {audio_data['duration']:.1f}s")
    print(f"   Lyrics: {len(lyrics)}")
    if not render_mode:
        print("\n   Press ESC or close window to exit")
    
    frame_count = 0
    if render_mode:
        total_frames = int(audio_data['duration'] * FPS)
    
    while running:
        if render_mode:
            # In render mode, generate frames sequentially
            if frame_count >= total_frames:
                running = False
                break
            current_time = frame_count / FPS
        else:
            # In interactive mode, handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # Calculate current playback time
            if playback_start:
                current_time = time.time() - playback_start
            else:
                current_time = 0.0
            
            # Stop if track finished
            if current_time > audio_data['duration']:
                running = False
        
        # Get audio features for current frame
        frame_idx = get_frame_index(current_time, len(audio_data['rms']))
        
        bass = float(audio_data['bass'][frame_idx])
        mid = float(audio_data['mid'][frame_idx])
        treble = float(audio_data['treble'][frame_idx])
        rms = float(audio_data['rms'][frame_idx])
        
        # Determine current lyric
        current_lyric = None
        for lyric in lyrics:
            if lyric.start_time <= current_time < lyric.end_time:
                current_lyric = lyric
                break
        
        # Calculate lyric opacity (fade in/out)
        lyric_opacity = 0.0
        if current_lyric:
            fade_time = 0.1  # 100ms fade
            time_in = current_time - current_lyric.start_time
            
            # Fade in
            if time_in < fade_time:
                lyric_opacity = time_in / fade_time
            # Fade out
            elif current_time > (current_lyric.end_time - fade_time):
                fade_out_progress = (current_time - (current_lyric.end_time - fade_time)) / fade_time
                lyric_opacity = 1.0 - fade_out_progress
            else:
                lyric_opacity = 1.0
        
        # Set uniforms
        program['resolution'].value = (WIDTH, HEIGHT)
        program['time'].value = current_time
        program['bass'].value = bass
        program['mid'].value = mid
        program['treble'].value = treble
        program['rms'].value = rms
        program['lyric_opacity'].value = lyric_opacity
        
        # Render
        ctx.clear(0.05, 0.02, 0.1)
        vao.render(moderngl.TRIANGLE_STRIP)
        
        # Render text overlay
        if current_lyric:
            text_surface = render_text_to_texture(current_lyric.text, font_size=96)
            text_width, text_height = text_surface.get_size()
            
            # Create full-screen surface with centered text
            screen_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT - 120))
            
            # Adjust opacity based on fade
            text_surface.set_alpha(int(lyric_opacity * 255))
            screen_surface.blit(text_surface, text_rect)
            
            # Convert pygame surface to OpenGL texture
            text_data = pygame.image.tostring(screen_surface, "RGBA", True)
            text_texture = ctx.texture((WIDTH, HEIGHT), 4, text_data)
            text_texture.use(0)
            text_program['tex'].value = 0
            
            # Render text on top
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            text_vao.render(moderngl.TRIANGLE_STRIP)
            text_texture.release()
        
        # Capture frame in render mode
        if render_mode:
            import cv2
            # Read from the framebuffer
            pixels = ctx.fbo.read(components=3)
            frame_array = np.frombuffer(pixels, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
            # OpenGL reads from bottom to top, so flip
            frame_array = np.flipud(frame_array)
            frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            frames.append(frame_array)
            
            # Print progress
            progress = (frame_count + 1) / total_frames * 100
            print(f"\r   Rendering... {progress:.1f}%", end="", flush=True)
            frame_count += 1
        
        pygame.display.flip()
        clock.tick(FPS)
    
    # Cleanup
    if not render_mode:
        stream.stop()
    
    # In render mode, save captured frames to video
    if render_mode and frames:
        try:
            from moviepy import VideoClip, AudioFileClip, concatenate_videoclips
            import cv2
            
            output_path = audio_path.with_stem(audio_path.stem + "_lyric_video").with_suffix(".mp4")
            
            # First save video without audio
            temp_video_path = output_path.with_stem(output_path.stem + "_temp")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(temp_video_path), fourcc, FPS, (WIDTH, HEIGHT))
            
            for frame in frames:
                out.write(frame)
            
            out.release()
            
            # Now combine with audio using moviepy
            video = VideoClip(make_frame=lambda t: frames[min(int(t * FPS), len(frames) - 1)], duration=audio_data['duration'])
            audio = AudioFileClip(str(audio_path))
            final_video = video.with_audio(audio)
            final_video.write_videofile(str(output_path), fps=FPS, verbose=False, logger=None)
            
            # Clean up temp file
            temp_video_path.unlink(missing_ok=True)
            
            print(f"\n✓ Video saved to: {output_path}")
        except Exception as e:
            print(f"\n✗ Error saving video: {e}")
    
    pygame.quit()
    print("\n✓ Closed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio-reactive lyric video shader")
    parser.add_argument("-r", "--render", action="store_true", help="Render to video file instead of displaying")
    args = parser.parse_args()
    
    main(render_mode=args.render)

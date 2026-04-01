import time
import json
import argparse
from pathlib import Path
import numpy as np
import pygame
import moderngl
import librosa
import cv2
from dataclasses import dataclass

WIDTH = 1280
HEIGHT = 720
FPS = 60

AUDIO_DIR = Path("audio")
VIDEO_DIR = Path("videos")
SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

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


def list_video_files(video_dir: Path) -> list[Path]:
    if not video_dir.exists():
        return []
    files = [p for p in video_dir.iterdir() if p.suffix.lower() in SUPPORTED_VIDEO_EXTS]
    files.sort()
    return files


def load_lyrics(audio_path: Path) -> list[Lyric]:
    """Load lyrics from JSON file alongside audio file"""
    lyrics_path = audio_path.with_suffix(".json")
    if not lyrics_path.exists():
        print(f"⚠️  No lyrics file found: {lyrics_path.name}")
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


def load_video_frames(video_paths: list[Path], target_fps: int = FPS) -> tuple[list, dict]:
    """Load video frames and return list of frames + metadata"""
    frames = []
    
    if not video_paths:
        print("⚠️  No video files found")
        return [], {"duration": 0, "frame_count": 0}
    
    video_path = video_paths[0]  # Use first video
    print(f"Loading video: {video_path.name}")
    
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    
    frame_skip = max(1, int(video_fps / target_fps))
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_skip == 0:
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            frames.append(frame)
        frame_idx += 1
    
    cap.release()
    print(f"✓ Loaded {len(frames)} video frames")
    
    return frames, {
        "duration": duration,
        "frame_count": len(frames),
        "fps": target_fps
    }


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

uniform sampler2D video_tex;
uniform vec2 resolution;
uniform float time;
uniform float bass;
uniform float mid;
uniform float treble;
uniform float rms;
uniform float lyric_opacity;
uniform float video_opacity;

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
    // Sample video texture
    vec2 video_uv = uv;
    
    // Add video distortion based on audio
    float glitch_intensity = bass * 0.15 + treble * 0.1;
    video_uv += fbm(uv * 3.0 + time * 0.3) * 0.025 * glitch_intensity;
    video_uv += sin(uv.y * 10.0 + time * 2.0) * 0.015 * bass;
    
    vec4 video_color = texture(video_tex, video_uv);
    
    // Aspect ratio correction
    vec2 aspect = vec2(resolution.x / resolution.y, 1.0);
    vec2 uv_centered = (uv - 0.5) * 2.0 * aspect;
    
    // Create shader background
    vec2 distorted_uv = uv;
    distorted_uv += fbm(uv * 3.0 + time * 0.3) * 0.03 * glitch_intensity;
    distorted_uv += sin(uv.y * 10.0 + time * 2.0) * 0.015 * bass;
    distorted_uv += cos(uv.x * 8.0 + time * 1.5) * 0.015 * mid;
    distorted_uv += noise(uv * 5.0 + time * 0.5) * 0.02 * treble;
    
    float layer1 = fbm(distorted_uv * 2.0 + time * 0.2);
    float layer2 = fbm(distorted_uv * 4.0 - time * 0.15);
    float layer3 = cellular(distorted_uv * 6.0 + time * 0.1);
    
    vec3 dark_color = vec3(0.08, 0.04, 0.15);
    vec3 mid_color = vec3(0.15, 0.08, 0.25);
    vec3 bright_color = vec3(0.25, 0.12, 0.35);
    
    vec3 bg_color = mix(dark_color, mid_color, layer1);
    bg_color = mix(bg_color, bright_color, layer2 * 0.4);
    bg_color += layer3 * 0.15 * vec3(0.1, 0.2, 0.3);
    
    // Add frequency colors
    bg_color += vec3(bass * 0.6, bass * 0.3, 0.0) * 0.8;
    bg_color += vec3(0.0, mid * 0.35, 0.0) * 0.6;
    bg_color += vec3(0.0, treble * 0.35, treble * 0.5) * 0.8;
    
    // Vignette
    float vignette = length(uv_centered) * 0.5;
    vignette = 1.0 - smoothstep(0.4, 1.8, vignette);
    bg_color *= vignette * 0.85 + 0.15;
    
    // Scanlines
    float scanlines = sin(uv.y * 300.0 + time) * 0.5 + 0.5;
    scanlines = mix(1.0, scanlines, 0.15);
    bg_color *= scanlines;
    
    // Pulse effect
    float pulse = sin(time * 6.28 * 2.0) * 0.5 + 0.5;
    float audio_pulse = rms * pulse;
    bg_color += audio_pulse * vec3(0.3, 0.15, 0.2);
    
    // Digital lines
    float line_pattern = sin(uv.y * 50.0) * 0.5 + 0.5;
    line_pattern = smoothstep(0.4, 0.5, line_pattern);
    bg_color += line_pattern * vec3(0.0, 0.3, 0.5) * (0.3 + treble * 0.3);
    
    float v_line_pattern = sin(uv.x * 40.0 + time * bass * 2.0) * 0.5 + 0.5;
    v_line_pattern = smoothstep(0.3, 0.5, v_line_pattern);
    bg_color += v_line_pattern * vec3(0.5, 0.0, 0.3) * (0.2 + mid * 0.4);
    
    // Blend video on top with dynamic opacity
    vec3 final_color = mix(bg_color, video_color.rgb, video_opacity * 0.7 + bass * 0.2);
    
    // Lyric text area highlight
    vec2 text_center = vec2(0.5, 0.5);
    float text_dist = distance(uv, text_center);
    
    float text_glow = exp(-text_dist * 1.5) * lyric_opacity * 0.5;
    final_color += vec3(0.2, 0.6, 1.0) * text_glow;
    
    // Add glow corona when lyric is visible
    if(lyric_opacity > 0.1){
        float corona = sin(text_dist * 10.0 - time * 3.0) * 0.5 + 0.5;
        corona *= exp(-text_dist * 1.2);
        final_color += vec3(0.3, 0.8, 1.0) * corona * lyric_opacity * 0.3;
    }
    
    final_color = pow(final_color, vec3(0.95));
    
    fragColor = vec4(final_color, 1.0);
}
"""


def render_text_to_texture(text: str, font_size: int = 96) -> pygame.Surface:
    """Render text to a pygame surface"""
    font = pygame.font.Font(None, font_size)
    surface = font.render(text, True, (255, 255, 255))
    return surface


def main(render_mode: bool = False):
    # Setup Pygame and ModernGL
    pygame.init()
    pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Audio-Video-Lyric Visualizer")
    
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
    
    # Create text shader
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
    
    # Load audio, lyrics, and video
    audio_data = analyze_audio(audio_path)
    lyrics = load_lyrics(audio_path)
    video_files = list_video_files(VIDEO_DIR)
    video_frames, video_meta = load_video_frames(video_files)
    
    # Load audio for playback
    import sounddevice as sd
    y, sr = librosa.load(audio_path.as_posix(), sr=None, mono=True)
    
    if render_mode:
        print("\n🎬 Rendering mode - generating combined video...")
        frames = []
    else:
        stream = sd.OutputStream(channels=1, samplerate=sr, blocksize=2048)
        stream.start()
    
    playback_start = None
    
    # Play audio in a separate thread
    def play_audio():
        sd.play(y, sr, blocking=True)
    
    import threading
    
    if render_mode:
        total_frames = int(audio_data['duration'] * FPS)
        playback_start = 0.0
    else:
        audio_thread = threading.Thread(target=play_audio, daemon=True)
        audio_thread.start()
        playback_start = time.time()
    
    # Main loop
    clock = pygame.time.Clock()
    running = True
    
    print("\n🎬 Audio-Video-Lyric Visualizer Started")
    print(f"   Track: {audio_path.name}")
    print(f"   Duration: {audio_data['duration']:.1f}s")
    print(f"   Lyrics: {len(lyrics)}")
    print(f"   Video Frames: {len(video_frames)}")
    if not render_mode:
        print("\n   Press ESC or close window to exit")
    
    frame_count = 0
    if render_mode:
        total_frames = int(audio_data['duration'] * FPS)
    
    while running:
        if render_mode:
            if frame_count >= total_frames:
                running = False
                break
            current_time = frame_count / FPS
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            if playback_start:
                current_time = time.time() - playback_start
            else:
                current_time = 0.0
            
            if current_time > audio_data['duration']:
                running = False
        
        # Get audio features
        audio_idx = get_frame_index(current_time, len(audio_data['rms']))
        
        bass = float(audio_data['bass'][audio_idx])
        mid = float(audio_data['mid'][audio_idx])
        treble = float(audio_data['treble'][audio_idx])
        rms = float(audio_data['rms'][audio_idx])
        
        # Get video frame
        video_opacity = 0.0
        if video_frames:
            video_idx = min(int(current_time / audio_data['duration'] * len(video_frames)), len(video_frames) - 1)
            video_frame = cv2.cvtColor(video_frames[video_idx], cv2.COLOR_BGR2RGB)
            video_opacity = 0.3 + bass * 0.4  # Increase with bass
            
            # Convert to OpenGL texture
            video_texture = ctx.texture((WIDTH, HEIGHT), 3, video_frame)
            video_texture.use(0)
            program['video_tex'].value = 0
        else:
            video_opacity = 0.0
        
        # Determine current lyric
        current_lyric = None
        for lyric in lyrics:
            if lyric.start_time <= current_time < lyric.end_time:
                current_lyric = lyric
                break
        
        # Calculate lyric opacity (fade in/out)
        lyric_opacity = 0.0
        if current_lyric:
            fade_time = 0.1
            time_in = current_time - current_lyric.start_time
            
            if time_in < fade_time:
                lyric_opacity = time_in / fade_time
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
        program['video_opacity'].value = video_opacity
        
        # Render background + video blend
        ctx.clear(0.05, 0.02, 0.1)
        vao.render(moderngl.TRIANGLE_STRIP)
        
        # Render text overlay
        if current_lyric:
            text_surface = render_text_to_texture(current_lyric.text, font_size=96)
            screen_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT - 120))
            
            text_surface.set_alpha(int(lyric_opacity * 255))
            screen_surface.blit(text_surface, text_rect)
            
            text_data = pygame.image.tostring(screen_surface, "RGBA", True)
            text_texture = ctx.texture((WIDTH, HEIGHT), 4, text_data)
            text_texture.use(0)
            text_program['tex'].value = 0
            
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            text_vao.render(moderngl.TRIANGLE_STRIP)
            text_texture.release()
        
        # Capture frame in render mode
        if render_mode:
            pixels = ctx.fbo.read(components=3)
            frame_array = np.frombuffer(pixels, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
            frame_array = np.flipud(frame_array)
            frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            frames.append(frame_array)
            
            progress = (frame_count + 1) / total_frames * 100
            print(f"\r   Rendering... {progress:.1f}%", end="", flush=True)
            frame_count += 1
        
        pygame.display.flip()
        clock.tick(FPS)
    
    # Cleanup
    if not render_mode:
        stream.stop()
    
    # Save video if in render mode
    if render_mode and frames:
        try:
            import subprocess
            import os
            
            output_path = audio_path.with_stem(audio_path.stem + "_combined_viz").with_suffix(".mp4")
            temp_video = output_path.with_stem(output_path.stem + "_temp_no_audio").with_suffix(".mp4")
            
            print("\n   Creating video file...")
            
            # First save video frames without audio
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(temp_video), fourcc, FPS, (WIDTH, HEIGHT))
            
            for i, frame in enumerate(frames):
                out.write(frame)
                if (i + 1) % 30 == 0:
                    print(f"   Writing frames... {(i+1)}/{len(frames)}", end='\r')
            
            out.release()
            print(f"   Video frames written: {len(frames)} frames        ")
            
            # Now embed audio using ffmpeg (much more reliable)
            print(f"   Embedding audio with ffmpeg...")
            
            cmd = [
                'ffmpeg',
                '-i', str(temp_video),        # input video
                '-i', str(audio_path),         # input audio
                '-c:v', 'copy',                # copy video codec (no re-encoding)
                '-c:a', 'aac',                 # use AAC for audio
                '-shortest',                   # stop when shortest stream ends
                '-y',                          # overwrite output file
                str(output_path)
            ]
            
            # Run ffmpeg
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Clean up temp video file
                if os.path.exists(temp_video):
                    os.remove(temp_video)
                print(f"\n✓ Video with EMBEDDED AUDIO saved to: {output_path}")
                print(f"   Audio codec: AAC")
                print(f"   Video codec: H.264 (copied, no re-encode)")
            else:
                print(f"\n✗ FFmpeg error: {result.stderr}")
                print(f"   Trying fallback method...")
                
                # Fallback: use moviepy
                from moviepy import VideoClip, AudioFileClip
                video_clip = VideoClip(
                    make_frame=lambda t: frames[min(int(t * FPS), len(frames) - 1)],
                    duration=audio_data['duration']
                )
                audio_clip = AudioFileClip(str(audio_path))
                final_video = video_clip.set_audio(audio_clip)
                final_video.write_videofile(
                    str(output_path),
                    fps=FPS,
                    verbose=False,
                    logger=None
                )
                print(f"✓ Video saved (via MoviePy): {output_path}")
                
        except FileNotFoundError:
            print("\n✗ FFmpeg not found. Install it via: choco install ffmpeg")
            print("   Or download from: https://ffmpeg.org/download.html")
        except Exception as e:
            print(f"\n✗ Error saving video: {e}")
            import traceback
            traceback.print_exc()
    
    pygame.quit()
    print("\n✓ Closed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio-Video-Lyric combined visualizer")
    parser.add_argument("-r", "--render", action="store_true", help="Render to video file instead of displaying")
    args = parser.parse_args()
    
    main(render_mode=args.render)

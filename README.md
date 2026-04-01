# Run Instuctions 

Requirements: 
Python 3+

1. Create Virtual environment 
    - `python -m venv venv`
2. activate Virtual environment 
    - bash: `source venv/bin/activate`
    - powershell: `.\\venv\\Scripts\\Activate.ps1`
3. install dependencies 
    - `pip install -r requirements.txt`
4. add desired tracks into a folder named 'audio' in project root
    - .wav, .mp3 supported
5. for video shaders, add video clips into a folder named 'videos' in project root
    - .mp4, .avi, .mov, .mkv, .webm supported
6. run desired shader 
    - `python [shader-name].py`

Available shaders:
- av-shader.py: Basic shader
- av-shader-abstract.py: Abstract geometric visuals
- av-shader-complex.py: More complex abstract visuals
- av-shader-video.py: Video clips chopped to the beat
- av-shader-lyrics.py: Lyric video with rugged glitch effects
- av-shader-combined.py: Combines video + lyrics + shader effects (NEW!)

## Lyric Video Shader

To create a lyric video:

1. Place your audio file in the `audio/` folder
2. Create a `.json` file with the same name as your audio file containing lyrics and timestamps:

```json
{
  "lyrics": [
    {
      "text": "First line of lyrics",
      "start": 0.0,
      "duration": 3.5
    },
    {
      "text": "Second line of lyrics",
      "start": 3.5,
      "duration": 3.0
    }
  ]
}
```

3. Run: `python av-shader-lyrics.py`

**Parameters explained:**
- `text`: The lyric line to display
- `start`: Time in seconds when the lyric appears
- `duration`: How long the lyric stays visible in seconds

The shader features:
- **Rugged glitch effects** with noise and distortion
- **Audio reactivity** - bass and treble frequencies control visual intensity
- **Smooth fade in/out** of lyrics
- **Vignette and scanline effects** for retro aesthetic
- **Dynamic color shifting** based on audio spectrum

## Combined Audio-Video-Lyric Visualizer

Create an ultra-engaging visualization combining all three elements:

1. Place your **audio file** in the `audio/` folder
2. (Optional) Place **video clips** in the `videos/` folder - they'll be blended in!
3. Create a `.json` file with lyrics (same format as lyric shader)
4. Run: `python av-shader-combined.py` or `python av-shader-combined.py -r` to render

**Features:**
- **Video Blending**: Video clips overlay on top of the shader system with dynamic opacity
- **Lyric Display**: Synced lyrics with smooth fade effects
- **Dual Distortion**: Both video and background respond to audio reactivity
- **Bass-Driven Opacity**: Video becomes more visible during bass-heavy moments
- **Field Tech Aesthetic**: Maintains the rugged, minimalist sci-fi vibe
- **Digital Effects**: Scanlines, glitch bars, cellular patterns, and digital lines
- **Audio Sync**: All visual elements respond to bass, mid, treble frequencies

**Video Integration:**
- If video files exist, they'll automatically blend with the background
- Video opacity increases during bass-heavy sections
- Video is distorted by the same noise effects as the background for cohesion
- Without video files, the shader works as a pure lyric visualizer

**Render Mode:**
```powershell
python av-shader-combined.py -r
```
Saves as `audio_filename_combined_viz.mp4` with embedded audio.
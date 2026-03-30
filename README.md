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
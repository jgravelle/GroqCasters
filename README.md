# GroqCasters

GroqCasters is a Python application that generates podcast scripts and corresponding audio using AI technologies. It leverages PocketGroq for script generation and Bark for text-to-speech conversion, allowing for custom voice cloning.

![image](https://github.com/user-attachments/assets/da5b4697-5324-47b6-8958-e98a1185f3da)


## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Configuration](#configuration)
4. [Custom Voice Samples](#custom-voice-samples)
5. [Operational Parameters](#operational-parameters)
6. [Dependencies](#dependencies)
7. [Additional Resources](#additional-resources)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/groqcasters.git
   cd groqcasters
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your GROQ API key as an environment variable:
   ```
   export GROQ_API_KEY=your_api_key_here
   ```
   On Windows, use `set GROQ_API_KEY=your_api_key_here`

## Usage

GroqCasters can be used in two modes:

1. Generate a new script and audio:
   ```
   python groqcasters.py path/to/input_text.txt path/to/output_directory
   ```

2. Use a pre-written script:
   ```
   python groqcasters.py path/to/script.txt path/to/output_directory --use-script
   ```

The generated audio will be saved as `full_podcast.wav` in the specified output directory.

## Configuration

Create a `config.py` file in the project directory with the following content:

```python
DEFAULT_MODEL = "your_default_model_name"
MAX_TOKENS = {
    "outline": 8000,
    "full_script": 8000,
    "dialogue": 8000
}
HOST_PROFILES = """
Host1 (Rachel): Enthusiastic, prone to personal anecdotes.
Host2 (Mike): More analytical, enjoys making pop culture references.
"""
OUTLINE_PROMPT_TEMPLATE = "Your outline prompt template here"
EXPAND_PROMPT_TEMPLATE = "Your expand prompt template here"
DIALOGUE_PROMPT_TEMPLATE = "Your dialogue prompt template here"
```

Adjust these values according to your needs.

## Custom Voice Samples

To use custom voice samples:

1. Prepare a short (15-30 second) clear audio clip of the desired voice.
2. Save the audio as a WAV file.
3. Update the `CUSTOM_MALE_VOICE_PATH` and `CUSTOM_FEMALE_VOICE_PATH` variables in `groqcasters.py` with the paths to your custom voice files.

## Operational Parameters

- `SUNO_USE_SMALL_MODELS`: Set to "True" to use smaller Bark models (default: True)
- `SUNO_OFFLOAD_CPU`: Set to "True" to offload processing to CPU if GPU is unavailable (default: False)
- `MALE_VOICE_PRESET`: Default male voice preset for Bark (default: "v2/en_speaker_6")
- `FEMALE_VOICE_PRESET`: Default female voice preset for Bark (default: "v2/en_speaker_9")

These can be adjusted in the `groqcasters.py` file.

## Dependencies

See the `requirements.txt` file for a full list of dependencies. Key dependencies include:

- pocketgroq
- bark
- torch
- numpy
- scipy

## Extending GroqCasters for Multiple Characters

GroqCasters is initially designed for two characters, but you can extend it to support more. Here's a guide on how to modify the application for multiple characters:

1. Update the `config.py` file:
   - Extend the `HOST_PROFILES` to include additional characters:
     ```python
     HOST_PROFILES = """
     Host1 (Rachel): Enthusiastic, prone to personal anecdotes.
     Host2 (Mike): More analytical, enjoys making pop culture references.
     Host3 (Alex): Tech-savvy, often explains complex concepts.
     Host4 (Sarah): Creative, brings in artistic perspectives.
     """
     ```

2. Modify the `groqcasters.py` file:
   - Add voice presets for new characters:
     ```python
     VOICE_PRESETS = {
         "rachel": "v2/en_speaker_9",
         "mike": "v2/en_speaker_6",
         "alex": "v2/en_speaker_2",
         "sarah": "v2/en_speaker_4"
     }
     ```
   - Update the `_create_voice_prompt` method to handle multiple custom voices:
     ```python
     def __init__(self):
         # ... existing code ...
         self.custom_voices = {
             "rachel": self._create_voice_prompt("path/to/rachel_voice.wav"),
             "mike": self._create_voice_prompt("path/to/mike_voice.wav"),
             "alex": self._create_voice_prompt("path/to/alex_voice.wav"),
             "sarah": self._create_voice_prompt("path/to/sarah_voice.wav")
         }
     ```
   - Modify the `generate_audio_from_script` method to use the new voice selection:
     ```python
     def generate_audio_from_script(self, script, output_dir):
         lines = script.split('\n')
         audio_segments = []
         
         for line in lines:
             if line.strip():
                 speaker, text = line.split(':', 1)
                 speaker = speaker.strip().lower()
                 text = text.strip()
                 
                 voice = self.custom_voices.get(speaker) or VOICE_PRESETS.get(speaker, VOICE_PRESETS["rachel"])
                 
                 try:
                     # ... existing audio generation code ...
                 except Exception as e:
                     print(f"Error generating audio for line: {line}")
                     print(f"Error details: {e}")
         
         # ... rest of the method remains the same ...
     ```

3. Update the script generation prompts:
   - Modify the `OUTLINE_PROMPT_TEMPLATE`, `EXPAND_PROMPT_TEMPLATE`, and `DIALOGUE_PROMPT_TEMPLATE` in `config.py` to include instructions for handling multiple characters.

4. Adjust the script parsing:
   - If your input scripts have a specific format for multiple speakers, update the script parsing logic in `generate_audio_from_script` to handle this format correctly.

5. Test thoroughly:
   - Create test scripts with multiple characters to ensure the system handles them correctly.
   - Generate audio for these test scripts and verify that each character has the correct voice.

Remember to update any other parts of the code that might assume a two-character setup, such as any hardcoded references to "Rachel" or "Mike".

By following these steps, you can extend GroqCasters to support as many characters as you need. This allows for creating more diverse and dynamic podcast scripts with a wider range of voices and personalities.

## Additional Resources

- [Bark GitHub Repository](https://github.com/suno-ai/bark)
- [PocketGroq Documentation](https://pocketgroq.readthedocs.io/)
- [Groq API Documentation](https://console.groq.com/docs/quickstart)

For more information on using Bark and PocketGroq, refer to their respective documentation:

- [Bark GitHub Docs](https://github.com/suno-ai/bark/blob/main/README.md)
- [PocketGroq Docs](https://github.com/jgravelle/pocketgroq/blob/main/README.md)



Contributing
Contributions to GroqCasters are welcome! If you encounter any problems, have feature suggestions, or want to improve the codebase, feel free to:

Open issues on the GitHub repository.
Submit pull requests with bug fixes or new features.
Improve documentation or add examples.
When contributing, please:

Follow the existing code style and conventions.
Write clear commit messages.
Add or update tests for new features or bug fixes.
Update the README if you're adding new functionality.
License
This project is licensed under the MIT License. When using any of Gravelle's code in your projects, please include a mention of "J. Gravelle" in your code and/or documentation. He's kinda full of himself, but he'd appreciate the acknowledgment.

## Support

For issues and feature requests, please [open an issue](https://github.com/yourusername/groqcasters/issues) on the GitHub repository.

![image](https://github.com/user-attachments/assets/88bf1801-134e-4479-8314-102ff418ddf4)

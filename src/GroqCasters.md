# config.py

```python
# config.py

# GroqProvider settings
DEFAULT_MODEL = "llama-3.1-70b-versatile"
MAX_TOKENS = {
    "outline": 8000,
    "full_script": 8000,
    "dialogue": 8000
}

# Host profiles
HOST_PROFILES = """
Host1 (Rachel): Enthusiastic, prone to personal anecdotes, likes to relate concepts to everyday life. Occasionally interrupts with excitement to add to a point.
Host2 (Mike): More analytical, enjoys making pop culture references, often asks clarifying questions. Sometimes finishes Rachel's sentences when he sees where she's going.
"""

# Prompt templates
OUTLINE_PROMPT_TEMPLATE = """
Create a detailed outline for a two-person podcast episode based on the following input:

{input_text}

The outline should include:
1. An attention-grabbing introduction
2. Main points to be discussed, with potential for personal anecdotes or examples
3. Interesting analogies or pop culture references
4. A conclusion that summarizes key takeaways and teases the next episode

Format the outline with clear sections and bullet points.
"""

EXPAND_PROMPT_TEMPLATE = """
Expand the following outline into a full podcast script for two hosts, Rachel and Mike:

{outline}

Host Profiles:
{host_profiles}

Guidelines:
- Make the script engaging, conversational, and easy to understand.
- Include analogies, examples, and explanations to make complex concepts accessible.
- Incorporate personal anecdotes and experiences for each host.
- Use casual language, including filler words and interjections (e.g., "um", "you know", "I mean").
- Include moments of humor, enthusiasm, and other emotions.
- Ensure the hosts build on each other's points and occasionally ask each other questions.
- Add smooth transitions between topics using personal comments or questions.
- Occasionally, have one host interrupt the other to add a point or finish their thought.

The script should feel like a natural conversation between friends, not a formal presentation.
"""

DIALOGUE_PROMPT_TEMPLATE = """
Convert the following podcast script into a natural, engaging dialogue between Rachel and Mike:

{full_script}

Host Profiles:
{host_profiles}

Guidelines:
- Our goal is to be engaging and informative, interesting and entertaining, like a real conversation between friends.
- Alternate between Rachel and Mike for each part of the dialogue.
- Make the conversation flow naturally, with hosts building on each other's points.
- Discuss both sides of any subject fairly and candidly.
- If there are disagreements, present them respectfully and explore both perspectives.
- Include casual language, interjections, and filler words (e.g., "like", "you know", "I mean").
- Add brief personal anecdotes and experiences to make it more relatable.
- Incorporate moments of humor, enthusiasm, and other emotions.
- Use rewording or clarification of points occasionally, as in natural speech.
- Ensure smooth transitions between topics using personal comments or questions.
- Maintain the scientific accuracy and main points while making the dialogue feel spontaneous and engaging.
- Rarely and sporadically (about 2-3 times in the entire script), include interruptions where one host interjects or finishes the other's thought. For example:
  Rachel: "Like they say, to err is human, and to--"
  Mike: "To forgive is divine! Exactly."
  or
  Mike: "...whole numbers, round numbers--"
  Rachel: "Even imaginary numbers! Right?"
  Mike: "Yeah! I hadn't considered that."

Format the output as:
Rachel: [Rachel's dialogue]
Mike: [Mike's dialogue]
Rachel: [Rachel's dialogue]
...and so on.

Remember to make the conversation sound as natural and engaging as possible, as if two friends are casually discussing the topic, with occasional friendly interruptions.
"""
```

# groqcasters.py

```python
import os
import sys
import torch
import numpy as np
from pocketgroq import GroqProvider, GroqAPIKeyMissingError, GroqAPIError
from config import (
    DEFAULT_MODEL,
    MAX_TOKENS,
    HOST_PROFILES,
    OUTLINE_PROMPT_TEMPLATE,
    EXPAND_PROMPT_TEMPLATE,
    DIALOGUE_PROMPT_TEMPLATE
)
from bark import SAMPLE_RATE, generate_audio, preload_models
from bark.generation import generate_text_semantic
from bark.api import semantic_to_waveform
from scipy.io.wavfile import write as write_wav, read as read_wav

# Set environment variables for Bark
os.environ["SUNO_USE_SMALL_MODELS"] = "True"
os.environ["SUNO_OFFLOAD_CPU"] = "False"  # We'll try to use GPU first

MALE_VOICE_PRESET = "v2/en_speaker_6"
FEMALE_VOICE_PRESET = "v2/en_speaker_9"

# Paths for custom voice samples (update these with your actual file paths)
CUSTOM_MALE_VOICE_PATH = "null"
CUSTOM_FEMALE_VOICE_PATH = "bark_voice_samples/female.wav"

class GroqCasters:
    def __init__(self):
        try:
            self.groq = GroqProvider()
            self._setup_gpu()
            preload_models()
            self.custom_male_voice = self._create_voice_prompt(CUSTOM_MALE_VOICE_PATH)
            self.custom_female_voice = self._create_voice_prompt(CUSTOM_FEMALE_VOICE_PATH)
        except GroqAPIKeyMissingError:
            print("Error: GROQ_API_KEY not found. Please set it in your environment variables.")
            sys.exit(1)
        except Exception as e:
            print(f"Error during initialization: {e}")
            sys.exit(1)

    def _setup_gpu(self):
        if torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            torch.cuda.set_device(0)
        else:
            print("No GPU available. Using CPU.")
            os.environ["SUNO_OFFLOAD_CPU"] = "True"

    def _create_voice_prompt(self, audio_file_path):
        if audio_file_path == "null" or not os.path.exists(audio_file_path):
            print(f"Warning: Custom voice file not found at {audio_file_path}. Using default voice.")
            return None

        try:
            sample_rate, audio_data = read_wav(audio_file_path)
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32767).astype(np.int16)
            audio_data = audio_data.astype(np.float32) / 32767.0
            audio_data = audio_data[:, 0] if len(audio_data.shape) > 1 else audio_data

            if sample_rate != SAMPLE_RATE:
                print(f"Warning: Audio file sample rate ({sample_rate} Hz) does not match required rate ({SAMPLE_RATE} Hz). This may affect voice quality.")

            # Generate text semantic tokens
            semantic_tokens = generate_text_semantic(
                "Hello, this is a voice prompt.",
                history_prompt=audio_data,
                temp=0.7,
                min_eos_p=0.05,
            )

            return semantic_tokens
        except Exception as e:
            print(f"Error processing custom voice file: {e}")
            return None

    def generate_podcast_script(self, input_text):
        outline = self._generate_outline(input_text)
        if not outline:
            return None

        full_script = self._expand_outline(outline)
        if not full_script:
            return None

        dialogue_script = self._convert_to_dialogue(full_script)
        return dialogue_script

    def _generate_outline(self, input_text):
        prompt = OUTLINE_PROMPT_TEMPLATE.format(input_text=input_text)
        try:
            return self.groq.generate(prompt, model=DEFAULT_MODEL, max_tokens=MAX_TOKENS["outline"])
        except GroqAPIError as e:
            print(f"Error generating outline: {e}")
            return None

    def _expand_outline(self, outline):
        prompt = EXPAND_PROMPT_TEMPLATE.format(outline=outline, host_profiles=HOST_PROFILES)
        try:
            return self.groq.generate(prompt, model=DEFAULT_MODEL, max_tokens=MAX_TOKENS["full_script"])
        except GroqAPIError as e:
            print(f"Error expanding outline: {e}")
            return None

    def _convert_to_dialogue(self, full_script):
        prompt = DIALOGUE_PROMPT_TEMPLATE.format(full_script=full_script, host_profiles=HOST_PROFILES)
        try:
            return self.groq.generate(prompt, model=DEFAULT_MODEL, max_tokens=MAX_TOKENS["dialogue"])
        except GroqAPIError as e:
            print(f"Error converting to dialogue: {e}")
            return None

    def generate_audio_from_script(self, script, output_dir):
        lines = script.split('\n')
        audio_segments = []
        
        for line in lines:
            if line.strip():
                speaker, text = line.split(':', 1)
                speaker = speaker.strip().lower()
                text = text.strip()
                
                # Choose a voice preset or custom voice based on the speaker
                if speaker == "mike":
                    voice = self.custom_male_voice if self.custom_male_voice is not None else MALE_VOICE_PRESET
                else:
                    voice = self.custom_female_voice if self.custom_female_voice is not None else FEMALE_VOICE_PRESET
                
                try:
                    if isinstance(voice, str):
                        audio_array = generate_audio(text, history_prompt=voice)
                    else:
                        semantic_tokens = generate_text_semantic(
                            text,
                            history_prompt=voice,
                            temp=0.7,
                            min_eos_p=0.05,
                        )
                        audio_array = semantic_to_waveform(semantic_tokens)
                    
                    audio_segments.append(audio_array)
                except Exception as e:
                    print(f"Error generating audio for line: {line}")
                    print(f"Error details: {e}")
        
        # Combine all audio segments
        full_audio = np.concatenate(audio_segments)
        
        # Save the full audio file
        output_file = os.path.join(output_dir, "full_podcast.wav")
        write_wav(output_file, SAMPLE_RATE, full_audio)
        
        print(f"Full podcast audio saved to: {output_file}")

def process_input_text(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except IOError:
        print(f"Error: Unable to read file at {file_path}")
        return None

def main():
    if len(sys.argv) not in [3, 4]:
        print("Usage:")
        print("  To generate script and audio: python groqcasters.py <input_file_path> <output_directory>")
        print("  To use pre-written script:    python groqcasters.py <script_file_path> <output_directory> --use-script")
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_directory = sys.argv[2]
    use_existing_script = len(sys.argv) == 4 and sys.argv[3] == "--use-script"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    casters = GroqCasters()

    if use_existing_script:
        print("Using pre-written script...")
        script = process_input_text(input_file_path)
        if not script:
            print("Failed to read the script file.")
            return
    else:
        print("Generating new podcast script...")
        input_text = process_input_text(input_file_path)
        if not input_text:
            print("Failed to process input text.")
            return
        script = casters.generate_podcast_script(input_text)
        if not script:
            print("Failed to generate podcast script.")
            return

    print("Generated/Loaded podcast script:")
    print(script)
    print("\nGenerating audio...")
    casters.generate_audio_from_script(script, output_directory)

if __name__ == "__main__":
    main()
```

# test.py

```python
import os
import torch
import numpy as np
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

# Set environment variables for Bark
os.environ["SUNO_USE_SMALL_MODELS"] = "True"
os.environ["SUNO_OFFLOAD_CPU"] = "False"  # We'll try to use GPU first

OUTPUT_DIR = "bark_voice_samples"
TEXT_PROMPT = "I don't know. [laughs] I mean, does it matter?"

def setup_gpu():
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.cuda.set_device(0)
        return True
    else:
        print("No GPU available. Using CPU.")
        os.environ["SUNO_OFFLOAD_CPU"] = "True"
        return False

def generate_audio_with_voice(text, voice_preset):
    try:
        return generate_audio(text, history_prompt=voice_preset)
    except Exception as e:
        print(f"Error generating audio for voice {voice_preset}: {e}")
        return None

def main():
    print("Starting Bark voice comparison test...")
    
    # Setup GPU
    gpu_available = setup_gpu()

    # Print PyTorch and CUDA versions
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Preloading models...")
    try:
        preload_models()
    except Exception as e:
        print(f"Error preloading models: {e}")
        return

    print(f"Generating audio samples for voices 0-9 saying: '{TEXT_PROMPT}'")
    
    for i in range(10):
        voice_preset = f"v2/en_speaker_{i}"
        print(f"Generating audio with voice preset: {voice_preset}")
        
        audio_array = generate_audio_with_voice(TEXT_PROMPT, voice_preset)
        
        if audio_array is not None:
            output_file = os.path.join(OUTPUT_DIR, f"bark_output_voice_{i}.wav")
            write_wav(output_file, SAMPLE_RATE, audio_array)
            print(f"Audio saved to {output_file}")
        else:
            print(f"Failed to generate audio for voice {i}")

    print(f"Test complete. Audio samples saved in the '{OUTPUT_DIR}' directory.")

    if gpu_available:
        print(f"GPU memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

if __name__ == "__main__":
    main()
```


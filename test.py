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
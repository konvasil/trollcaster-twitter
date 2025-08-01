import os
import time
import numpy as np
import sounddevice as sd
import sox
from pythonosc import udp_client, osc_server
from pythonosc.dispatcher import Dispatcher
from elevenlabs.client import ElevenLabs
from threading import Thread
import logging
import librosa
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize ElevenLabs client
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# Project directory for temporary files
PROJECT_DIR = os.path.expanduser("~/echo-chamber")
PROCESSED_AUDIO_DIR = os.path.join(PROJECT_DIR, "processed_audio")
if not os.path.exists(PROCESSED_AUDIO_DIR):
    os.makedirs(PROCESSED_AUDIO_DIR)
SAVE_AUDIO_FILES = os.getenv("SAVE_AUDIO_FILES", "false").lower() == "true"

# Set sounddevice output device (optional, uncomment and set index if needed)
# sd.default.device = <device_index>  # Replace with index from sd.query_devices()

# Initialize OSC clients
supercollider_client = udp_client.SimpleUDPClient("127.0.0.1", 57120)
processing_client = udp_client.SimpleUDPClient("127.0.0.1", 9000)

def apply_glitch_effects(audio_file, toxicity, output_file):
    """Apply glitch effects based on toxicity using SoX."""
    tfm = sox.Transformer()
    # Pitch shift
    pitch_factor = 1.0 + (toxicity * 0.25)  # +5% to +25%
    pitch_semitones = pitch_factor * 1  # Reduced to +0.2 to +1 semitone
    logging.debug(f"Applying SoX pitch shift: {pitch_semitones} semitones")
    tfm.pitch(pitch_semitones)
    # Speed (rate)
    speed_factor = 0.8 + (toxicity * 0.4)  # 0.8 to 1.2
    tfm.tempo(speed_factor)
    # Downsampling (target sample rate)
    target_rate = int(44100 * (1 - toxicity * 0.5))  # 22050–44100 Hz
    target_rate = max(8000, min(44100, target_rate))  # Clamp to 8000–44100 Hz
    logging.debug(f"Applying SoX downsample with target rate: {target_rate} Hz")
    tfm.rate(target_rate)
    # Apply effects
    tfm.build(audio_file, output_file)
    logging.debug(f"SoX processed: {audio_file} -> {output_file}")
    return output_file

def apply_stutter(samples, toxicity, sample_rate):
    """Apply stuttering effect by repeating short segments."""
    if toxicity <= 0.5:
        return samples
    chunk_size = int(0.05 * sample_rate)  # 50ms chunks
    stuttered = []
    for i in range(0, len(samples), chunk_size):
        chunk = samples[i:i + chunk_size]
        if np.random.random() < toxicity * 0.3:
            repeats = int(2 + toxicity * 3)  # 2–5 repeats
            stuttered.extend(np.tile(chunk, repeats))
        else:
            stuttered.extend(chunk)
    result = np.array(stuttered, dtype=np.float32)
    logging.debug(f"Stutter applied: {len(result)} samples")
    return result

def play_tts(text, toxicity, duration):
    """Generate and play TTS with glitch effects."""
    audio_file = os.path.join(PROJECT_DIR, f"tts_temp_{time.time()}.mp3")
    processed_file = os.path.join(PROCESSED_AUDIO_DIR, f"tts_processed_{time.time()}.wav")
    try:
        # Generate TTS with ElevenLabs
        logging.debug(f"Generating TTS for text: {text}")
        audio = client.generate(
            text=text,
            voice="Rachel",
            model="eleven_monolingual_v1"
        )
        # Save audio to file
        logging.debug(f"Saving TTS to: {audio_file}")
        with open(audio_file, "wb") as f:
            for chunk in audio:
                if chunk:
                    f.write(chunk)
        # Apply glitch effects with SoX
        logging.debug(f"Applying glitch effects: {audio_file} -> {processed_file}")
        processed_file = apply_glitch_effects(audio_file, toxicity, processed_file)
        # Load audio with librosa
        logging.debug(f"Loading audio: {processed_file}")
        samples, sample_rate = librosa.load(processed_file, sr=None, mono=True)
        samples = samples.astype(np.float32)
        logging.debug(f"Loaded {len(samples)} samples at {sample_rate} Hz")
        # Apply stuttering
        samples = apply_stutter(samples, toxicity, sample_rate)
        # Normalize for playback
        if np.max(np.abs(samples)) > 0:
            samples /= np.max(np.abs(samples)) * 1.1  # Avoid clipping
        else:
            logging.error("No valid audio samples for playback")
            return
        # Play audio
        logging.debug("Playing audio with sounddevice")
        sd.play(samples, sample_rate)
        # Send OSC for sync
        supercollider_client.send_message("/ttsTrigger", [float(toxicity), float(duration)])
        processing_client.send_message("/ttsTrigger", [float(toxicity), float(duration)])
        logging.info(f"TTS playing: toxicity={toxicity}, duration={duration}, text={text}")
        # Wait for duration
        sd.wait()
    except Exception as e:
        if "quota_exceeded" in str(e):
            logging.error(f"ElevenLabs quota exceeded: {e}. Upgrade plan or use new API key at https://elevenlabs.io/")
        else:
            logging.error(f"Error in play_tts: {e}")
    finally:
        # Clean up files unless SAVE_AUDIO_FILES=true
        if not SAVE_AUDIO_FILES:
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
                if os.path.exists(processed_file):
                    os.remove(processed_file)
                logging.debug(f"Cleaned up: {audio_file}, {processed_file}")
            except Exception as e:
                logging.error(f"Error cleaning up files: {e}")

def handle_tts_data(unused_addr, text, toxicity, duration):
    """OSC handler for /ttsData."""
    logging.info(f"Received OSC: /ttsData, text={text}, toxicity={toxicity}, duration={duration}")
    # Run TTS in a separate thread to avoid blocking OSC server
    Thread(target=play_tts, args=(text, toxicity, duration)).start()

def main():
    # Setup OSC server
    dispatcher = Dispatcher()
    dispatcher.map("/ttsData", handle_tts_data)
    server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 57121), dispatcher)
    logging.info("TTS OSC server running on 127.0.0.1:57121")
    server.serve_forever()

if __name__ == "__main__":
    main()

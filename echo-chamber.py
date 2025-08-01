import json
import time
import logging
import os
from pythonosc import udp_client
from transformers import pipeline
import numpy as np
from dotenv import load_dotenv

# Load environment variables (if needed for other APIs)
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize toxicity classifier
classifier = pipeline("text-classification", model="unitary/toxic-bert", return_all_scores=True)

# Initialize OSC clients
supercollider_client = udp_client.SimpleUDPClient("127.0.0.1", 57120)
blender_client = udp_client.SimpleUDPClient("127.0.0.1", 57121)
tts_client = udp_client.SimpleUDPClient("127.0.0.1", 9000)

def get_toxicity_score(text):
    """Get toxicity score from text."""
    try:
        results = classifier(text)[0]
        for result in results:
            if result['label'] == 'toxic':
                return result['score']
        return 0.0
    except Exception as e:
        logging.error(f"Error in toxicity scoring: {e}")
        return 0.0

def process_tweet(tweet):
    """Process a single tweet and send OSC messages."""
    text = tweet.get('text', '')
    toxicity_scores = tweet.get('toxicity_scores', {})
    toxicity = toxicity_scores.get('toxic', get_toxicity_score(text))

    if toxicity > 0.1:  # Threshold for processing
        logging.warning(f"Toxic tweet: {text}")
        duration = 5.0 + toxicity * 2.0  # 5–7s based on toxicity

        # Determine OSC address based on toxicity
        if toxicity > 0.8:
            osc_address = "/highToxic"
        elif toxicity > 0.4:
            osc_address = "/moderateToxic"
        else:
            osc_address = "/mildToxic"

        # Send OSC to SuperCollider
        supercollider_client.send_message(osc_address, [float(toxicity), float(duration)])
        logging.info(f"OSC sent to SuperCollider: {osc_address} with score {toxicity}, duration {duration}")

        # Send OSC to Processing
        blender_client.send_message("/toxicity", [float(toxicity), text, float(duration), time.strftime("%Y-%m-%d %H:%M:%S"), osc_address[1:]])
        logging.info(f"OSC sent to Processing: /toxicity with score {toxicity}, duration {duration}")

        # Send OSC to TTS processor
        tts_client.send_message("/ttsData", [text, float(toxicity), float(duration)])
        logging.info(f"OSC sent to TTS: /ttsData with score {toxicity}, duration {duration}")

        # Delay after processing tweet to slow down OSC triggers
        time.sleep(8)  # 8-second delay

def main():
    """Main loop to monitor toxic_tweets_log.jsonl."""
    log_file = "toxic_tweets_log.jsonl"
    last_position = 0

    while True:
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    f.seek(last_position)
                    for line in f:
                        if line.strip():
                            tweet = json.loads(line.strip())
                            process_tweet(tweet)
                    last_position = f.tell()
            time.sleep(5)  # Check every 5 seconds
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            time.sleep(25)

if __name__ == "__main__":
    logging.info("Starting echo-chamber")
    main()

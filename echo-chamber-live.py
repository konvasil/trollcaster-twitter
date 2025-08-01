#!/usr/bin/env python3
"""
Troll Caster - Echo Chamber Live
Real-time toxic tweet detection and processing system

This module handles:
- Live Twitter API monitoring
- Dual toxicity detection (keywords + BERT)
- OSC communication to audio/visual systems
- Data logging and caching
- Rate limiting and ethical collection

Authors: Dr. Konstantinos Vasilakos & Serra Inderim Vasilakos
Project: Troll Caster - XARTS 2025
"""

import tweepy
import json
import time
import re
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from transformers import pipeline
from pythonosc import udp_client
import threading
import queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('troll_caster.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ToxicTweetProcessor:
    """
    Main class for processing toxic tweets in real-time
    Handles Twitter API, toxicity detection, and OSC communication
    """

    def __init__(self, config_file: str = "config.json"):
        """Initialize the toxic tweet processor"""
        self.config = self.load_config(config_file)
        self.setup_twitter_api()
        self.setup_toxicity_detection()
        self.setup_osc_clients()
        self.setup_data_logging()

        # Rate limiting and ethical constraints
        self.tweets_processed_today = 0
        self.last_search_time = None
        self.monthly_tweet_count = 0
        self.max_monthly_tweets = 100

        # Processing queue for real-time handling
        self.tweet_queue = queue.Queue()
        self.processing_active = True

    def load_config(self, config_file: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_file}")
            return config
        except FileNotFoundError:
            logger.error(f"Config file {config_file} not found. Creating default config.")
            return self.create_default_config(config_file)

    def create_default_config(self, config_file: str) -> Dict:
        """Create default configuration file"""
        default_config = {
            "twitter": {
                "bearer_token": "YOUR_BEARER_TOKEN_HERE",
                "api_key": "YOUR_API_KEY_HERE",
                "api_secret": "YOUR_API_SECRET_HERE",
                "access_token": "YOUR_ACCESS_TOKEN_HERE",
                "access_token_secret": "YOUR_ACCESS_TOKEN_SECRET_HERE"
            },
            "osc": {
                "supercollider": {"host": "127.0.0.1", "port": 57120},
                "blender": {"host": "127.0.0.1", "port": 57121},
                "tts": {"host": "127.0.0.1", "port": 9000}
            },
            "processing": {
                "search_query": "hate lang:en",
                "search_interval_hours": 1,
                "tweets_per_search": 10,
                "delay_between_tweets": 5,
                "toxicity_threshold": 0.4
            },
            "toxic_keywords": [
                "hate you", "loser", "stupid idiot", "go die", "shut up",
                "troll", "kill yourself", "worthless", "pathetic", "moron"
            ]
        }

        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=4)
        logger.info(f"Default config created: {config_file}")
        return default_config

    def setup_twitter_api(self):
        """Initialize Twitter API connection"""
        try:
            twitter_config = self.config["twitter"]

            # Twitter API v2 client
            self.client = tweepy.Client(
                bearer_token=twitter_config["bearer_token"],
                consumer_key=twitter_config["api_key"],
                consumer_secret=twitter_config["api_secret"],
                access_token=twitter_config["access_token"],
                access_token_secret=twitter_config["access_token_secret"],
                wait_on_rate_limit=True
            )

            # Test connection
            try:
                self.client.get_me()
                logger.info("Twitter API connection successful")
            except Exception as e:
                logger.error(f"Twitter API connection failed: {e}")

        except Exception as e:
            logger.error(f"Failed to setup Twitter API: {e}")

    def setup_toxicity_detection(self):
        """Initialize BERT toxicity detection model"""
        try:
            logger.info("Loading BERT toxicity detection model...")
            self.classifier = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                top_k=None,
                device=-1  # Use CPU, change to 0 for GPU
            )
            logger.info("BERT model loaded successfully")

            # Compile toxic keywords regex pattern
            toxic_keywords = self.config["toxic_keywords"]
            self.toxic_pattern = '|'.join(map(re.escape, toxic_keywords))
            logger.info(f"Loaded {len(toxic_keywords)} toxic keywords")

        except Exception as e:
            logger.error(f"Failed to setup toxicity detection: {e}")
            self.classifier = None

    def setup_osc_clients(self):
        """Initialize OSC clients for audio/visual communication"""
        try:
            osc_config = self.config["osc"]

            # SuperCollider audio client
            self.sc_client = udp_client.SimpleUDPClient(
                osc_config["supercollider"]["host"],
                osc_config["supercollider"]["port"]
            )

            # Blender visualization client
            self.blender_client = udp_client.SimpleUDPClient(
                osc_config["blender"]["host"],
                osc_config["blender"]["port"]
            )

            # Text-to-speech client
            self.tts_client = udp_client.SimpleUDPClient(
                osc_config["tts"]["host"],
                osc_config["tts"]["port"]
            )

            logger.info("OSC clients initialized")

        except Exception as e:
            logger.error(f"Failed to setup OSC clients: {e}")

    def setup_data_logging(self):
        """Initialize data logging system"""
        self.log_file = "toxic_tweets_log.jsonl"

        # Create log file if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                pass
            logger.info(f"Created log file: {self.log_file}")

    def analyze_toxicity(self, text: str) -> Dict[str, float]:
        """
        Analyze tweet text for toxicity using toxic-bert
        Returns dictionary of toxicity scores
        """
        if not self.classifier:
            return {}

        try:
            results = self.classifier(text)[0]
            toxic_labels = {}

            # Extract scores for specific toxicity categories
            for result in results:
                label = result['label'].lower()
                if label in ["toxic", "threat", "insult", "obscene", "identity_hate"]:
                    toxic_labels[label] = result['score']

            return toxic_labels

        except Exception as e:
            logger.error(f"Error analyzing toxicity: {e}")
            return {}

    def check_for_toxic(self, tweet_text: str) -> Tuple[bool, Dict[str, float]]:
        """
        Check if tweet is toxic using dual detection system
        Returns (is_toxic, toxicity_scores)
        """
        # Keyword-based detection
        keyword_toxic = bool(re.search(self.toxic_pattern, tweet_text.lower()))

        # Model-based detection
        toxic_scores = self.analyze_toxicity(tweet_text)
        threshold = self.config["processing"]["toxicity_threshold"]
        model_toxic = any(score > threshold for score in toxic_scores.values())

        # Combined decision using OR logic
        is_toxic = keyword_toxic or model_toxic

        logger.debug(f"Toxicity check - Keywords: {keyword_toxic}, Model: {model_toxic}, Result: {is_toxic}")

        return is_toxic, toxic_scores

    def categorize_toxicity_level(self, toxic_scores: Dict[str, float]) -> str:
        """Categorize toxicity level based on scores"""
        if not toxic_scores:
            return "none"

        max_score = max(toxic_scores.values())

        if max_score < 0.5:
            return "mild"
        elif max_score < 0.8:
            return "moderate"
        else:
            return "high"

    def send_osc_messages(self, tweet_data: Dict):
        """Send OSC messages to audio, visual, and TTS systems"""
        try:
            toxicity_level = tweet_data["toxicity_level"]
            max_score = max(tweet_data["toxicity_scores"].values()) if tweet_data["toxicity_scores"] else 0
            text_excerpt = tweet_data["text"][:100]  # Limit text length

            # SuperCollider audio synthesis
            if toxicity_level == "mild":
                self.sc_client.send_message("/mildToxic", [max_score, 8.0])
            elif toxicity_level == "moderate":
                self.sc_client.send_message("/moderateToxic", [max_score, 8.0])
            elif toxicity_level == "high":
                self.sc_client.send_message("/highToxic", [max_score, 8.0])

            # Blender visualization
            self.blender_client.send_message("/toxicity", [
                max_score,
                text_excerpt,
                8.0,  # duration
                tweet_data["timestamp"],
                toxicity_level
            ])

            # Text-to-speech processing
            self.tts_client.send_message("/ttsData", [
                text_excerpt,
                max_score,
                8.0  # duration
            ])

            logger.info(f"OSC messages sent for {toxicity_level} toxicity (score: {max_score:.3f})")

        except Exception as e:
            logger.error(f"Error sending OSC messages: {e}")

    def log_toxic_tweet(self, tweet_data: Dict):
        """Log toxic tweet data to JSONL file"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                json.dump(tweet_data, f, ensure_ascii=False)
                f.write('\n')
            logger.debug(f"Logged tweet {tweet_data['tweet_id']}")

        except Exception as e:
            logger.error(f"Error logging tweet: {e}")

    def process_tweet(self, tweet) -> Optional[Dict]:
        """Process individual tweet for toxicity"""
        try:
            tweet_text = tweet.text
            tweet_id = tweet.id
            author_id = tweet.author_id
            created_at = tweet.created_at.isoformat()

            # Check for toxicity
            is_toxic, toxic_scores = self.check_for_toxic(tweet_text)

            if is_toxic:
                toxicity_level = self.categorize_toxicity_level(toxic_scores)

                tweet_data = {
                    "tweet_id": str(tweet_id),
                    "author_id": str(author_id),
                    "text": tweet_text,
                    "timestamp": created_at,
                    "toxicity_scores": toxic_scores,
                    "toxicity_level": toxicity_level,
                    "processed_at": datetime.now().isoformat()
                }

                # Log the toxic tweet
                self.log_toxic_tweet(tweet_data)

                # Send to processing queue
                self.tweet_queue.put(tweet_data)

                logger.info(f"Toxic tweet detected: {toxicity_level} level (ID: {tweet_id})")
                return tweet_data

            return None

        except Exception as e:
            logger.error(f"Error processing tweet: {e}")
            return None

    def search_tweets(self) -> List[Dict]:
        """Search for tweets using Twitter API"""
        try:
            # Check rate limiting
            if self.monthly_tweet_count >= self.max_monthly_tweets:
                logger.warning("Monthly tweet limit reached")
                return []

            # Check time-based rate limiting
            if self.last_search_time:
                time_diff = datetime.now() - self.last_search_time
                min_interval = timedelta(hours=self.config["processing"]["search_interval_hours"])
                if time_diff < min_interval:
                    logger.info("Search interval not reached, skipping")
                    return []

            # Perform search
            query = self.config["processing"]["search_query"]
            max_results = self.config["processing"]["tweets_per_search"]

            logger.info(f"Searching for tweets: '{query}' (max: {max_results})")

            tweets = tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                max_results=max_results,
                tweet_fields=['created_at', 'author_id', 'public_metrics']
            ).flatten(limit=max_results)

            processed_tweets = []
            delay = self.config["processing"]["delay_between_tweets"]

            for tweet in tweets:
                tweet_data = self.process_tweet(tweet)
                if tweet_data:
                    processed_tweets.append(tweet_data)

                # Respectful delay between processing
                time.sleep(delay)

                # Update counters
                self.monthly_tweet_count += 1
                if self.monthly_tweet_count >= self.max_monthly_tweets:
                    logger.warning("Monthly limit reached during processing")
                    break

            self.last_search_time = datetime.now()
            logger.info(f"Processed {len(processed_tweets)} toxic tweets from {max_results} searched")

            return processed_tweets

        except Exception as e:
            logger.error(f"Error searching tweets: {e}")
            return []

    def process_queue_worker(self):
        """Worker thread for processing tweet queue"""
        while self.processing_active:
            try:
                # Get tweet from queue with timeout
                tweet_data = self.tweet_queue.get(timeout=1.0)

                # Send OSC messages for artistic processing
                self.send_osc_messages(tweet_data)

                # Mark task as done
                self.tweet_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in queue worker: {e}")

    def run_continuous(self):
        """Run the system continuously"""
        logger.info("Starting Troll Caster continuous monitoring...")

        # Start queue processing thread
        queue_thread = threading.Thread(target=self.process_queue_worker, daemon=True)
        queue_thread.start()

        try:
            while True:
                # Search for and process tweets
                self.search_tweets()

                # Wait before next search cycle
                search_interval = self.config["processing"]["search_interval_hours"] * 3600
                logger.info(f"Waiting {search_interval/3600:.1f} hours until next search...")
                time.sleep(search_interval)

        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
            self.processing_active = False
            queue_thread.join(timeout=5.0)

    def run_single_search(self):
        """Run a single search cycle for testing"""
        logger.info("Running single search cycle...")

        # Start queue processing thread
        queue_thread = threading.Thread(target=self.process_queue_worker, daemon=True)
        queue_thread.start()

        # Perform single search
        processed_tweets = self.search_tweets()

        # Wait for queue to process
        time.sleep(10)

        # Stop processing
        self.processing_active = False
        queue_thread.join(timeout=5.0)

        logger.info(f"Single search complete. Processed {len(processed_tweets)} toxic tweets.")
        return processed_tweets

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Troll Caster - Live Toxic Tweet Processor")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--single", action="store_true", help="Run single search instead of continuous")
    parser.add_argument("--test", action="store_true", help="Test mode - no OSC output")

    args = parser.parse_args()

    # Create processor instance
    processor = ToxicTweetProcessor(args.config)

    if args.single:
        # Run single search for testing
        processor.run_single_search()
    else:
        # Run continuous monitoring
        processor.run_continuous()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import os
from dotenv import load_dotenv
import requests
import json

# Load environment variables
load_dotenv()

# Get Bearer Token
BEARER_TOKEN = os.getenv("BEARER_TOKEN")
if not BEARER_TOKEN:
    print("Error: BEARER_TOKEN not found in .env file")
    exit(1)

# Set up headers
headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}

# Make API request
try:
    response = requests.get(
        "https://api.twitter.com/2/tweets/search/stream/rules",
        headers=headers
    )
    response.raise_for_status()  # Raise an error for bad status codes

    # Print status code and parsed response
    print(f"Status Code: {response.status_code}")
    try:
        data = response.json()
        print("Response JSON:")
        print(json.dumps(data, indent=2))  # Pretty-print JSON
    except ValueError:
        print("Error: Response is not valid JSON")
        print(f"Raw Response: {response.text}")

except requests.exceptions.HTTPError as http_err:
    print(f"HTTP Error: {http_err}")
    print(f"Response Text: {response.text}")
except requests.exceptions.RequestException as err:
    print(f"Request Error: {err}")

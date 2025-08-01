#!/usr/bin/env python3

from pythonosc import udp_client

client = udp_client.SimpleUDPClient("127.0.0.1", 57120)
client.send_message("/mildToxic", [0.5, "Test message"])

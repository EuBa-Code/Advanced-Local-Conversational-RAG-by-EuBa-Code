
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("No OPENROUTER_API_KEY set")
    exit(1)

resp = requests.get("https://openrouter.ai/api/v1/models")
if resp.status_code != 200:
    print(f"Error: {resp.status_code} {resp.text}")
    exit(1)

models = resp.json()["data"]
free_models = [m["id"] for m in models if "free" in m["id"] or m["pricing"]["prompt"] == "0"]

print("Available FREE models:")
for m in sorted(free_models):
    print(f"- {m}")

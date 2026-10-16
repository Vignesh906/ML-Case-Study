import json
import os

with open("data/routes_metadata.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

print("Airlines in metadata:")
airlines = meta.get("airplane", {}).get("airlines", [])
for a in airlines[:10]:
    print(a)

print("\nAirports count:", len(meta.get("airplane", {}).get("airports", [])))
print("Sample airports:", meta.get("airplane", {}).get("airports", [])[:5])

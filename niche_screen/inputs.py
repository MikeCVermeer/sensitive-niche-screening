"""Minimal helpers for caller-owned topic cues and exact destination lists."""
import json
from pathlib import Path
from urllib.parse import urlsplit


def load_lists(path):
    """Read {"cues": {topic:[terms]}, "destinations": {host:[topics]}}."""
    data = json.loads(Path(path).read_text())
    cues, destinations = data['cues'], data['destinations']
    if not isinstance(cues, dict) or not isinstance(destinations, dict):
        raise ValueError('cues and destinations must be mappings')
    if any(not isinstance(v,list) or not v or any(not isinstance(s,str) or not s.strip() for s in v) for v in cues.values()):
        raise ValueError('each cue list must contain nonempty strings')
    if any(not isinstance(v,list) or any(topic not in cues for topic in v) for v in destinations.values()):
        raise ValueError('destination labels must refer to known topics')
    return cues, {host.casefold().rstrip('.'): labels for host,labels in destinations.items()}


def categorize_links(urls, destinations):
    """Map full URLs to labels by exact host, without fetching any content."""
    result=[]
    for url in urls:
        host=(urlsplit(url).hostname or '').casefold().rstrip('.')
        result.extend(destinations.get(host, []))
    return result

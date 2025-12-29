# -*- coding: utf-8 -*-
import os
import requests

# Use Serper (Google Search API compatible)
serper_key = os.environ.get('SERPER_API_KEY', '1005592c05898cd38eb4b74d0fe9a03e0965d8c3')
endpoint = os.environ.get('SERPER_ENDPOINT', 'https://google.serper.dev/search')

if not serper_key:
    raise RuntimeError('Missing SERPER_API_KEY environment variable')

query = os.environ.get('SERPER_QUERY', 'Insert your query here')

headers = {
    'X-API-KEY': serper_key,
    'Content-Type': 'application/json'
}

payload = {"q": query}

try:
    resp = requests.post(endpoint, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    for i, item in enumerate((data.get('organic', []) or [])[:3]):
        print(f"{i}: {item.get('title', '')}\n{item.get('link', '')}\n{item.get('snippet', '')}\n")
except Exception as ex:
    raise ex
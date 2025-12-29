
import json
import os
import requests

os.environ['BING_SEARCH_V7_SUBSCRIPTION_KEY'] = "c350758795414502b5f7e994ec9418f0"
subscription_key = os.environ['BING_SEARCH_V7_SUBSCRIPTION_KEY']
endpoint = "https://api.bing.microsoft.com/v7.0/search"

queries = ["recent literature on multi-agent systems", "2023 research papers on multi-agent systems", "latest studies on multi-agent systems"]

headers = {'Ocp-Apim-Subscription-Key': subscription_key}

for query in queries:
    try:
        params = {'q': query}
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()['webPages']['value']
        for result in search_results:
            print(f"Title: {result['name']}")
            print(f"URL: {result['url']}")
            print(f"Snippet: {result['snippet']}\n")
    except Exception as ex:
        print(f"An error occurred: {ex}")

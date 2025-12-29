
import json
import os
import requests

os.environ['BING_SEARCH_V7_SUBSCRIPTION_KEY'] = "c350758795414502b5f7e994ec9418f0"
subscription_key = os.environ['BING_SEARCH_V7_SUBSCRIPTION_KEY']
endpoint = "https://api.bing.microsoft.com/v7.0/search"

queries = ["multi-agent systems research papers", "multi-agent systems literature review", "recent trends in multi-agent systems"]

headers = {'Ocp-Apim-Subscription-Key': subscription_key}

for query in queries:
    try:
        params = {'q': query}
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        print("The search results are:")
        print(response.json()['webPages']['value'][0]['snippet'])
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as ex:
        print(f"An error occurred: {ex}")

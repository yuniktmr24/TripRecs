import json
import requests

# Define the URL for your Solr update endpoint, including the commit parameter
solr_url = 'http://localhost:8983/solr/Airfares/update/json/docs?commit=true'
headers = {'Content-type': 'application/json'}

# Function to post data to Solr in batches
def post_to_solr(data_batch):
    json_data = json.dumps(data_batch)  # Convert list of dicts to JSON string
    response = requests.post(solr_url, data=json_data, headers=headers)
    return response

# Function to load JSON data and process in batches
def process_file(file_path, batch_size=1000):
    with open(file_path, 'r') as file:
        data = json.load(file)  # Load data from file
        total = len(data)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = data[start:end]
            response = post_to_solr(batch)
            print(f"Processed batch from {start} to {end}: {response.status_code} - {response.text}")

# Specify the path to your JSON file
file_path = './Merged_travel.json'

# Call the function to process the file
process_file(file_path)

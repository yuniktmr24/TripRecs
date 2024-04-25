import json
import requests

# URL to your Solr core for posting data
solr_url = 'http://localhost:8983/solr/Airfares/update/json/docs?commit=true'

# Header to specify the content type as JSON
headers = {
    'Content-type': 'application/json'
}

# Sample data - replace this with your actual data loading method
data = [
    {"id": "1", "destination": "00.Haleakala_National_Park", "fare_amount": 1800.5, "currency": "USD", 
     "departure_date": "2024-04-20T00:00:00Z", "return_date": "2024-04-30T00:00:00Z", "airline": "Qantas", 
     "departure_city": "Los Angeles", "arrival_city": "Maui", "flight_duration_hours": 5, "booking_reference": "XYZ123"}
]

# Convert the Python dictionary to a JSON string
json_data = json.dumps(data)

# Post the data to Solr
response = requests.post(solr_url, data=json_data, headers=headers)

# Print the response from Solr
print(response.text)

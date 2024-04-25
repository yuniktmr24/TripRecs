import json

# Function to read a JSON file and return the data
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to write data to a JSON file
def write_json_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Paths to the JSON files
file_path1 = 'Revised_Travel_Data_500K.json'
file_path2 = 'Additional_Travel_Data_500K.json'
merged_file_path = 'Merged_travel.json'

# Read data from both files
data1 = read_json_file(file_path1)
data2 = read_json_file(file_path2)

# Merge the data
merged_data = data1 + data2

# Write the merged data to a new JSON file
write_json_file(merged_data, merged_file_path)

print("Files have been merged and saved as", merged_file_path)

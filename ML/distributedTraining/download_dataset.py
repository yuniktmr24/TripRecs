import urllib.request
from zipfile import ZipFile
from io import BytesIO

url = "https://udacity-dlnfd.s3-us-west-1.amazonaws.com/datasets/landmark_images.zip",

print(f"Downloading and unzipping {url}. This will take a while...")

with urllib.request.urlopen(url) as resp:

    with ZipFile(BytesIO(resp.read())) as fp:
        fp.extractall(".")

print("done")
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
# or: requests.get(url).content

url = "https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip"
resp = urlopen(url)
myzip = ZipFile(BytesIO(resp.read()))
with open("data/raw/filtered.tsv", "wb") as f:
    f.write(myzip.open('filtered.tsv').read())
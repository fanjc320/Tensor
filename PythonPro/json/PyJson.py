import json
from pprint import pprint

#with open("json.txt",'rU','utf-8') as f:
with open("json.txt") as f:
	data = json.load(f)
	
pprint(data)
print("---------")
pprint(data["maps"][0]["id"])
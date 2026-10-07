import argparse
import json
import urllib.parse
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--results", required=True)
p.add_argument("--url", default="http://127.0.0.1:8000")
a = p.parse_args()

url = a.url + "/api/results/import?" + urllib.parse.urlencode({"path": a.results})
req = urllib.request.Request(url, method="POST")
with urllib.request.urlopen(req, timeout=60) as r:
    print(json.loads(r.read().decode()))

from http.client import responses

import requests
import json
import pickle as pk

def submit(results, url="http://"):
    res = json.dumps(results)
    # print(res)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['results']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")


            
with open(r"to_submit_2_2.pickle", "rb") as output_file:
    to_submit = pk.load(output_file)
    
submit(to_submit)

# accuracy is [0.3235294117647059, 0.4803921568627451, 0.5392156862745098] cosface baseline
# accuracy is [0.3333333333333333, 0.5196078431372549, 0.5588235294117647] + face crop
# accuracy is [0.8137254901960784, 0.8627450980392157, 0.8823529411764706] + beter facecrop
# accuracy is [0.8235294117647058, 0.8823529411764706, 0.9019607843137255] + sphereface normalization
# accuracy is [0.8627450980392157, 0.9117647058823529, 0.9215686274509803] + finetuning with lfw
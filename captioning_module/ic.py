import json as js
import subprocess as sp
import requests

defimg = "shot.jpg"

def write(url):
    response = requests.get(url)
    if response.status_code == 200:
        with open(defimg, 'wb') as f:
            f.write(response.content)

def fit(image=defimg,manual=False,url=""):
    if manual:
        pass
    else:
        write(url)
        pass
    x = 'curl -X POST "http://max-image-caption-generator.max.us-south.containers.appdomain.cloud/model/predict" -H  "accept: application/json" -H  "Content-Type: multipart/form-data" -F "image=@{};type=image/jpeg"'.format(image)
    return x

def prediction_for(command,shl=True,counter=False,i=0):
    data = sp.check_output(command, shell=shl)
    res = js.loads(data.decode('utf8'))
    if counter:
        print("Predictions_for ({}.jpg):\n{}".format(i,js.dumps(res['predictions'], indent=2)))
        print('*'*60)
    return res['predictions'][0]['caption']

if __name__ == '__main__':
    output=[]
    cmd = fit('image.jpeg',manual=True)
    y = prediction_for(cmd)
    output.append("prediction:   '{}'".format(y))

    for elem in output:
        print(elem)

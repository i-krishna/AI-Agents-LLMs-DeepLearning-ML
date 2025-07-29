import requests

url = 'http://localhost:5000/predict' 

# or Deplot to AWS Sagemaker & infer that end point 
# Ex: https://github.com/i-krishna/Business-Analytics/blob/main/Data-Science/Python/ai_model_integration.py
# https://github.com/i-krishna/Business-Analytics/blob/main/Data-Science/Python/aws_model_training.py 

files = {'file': open('test_image.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
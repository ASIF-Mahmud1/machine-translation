# Translator

This is a full-stack end-to-end application, that lets you translate language from English to French
Frontend is a mobile application based on react native. Backend is based on a python framework: Fast API.


# Folder Structure

**client** : source code for react native application

**server**: source code for REST-ful application based on Fast API. Training machine learning model along with dataset

## client
1. cd client

2. npm install

3. Open a terminal run:  _**npm start**_
## server
### Run the Application in your machine
 1.  cd *server*
 2.  create  a virtual environment:  ***python3.8 -m venv venv/*** 
 3.  run the   virtual environment:  ***source venv/bin/activate*** 
 4. install dependencies : ***pip install -r requirements.txt***
 5.  run the server :   ***uvicorn index:app --reload***
 ### Deploy in Heroku with Docker
 Follow this article :  https://testdriven.io/blog/fastapi-machine-learning/
### Run Docker Image in your machine 
 1. Build Docker image :    ***sudo docker build -t registry.heroku.com/polar-sea-42815/web  .***
 2. ***docker run --name fastapi-m -e PORT=8000 -p 8000:8000 -d  registry.heroku.com/polar-sea-42815/web:latest***
  or ...
 3. ***sudo docker run  -e PORT=8000 -p 8000:8000 -d registry.heroku.com/polar-sea-42815/web:latest***
 5.  Open in browser: http://localhost:8000/





  

# Keywords
Python, Fast API, REST API, docker,  machine learning, NLP

## Notes
For futher queries, please contact me via email at :  [asif01050105@gmail.com](mailto:asif01050105@gmail.com)  - Asif Mahmud


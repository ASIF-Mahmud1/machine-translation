
# How To Run Server Application

 
- ###    To Activate Virtual Environment
 1.  `conda env create  --file environment.yml --prefix ./venv`   

 2.  `conda activate /Users/learn/Desktop/Projects/machine-translation/server/venv`

 3. `! pip install numpy --upgrade`

 4. from the root of this directory run: 
	

    jupyter-notebook

 5. From vs code change the kernel.
		conda install -c conda-forge keras-preprocessing

- ###    Run the Server

  1. `uvicorn index:app --reload`

  2. Open in browser: http://localhost:8000/
  
  **Note**: You can use the ngrok url to test api in the mobile app
  

# Folder Structure
index.py :is the entry point 
inference.py:  preprocess the sentence that it gets from server and outputs the translated sentence. Model is loaded in this file that can translate sentence from hindi to english
[envirnment.yml](https://github.com/ASIF-Mahmud1/machine-translation) : has all the modules that need to be installed

# Keywords

Python, Fast API, REST API, ngrok, machine learning, NLP, machine translation 

  

## Notes

For futher queries, please contact me via email at :
 [asif01050105@gmail.com](mailto:asif01050105@gmail.com)  
 [1421015@iub.edu.bd](mailto:1421015@iub.edu.bd)
  
  **Asif Mahmud**

# SkillGPT: a RESTful API service for skill extraction and standardization using a Large Language Model
Nan Li, Bo Kang, Tijl De bie

IDLAB - Department of Electronics and Information Systems (ELIS), Ghent University

Submitted to the ECML-PKDD demo track 2023

### [Paper](https://arxiv.org/abs/2304.11060)

### [Demo Video](http://bokang.io/videos/SkillGPT.mp4)

## Requirements
- Environment
    - Python 3.8 or later
    - Docker
    - Redis

## Installation
1. Make sure you have all the requirements listed above

2. Clone the repository
    ``` bash
    git clone https://github.com/aida-ugent/SkillGPT.git
    ```
    
3. Navigate to the directory where the repository was downloaded
    ``` bash
    cd SkillGPT
    ```

4. Install the required dependencies
    ``` bash
    pip install -r requirements.txt
    ```
    
5. Configure SkillGPT
    1. Locate the file named .env.template in the main /SkillGPT folder.
    2. Create a copy of this file, called .env by removing the template extension.
    3. Open the .env file in a text editor.
    4. Enter Model server info as well as Redis server info.
    5. Save and close the .env file.

## Environment vairable setup
Set the following settings in `.env`
  ``` bash 
  API_HOST="127.0.0.1" # the IP or domain to launch the api gateway
  API_PORT=21002
  REDIS_HOST=localhost # the IP or domain of the running redis instance
  REDIS_PORT=6379
  MODEL_PATH=models/vicuna_13b # the path to Huggingface AutoModelForCausalLM model
  ```

## Usage
1. Launch docker service
    ``` bash
    sudo docker run --name redis-stack-server -p 6379:6379 redis/redis-stack-server:latest
    ```
    
2. Run `api` Python module in your terminal
    ``` bash
    python -m api
    ```

3. Launch gradio interface
    ``` bash
    python gradio_server.py
    ```

4. Process via API requests. See examples in `api_request.ipynb`.

5. (Optional) initalize Redis vector DB. See example in the last cell "Initialize ESCO embeddings" in `api_request.ipynb`.

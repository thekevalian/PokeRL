### Installation Steps
1. Clone the Repo:  
    ```git clone --recurse-submodules https://github.com/thekevalian/PokeRL.git```
2. Setup the Showdown Server: 
    ```
    git clone https://github.com/smogon/pokemon-showdown.git
    cd pokemon-showdown
    npm install
    cp config/config-example.js config/config.js
    node pokemon-showdown start --no-security
    ```
3. Create Python Environment and Install Dependencies
    ```
    python -m venv .venv
    .\.venv\Scripts\activate
    pip install -r requirements.txt
    ```
4. Step 2 should have launched the local pokemon showdown server. You should now be able to run rl_example.py
    ```
    python rl_example.py
    ```

### Citations
1. https://github.com/smogon/pokemon-showdown
2. https://github.com/hsahovic/poke-env

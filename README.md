### Installation Steps
1. Clone the Repo:  
    ```git clone --recurse-submodules https://github.com/thekevalian/PokeRL.git```
2. Setup the Showdown Server: 
    ```
    cd PokeRL
    cd pokemon-showdown
    npm install
    cp config/config-example.js config/config.js
    ```
3. Create Python Environment and Install Dependencies (You will need to modify some steps for MacOS and Linux)
    ```
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    git clone https://github.com/hsahovic/poke-env.git
    cd poke-env
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    cd ..
    pip install -e poke-env
    ```
4. You are all setup. Checkout the rl_train, rl_test and rl_play scripts 

When running all poke-env scripts, make sure that the Pokemon Server is up. You can turn it on by running the following command in your terminal:
```
node pokemon-showdown start --no-security
```



### Citations
1. https://github.com/smogon/pokemon-showdown
2. https://github.com/hsahovic/poke-env


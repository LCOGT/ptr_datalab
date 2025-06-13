# Datalab Backend

This application is the backend server for the PhotonRanch Datalab. It is a django application with a REST API for communicating with the Datalab UI.

## Prerequisites
-   [Python](https://www.python.org/downloads/)
-   [Poetry](https://python-poetry.org/docs/)
-   [Redis](https://redis.io/docs/latest/operate/oss_and_stack/install/install-stack/)


## Bare Metal Development
1. Create a virtualenv for this project and entering it: 
```
    python -m venv ./venv
    source ./venv/bin/activate
```
2. Install the dependencies:
```
    poetry install
```
3. Run the migrations to setup the database. The project is uses a local sqlite database. You can change that to a postgres one if you want.
```
    ./manage.py migrate
```
4. Start a Redis Server at port `6379`.
```
    redis-server
```
5. Start the dramatiq worker threads, here we use a minimal number of processes and threads
```
    ./manage.py rundramatiq --processes 1 --threads 2
```
6. Start the Django server
```
    ./manage.py runserver
```

## Nix development
For this mode of development, you must install:
-   [nix with flakes support](https://github.com/LCOGT/public-wiki/wiki/Install-Nix)

Then to develop, run these commands:
-   `nix develop --impure` to start your nix development environment - **called anytime you use a new terminal**
-   `ctlptl apply -f local-registry.yaml -f local-cluster.yaml` to start up the registry and cluster - **should only need to be called one time within the nix environment**
-   `skaffold dev -m deps` to start the dependencies - **run this in a different tab to keep running during development or use 'run' instead of 'dev'**
-   Copy `./k8s/envs/local/secrets.env.changeme` to a version without `.changeme` and fill in values for connecting to the appropriate services.
-   `skaffold dev -m app --port-forward` to start the servers and worker. This will auto-redeploy as you make changes to the code.
-   Once running you will need to initialize your account with the server before using the API - the easiest way to do this is to login one time to the admin interface at `http://127.0.0.1:8080/admin`

### Connecting a frontend
You can also run a local [datalab-ui](https://github.com/LCOGT/datalab-ui) to connect to your datalab. Assuming you've cloned that repo:
-   Change the `./public/config/config.json` "datalabApiBaseUrl" to be `http://127.0.0.1:8080/api/` or wherever your backend is deployed to
-   `npm install` to install the libraries
-   `npm run serve` to run the server at `http://127.0.0.1:8081` assuming your backend was already running (otherwise it will try to be :8080)

## API Structure
The application has a REST API with the following endpoints you can use. You must pass your user's API token in the request header to access any of the endpoints - the headers looks like `{'Authorization': 'Token 123456789abcdefg'}` if you are using python's requests library.

### Input Data structure
Datasessions can take an `input_data` parameter, which should contain a list of data objects. The current format is described below, but this is probably something that will evolve as we learn more how we are using it.

    session_input_data = [
        {
            'type': 'fitsfile',
            'source': 'archive',
            'basename': 'mrc1-sq005mm-20231114-00010332'
        },
        {
            'type': 'fitsfile',
            'source': 'archive',
            'basename': 'mrc1-sq005mm-20231114-00010333'
        },
    ]

Data operations can have a varying set of named keys within their `input_data` that is specific to each operation. For example it would look like this for an operation that just expects a list of files and a threshold value:

    operation_input_data = {
        'input_files': [
            {
                'type': 'fitsfile',
                'source': 'archive',
                'basename': 'mrc1-sq005mm-20231114-00010332'
            }
        ],
        'threshold': 255.0
    }

### Datasessions API
#### Create a new Datasession
`POST /api/datasessions/`

    post_data = {
        'name': 'My New Session Name',
        'input_data': session_input_data
    }

#### Get all existing Datasessions
`GET /api/datasessions/`

#### Get Datasession by id
`GET /api/datasessions/datasession_id/`

#### Delete Datasession by id
`DELETE /api/datasessions/datasession_id/`

### Operations API
Available Operations are introspected from the `data_operations` directory and must implement the `BaseDataOperation` class. I expect we will add more flesh to those classes when we actually start using them. 
#### Get Operations for a Datasession
`GET /api/datasessions/datasession_id/operations/`

#### Create new Operation for a Datasession
`POST /api/datasessions/datasession_id/operations/`
    
    post_data = {
        'name': 'Median',  # This must match the exact name of an operation
        'input_data': operation_input_data
    }

#### Delete Operation from a Datasession
`DELETE /api/datasessions/datasession_id/operations/operation_id/`

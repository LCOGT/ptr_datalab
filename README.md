# Getting Started with Datalab Backend

This application is the backend server for the PhotonRanch Datalab. It is a django application with a REST API for communicating with the Datalab UI.

## Prerequisites
-   [Python v3.10 - v3.12](https://www.python.org/downloads/)
-   [Poetry](https://python-poetry.org/docs/)
-   [Redis](https://redis.io/docs/latest/operate/oss_and_stack/install/install-stack/)
-   [AWS Cli](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html#getting-started-install-instructions)


## Bare Metal Development
### Setup
1. Create a virtualenv for this project and entering it: 
```
    python -m venv ./venv
    source ./venv/bin/activate
```

2. Install pyproject.toml:
```
    poetry install
```
  - If the previous step fails to install check your python version is not `>= 3.13` if so switch to `3.11`,
    ```
        brew install python@3.11
        poetry env use <<install/path/to/python3.11>>
        poetry shell
        pip install -e .
    ```

3. Run the migrations command to setup the local sqlite database
```
    ./manage.py migrate
```
4. Create a Django superuser, for convenience you can use your LCO credentials
```
    python manage.py createsuperuser
```
5. Start the Django app and navigate to the /admin panel (e.g http://127.0.0.1:8000/admin)
```
    ./manage.py runserver

    Django version 4.2.20, using settings 'datalab.settings'
    Starting development server at http://127.0.0.1:8000/
    Quit the server with CONTROL-C 
```
6. Set up LCO credentials. Navigate to the Auth Profile's Tab and create an authuser from your superuser account and add your [LCO archive api token](https://observe.lco.global/accounts/profile) to the token field
7. Log into AWS on the browser and navigate to IAM roles to find your user profile. On your profile you should create/use an access and secret-access key that is your personal token to talk to aws. Permissions to access the datalab bucket will need to be requested from the Datalab Dev team (Jon, Lloyd, Carolina) as of Jun 2025.
8. Once you have your Access Key and Secret Access Key from a datalab dev, run the configure command, and then confirm proper configuration with the `get-caller-identity` command
```
    > aws configure

    AWS Access Key ID [****************UVN2]:
    AWS Secret Access Key [****************d7X5]:
    Default region name [us-west-2]:
    Default output format [json]:

    > aws sts get-caller-identity

    {
        "UserId": "****************TIFAX",
        "Account": "********0537",
        "Arn": "arn:aws:iam::********0537:user/datalab-server"
    }
```
9. Finally Restart your machine to update it's aws credentials cache

### Running the Django App
1. Start up a Redis Server that will faciliate caching as well as the rabbitmq queue. To do this make sure you have Redis installed and then start a server at port 6379
```
    // run in shell
    redis-server
    // run in background
    brew services start redis
```

2. Start the dramatiq worker threads
```
    ./manage.py rundramatiq --processes 1 --threads 2
```
3. Start the Django server
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
-   Follow steps 4. - 9. in The Bare Metal Development section to setup Django auth, LCO creds, and AWS creds

### Connecting a frontend
You can also run a local [datalab-ui](https://github.com/LCOGT/datalab-ui) to connect to your datalab.
1. Change the `./public/config/config.json` `"datalabApiBaseUrl"` to be `http://127.0.0.1:8080/api/` or wherever your backend is deployed to
2. `npm install` to install the libraries
3. `npm run serve` to start the frontend

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

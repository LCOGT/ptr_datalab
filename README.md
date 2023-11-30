# Datalab Backend

This application is the backend server for the PhotonRanch Datalab. It is a django application with a REST API for communicating with the Datalab UI.

## Prerequisites
-   Python >= 3.9
-   Django >= 4


## Local Development
Start by creating a virtualenv for this project and entering it: 

    python -m venv /path/to/my/virtualenv
    source /path/to/my/virtualenv/bin/activate

Then install the dependencies:

    pip install -e .

The project is configured to use a local sqlite database. You can change that to a postgres one if you want but sqlite is easy for development. Run the migrations to setup the database and then you can run the server.

    ./manage.py migrate
    ./manage.py runserver

If you want to start with some test data, run this management command one time after running migrations to add some test data to the database. The test data creates two datasessions with some operations, and a user `test_user` with password `test_pass` and API token `123456789abcdefg`.

    ./manage.py populate_test_data

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

## ROADMAP
* Come up with operation `wizard_description` format and add endpoint to get them for all available operations so the frontend can auto-create UI wizards for new operations.
* Figure out user accounts between PTR and datalab - datalab needs user accounts for permissions to gate access to only your own sessions.
* Implement operations to actually do something when they are added to a session
    * Figure out caching and storage of intermediate results
    * Figure out asynchronous task queue or temporal for executing operations
    * Add in operation results/status to the serialized operations output (maybe to the model too as needed)

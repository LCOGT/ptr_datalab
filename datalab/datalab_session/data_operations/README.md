# DataOperation Development

Welcome datalab developer, this guide exists to walk you through the creation of new `Data Operations` from the parent `BaseDataOperation.py` class

## Getting Started
Data Operations are classes based on the `BaseDataOperation.py` Parent class.
Data Operations are run in dramatiq workers which is an asynchronus task queue that lets the backend run many operations at once.

### BaseDataOperation.py Anatomy
We'll go over the main methods you will need to implement for a new data operation here, there are more funcitons than these but those are not as important

- **`name()`** : Name of operation shown on frontend
- **`description()`** : A text description of the DataOperation, to be shown to the user
- **`wizard_description()`** : A json-formatted DSL describing the expected inputs for this DataOperation, for the frontend to create custom input widgets for it in a wizard
- **`generate_cache_key()`** : A key attribute of a data operation. The cache key is the unique identifier for the operation and it's set of inputs. This key is used to identify the output data in the s3 bucket, as well as cache operations so identical operations arn't rerun. Instead fetching the cached output
- **`operate()`** : entry point to the code that is run in the dramatiq worker. Main bulk of the operation logic
- **`set_operation_progress()`** : updates the operation progress from the value of 0 - 1 representing a percentage completion value
- **`set_output()`** : should be passed a formatted output dictionary that contains s3 links to the completed output
- **`set_failed()`** : fails the operation with an optional message

### Typical Data Operation flow
1. Frontend kicks off a data operation with a set of inputs defined by the wizard description of the data operation. Currently there is only the format of `fits` in `format.py` but future operations may need `number`, `string`, `area`, etc.
2. The Django Viewsets.py routes the call to the proper data operation and passes the `operate()` method to a worker
3. The data operation then typically fetches input FITs files from the archive/datalab bucket when you use the `input_data_handler` object with basenames of the FITs files.
4. Work specific to that data operation happens (e.g a median extracts the numpy arrays of the FITs SCI hdu data tables and averages them )
5. An output is created with an `output_data_handler` object when you pass it the finished numpy data.
6. Output is set with `set_output()` and the dictionary is returned to the frontend to be displayed

#### Something goes wrong in the operation? 
Raise a `ClientAlertException` to pass a message back to the frontend which will be dislayed as an alert. 
This is caught in `tasks.py` and where it sets the operation as failed and puts your error message in the message field of the data operation class.

#### Skipping `input_data_handler` and `output_data_handler`
Some methods require more granular control of getting the input data and creating the output. For example the Color_Image operation creates a color image so it skips using the data_output_handler class. In this case you'll need to use the methods in `file_utils.py` and `s3_utils.py`

## File Reference
**`data_operation.py`**
Parent class that should be extended to create any new child data operation, has a `cache_key` that is automatically made from the normalized input and operation type.
**`filecache.py`**
Datalab Server's temporary file management system. Should be your go to method of downloading and saving FITs files. Will monitor available space left on the server's enviorment and delete the LRU (Least Recently Used) files. 
**`input_data_handler`**
Will fetch the data for you and offers methods to access its headers
**`output_data_handler`**
Pass it data and it creates output FITs files and images. Has methods to return a properly formatted output dictionary to send to `BaseDataOperation.set_output()`
**`s3_utils.py`**
Utils for fetching/checking existence/uploading files to the datalab s3 bucket where we store datalab outputs
**`file_utils.py`**
Utils for working with the common filetypes in datalab

## Walkthrough of creating a data operation
In this section we'll walk step by step creating a mock data operation called `Increase_Brightness`

1. Create a new file `increase_brightness.py` in the directory `/datalab_session/data_operations`
2. import numpy, User, input handler, output handler, the base operation class, and client alert excception
```
import numpy as np
from django.contrib.auth.models import User

from datalab.datalab_session.data_operations.input_data_handler import InputDataHandler
from datalab.datalab_session.data_operations.fits_output_handler import FITSOutputHandler
from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.exceptions import ClientAlertException
```
3. define the `name()`, `description()` methods
```
@staticmethod
    def name():
        return 'Increase Brightness'
    
  @staticmethod
  def description():
      return """Increases the image's brightness by a value N"""
```
4. Create the wizard description that will define how the UI looks and what inputs will be available to the user. The decription below will create an image input where the user can pick one N images, and a number input to brighten those images by
```
@staticmethod
    def wizard_description():
        return {
            'name': Increase_Brightness.name(),
            'description': Increase_Brightness.description(),
            'category': 'image',
            'inputs': {
                'input_files': {
                    'name': 'Image to Brighten',
                    'description': 'A image to increase it's brightness',
                    'type': Format.FITS,
                    'minimum': 1,
                    'maximum': 999
                }
                `increase_by`: {
                    'name': 'Value to brighten by',
                    'description': 'Value to add to the image',
                    'type': Format.number,
                    'minimum': 1,
                    'maximum': 999
                }
            }
        }
```
**Note** As of Jun 2025 we only support the `fits` input type, to add things like strings, numbers, or more custom inputs you'll need to create the format in `format.py` and then handle that format in the `datalab-ui` repo. The UI reads the wizard description to build the input fields you'll have to define what that will look like. 

1. Now we can start writing the `operate()` method, we'll fetch all the input with the names we defined in our wizard description
```
input_list = self.input_data.get('input_files', [])
increase_by = self.input_data.get('increase_by', [])
```
1. We'll loop over the inputs and do a couple things. Create `input_data_handlers` for each, get their sci_data and add the `increase_by` value to the whole 2d array, and create the output using `fits_output_handler`
```
output_files = []
  ~for index, input in enumerate(input_list, start=1):
    with InputDataHandler(user, input['basename'], input['source']) as image:
      self.set_operation_progress(0.9 * (index-0.5) / len(input_list))

      brightened_data = np.add(image.sci_data, increase_by)

      comment = f'Datalab Increase_By on file {input_list[index-1]["basename"]}'
      output = FITSOutputHandler(f'{self.cache_key}', brightened_data, self.temp, comment, data_header=image.sci_hdu.header.copy()).create_and_save_data_products(Format.FITS, index=index)
      output_files.append(output)
      self.set_output(output_files)
      self.set_operation_progress(0.9 * index / len(input_list))
```
1. Finally we send the output to the `set_output()` method and we're done!
```
  log.info(f'Increase By output: {output_files}')
  self.set_output(output_files)
  self.set_operation_progress(1.0)
  self.set_status('COMPLETED')
```

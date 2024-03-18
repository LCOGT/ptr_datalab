import boto3
import requests
import logging
log = logging.getLogger()
log.setLevel(logging.INFO)

bucket_name = 'datalab-operation-output-bucket'
archive_frames_url = 'https://datalab-archive.photonranch.org/frames/'

def store_fits_output(item_key, fits_buffer):
  log.info(f'Adding {item_key} to {bucket_name}')

  s3 = boto3.resource('s3')
  response = s3.Bucket(bucket_name).put_object(Key = item_key, Body = fits_buffer.getvalue())
  return response

def find_fits(basename):
  query_params = {'basename_exact': basename }

  response = requests.get(archive_frames_url, params=query_params)
  response.raise_for_status()

  if response.status_code != 204:
      image_data = response.json()
      results = image_data.get('results', None)
      
      return results

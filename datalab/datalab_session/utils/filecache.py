import os
import time
import logging
from pathlib import Path

from django.core.cache import cache
from django.conf import settings
from django.contrib.auth.models import User

from datalab.datalab_session.utils.s3_utils import download_fits

log = logging.getLogger()
log.setLevel(logging.INFO)


class FileCache():
    ''' Class for managing file access. It uses django_redis to support a LRU cache of files in redis.
        It is meant to be a drop-in non-contextmanager replacement for the get_fits method we previously
        used, as well as having an add_file_to_cache method which adds a created dataproduct that is on
        the filesystem to the file cache.
    '''
    LOCK_TIMEOUT = 5  # lock timeout in seconds
    def __init__(self):
        self.lock_name = f"{settings.CONTAINER_TYPE}_filecache_lock"
        self.list_name = f"{settings.CONTAINER_TYPE}_filecache_list"
        self.total_size_name = f"{settings.CONTAINER_TYPE}_filecache_size"
        self.client = cache.client.get_client()

    def _increment_file_usage(self, file_key):
        # Assumes you already have an the cache lock
        # Moves file_key to the front of the LRU list of files
        log.debug(f"_increment_file_usage moving {file_key} to most recently accessed")
        self.client.lrem(self.list_name, 0, file_key)
        self.client.lpush(self.list_name, file_key)

    def _remove_least_recently_used(self):
        # Assumes you already have an the cache lock
        # Removes least recently used file from the cache and deletes it from the local file system
        file_key = self.client.rpop(self.list_name)
        if file_key:
            # Get the file_details and remove those as well
            file_details = self.client.hgetall(file_key)
            log.info(f"_remove_least_recently_used removing {file_key} with size {int(file_details.get(b'size', 0))}")
            self.client.hdel(file_key, 'file_path', 'size')
            # Delete the file from the filesystem
            Path(file_details.get(b'file_path').decode('utf-8')).unlink(missing_ok=True)
            # Removes the file size from the size counter
            return int(self.client.decrby(self.total_size_name, int(file_details.get(b'size', 0))))
        return int(self.client.get(self.total_size_name))

    def add_file_to_cache(self, file_path):
        ''' This is called to add an already existing file that is in the temp dir into the file cache.
            This will mainly be used for adding operation output fits files into the file cache so they can
            persist along with other files.
        '''
        # Verify that the file exists on the local filesystem
        if not os.path.isfile(file_path):
            return False

        file_key = os.path.basename(file_path).split('.')[0]
        file_size = os.path.getsize(file_path)
        # Add it to the cache here
        with cache.lock(self.lock_name, timeout=self.LOCK_TIMEOUT):
            current_size = int(self.client.get(self.total_size_name))
            # First make sure we have enough room for this file, and if not delete things from the cache until there is enough room
            while current_size + file_size >= settings.FILECACHE_TOTAL_SIZE:
                log.debug(f"add_file_to_cache for {file_path}: Cache total size {current_size + file_size} exceeds the limit {settings.FILECACHE_TOTAL_SIZE} - removing LRU file")
                current_size = self._remove_least_recently_used()
            # Now there is enough room so add the file to the cache
            file_details = {
                'file_path': file_path,
                'size': file_size
            }
            self.client.hset(file_key, mapping=file_details)
            self.client.incrby(self.total_size_name, file_size)
            self._increment_file_usage(file_key)
        return True

    def _download_file_to_cache(self, basename: str, source: str = 'archive', user: User = User.objects.none):
        # First place the fits details into the cache, then download the file, then update the cache with the final file size
        file_key = f"{source}_{basename}"
        file_name = f"{file_key}.fits.fz"
        file_path = os.path.join(settings.TEMP_FITS_DIR, file_name)
        log.debug(f"_download_file_to_cache for {file_key}")
        try:
            log.info(f"_download_file_to_cache for {file_key}: initial cache set, downloading file to {file_path}")
            download_fits(file_path, basename, source, user)
            # Now download is finished, so get the file size and update the cache with it
            file_size = os.path.getsize(file_path)
            log.info(f"_download_file_to_cache for {file_key}: download complete with file size {file_size}")
            with cache.lock(self.lock_name, timeout=self.LOCK_TIMEOUT):
                self.client.hset(file_key, key='size', value=file_size)
                total_size = int(self.client.incrby(self.total_size_name, file_size))
                log.debug(f"_download_file_to_cache for {file_key}: Cache total size is now {total_size}")
                # If this file increased total size above the cache maximum, delete files from cache until it is below the maximum
                while total_size >= settings.FILECACHE_TOTAL_SIZE:
                    log.info(f"_download_file_to_cache for {file_key}: Cache total size {total_size} exceeds the limit {settings.FILECACHE_TOTAL_SIZE} - removing LRU file")
                    total_size = self._remove_least_recently_used()

            return file_path
        except Exception as e:
            log.error(f"Failed to download file {basename} from {source}: {repr(e)}")
            # Failed to download file, so clean up cache here
            with cache.lock(self.lock_name, timeout=self.LOCK_TIMEOUT):
                # Don't have to do anything with total size since that was never incremented for the file yet
                self.client.hdel(file_key, 'file_path', 'size')
                self.client.lrem(self.list_name, 0, file_key)
            # Raise an exception here since we failed to download the file
            raise

    def get_fits(self, basename: str, source: str = 'archive', user: User = User.objects.none):
        ''' This attempts to get the file out of the cache and increment its usage. If the file isn't in the cache,
            or if its in the cache but not on the filesystem, then the file will be redownloaded from S3 and placed
            in the cache. Returns the local temp dir file_path to the downloaded file.
        '''
        GET_FITS_TIMEOUT = 30
        start_time = time.time()

        basename = basename.replace('-large', '').replace('-small', '')
        file_path = self._get_fits_helper(basename, source, user)
        while file_path is None:
            if time.time() - start_time > GET_FITS_TIMEOUT:
                log.error(f"Timeout reached while waiting for {basename} to download.")
                raise TimeoutError(f"Failed to retrieve {basename} within {GET_FITS_TIMEOUT} seconds.")
            
            time.sleep(0.1)
            file_path = self._get_fits_helper(basename, source, user)

        return file_path

    def _get_fits_helper(self, basename: str, source: str = 'archive', user: User = User.objects.none):
        file_key = f"{source}_{basename}"
        log.debug(f"_get_fits_helper for {file_key}")
        # First see if the file is in the file cache and increment its name on the LRU list
        with cache.lock(self.lock_name, timeout=self.LOCK_TIMEOUT):
            file_details = self.client.hgetall(file_key)
            if file_details and file_details.get(b'file_path') and file_details.get(b'size'):
                if int(file_details.get(b'size')) == -1:
                    log.debug(f"_get_fits_helper for {file_key}: File is currently downloading")
                    # This means the file is currently loading / hasn't finished downloading yet, so we should sleep and try again later
                    return None
                elif os.path.isfile(file_details.get(b'file_path').decode('utf-8')):
                    log.debug(f"_get_fits_helper for {file_key}: File is retrieved and returned")
                    self._increment_file_usage(file_key)
                    return file_details.get(b'file_path').decode('utf-8')
                else:
                    log.warning(f"_get_fits_helper for {file_key}: File details exist but os.path.isfile fails")
            # Otherwise we have a problem where the file doesn't exist locally even though its in the cache - or its not in the cache at all
            # Here we will place it in the cache and then initiate downloading it to the filesystem
            log.debug(f"_get_fits_helper for {file_key}: File doesn't currently exist and will be downloaded")
            file_name = f"{file_key}.fits.fz"
            file_path = os.path.join(settings.TEMP_FITS_DIR, file_name)
            file_details = {
                'file_path': file_path,
                'size': -1  # Negative size implies download is in progress
            }
            self.client.hset(file_key, mapping=file_details)
            self._increment_file_usage(file_key)
        return self._download_file_to_cache(basename, source, user)

    def clear_cache(self):
        ''' Clears out the file cache - assumes you already have a lock open from the calling process
        '''
        filecache_len = self.client.llen(self.list_name)
        filecache_list = self.client.lrange(self.list_name, 0, filecache_len)
        self.client.delete(self.list_name)
        self.client.set(self.total_size_name, 0)
        for file_key in filecache_list:
            self.client.hdel(file_key, 'file_path', 'size')

    def reconcile_cache(self):
        ''' This looks through all the files currently on the system and rebuilds the cache using those files.
            This is mainly meant to be called on pod creation, especially when running locally, to make sure that what is in
            the redis cache matches what is in the temporary volume, because either could get blown away on pod redeploys.
        '''
        log.debug("reconcile_cache: begin reconciling cache files")
        with cache.lock(self.lock_name, timeout=60):
            # Get the files on the filesystem in the temp dir
            try:
                files = [f.decode('utf-8') for f in os.listdir(settings.TEMP_FITS_DIR) if os.path.isfile(os.path.join(settings.TEMP_FITS_DIR, f))]
            except FileNotFoundError:
                files = []
            if not files:
                # Special case where there are no temp files, i.e. temp volume was blown away, so clear the cache here
                log.warning("reconcile_cache: No files found on the temp drive - clearing cache")
                self.clear_cache()
            elif self.client.exists(self.total_size_name) == 0:
                # Special case where redis was reset and has no data in it but temp volume has files, so reset cache with those files
                log.warning("reconcile_cache: redis cache empty - resetting with current files in temp drive")
                for file_name in files:
                    file_key = os.path.basename(file_name).split('.')[0]
                    file_path = os.path.join(settings.TEMP_FITS_DIR, file_name)
                    file_size = os.path.getsize(file_path)
                    file_details = {
                        'file_path': file_path,
                        'size': file_size
                    }
                    self.client.hset(file_key, mapping=file_details)
                    self._increment_file_usage(file_key)
                    self.client.incrby(self.total_size_name, file_size)
            else:
                # Case where both redis and filesystem have stuff in them, so just reconcile the two to make sure they aggree here
                # It would be safer to blow away the cache and regenerate it, but we would lost the ordering of the LRU part
                log.warning("reconcile_cache: reconciling files from temp drive with what is in redis cache")
                filecache_len = self.client.llen(self.list_name)
                filecache_list = [f.decode('utf-8') for f in self.client.lrange(self.list_name, 0, filecache_len)]
                for file_key in filecache_list:
                    file_details = self.client.hgetall(file_key)
                    file_path = file_details.get(b'file_path').decode('utf-8')
                    file_path_base = os.path.basename(file_path)
                    if file_path_base not in files:
                        # File is in cache but no longer in the temp drive
                        self.client.lrem(self.list_name, 0, file_key)
                        self.client.decrby(self.total_size_name, int(file_details.get(b'size', 0)))
                        self.client.hdel(file_key, 'file_path', 'size')
                current_total_size = int(self.client.get(self.total_size_name))
                for file_name in files:
                    file_key = os.path.basename(file_name).split('.')[0]
                    file_path = os.path.join(settings.TEMP_FITS_DIR, file_name)
                    if file_key not in filecache_list:
                        # File is on the filesystem in the temp drive but not in the file cache - so add it here at the end
                        file_size = os.path.getsize(file_path)
                        if current_total_size + file_size < settings.FILECACHE_TOTAL_SIZE:
                            file_details = {
                                'file_path': file_path,
                                'size': file_size
                            }
                            self.client.hset(file_key, mapping=file_details)
                            current_total_size = int(self.client.incrby(self.total_size_name, file_size))
                            # Put it on the end of the LRU - first to be ejected since it wasn't already there
                            self.client.rpush(self.list_name, file_key)
                        else:
                            # We are over size - since we would put these on the end of the LRU anyway, just remove the files completely to clean them up
                            Path(file_path).unlink(missing_ok=True)
        log.debug("reconcile_cache: done reconciling cache files")

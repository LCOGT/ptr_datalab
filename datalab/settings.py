"""
Django settings for datalab project.

Generated by 'django-admin startproject' using Django 4.2.7.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/4.2/ref/settings/
"""

import os
from pathlib import Path


def str2bool(value):
    '''Convert a string value to a boolean'''
    value = value.lower()

    if value in ('t', 'true', 'y', 'yes', '1', ):
        return True

    if value in ('f', 'false', 'n', 'no', '0', ):
        return False

    raise RuntimeError(f'Unable to parse {value} as a boolean value')


def get_list_from_env(variable, default=None):
    value_as_list = []
    value = os.getenv(variable, default)
    if value:
        value_as_list = value.strip(', ').replace(' ', '').split(',')
    return value_as_list


# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_FITS_DIR = os.getenv('TEMP_FITS_DIR', os.path.join(BASE_DIR, 'tmp/fits/'))

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.getenv('SECRET_KEY', 'django-insecure-1kxt0@g^p#x3ycs&6fo()(v944zp&(jw)!2r^a4-&r4tzzihwv')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = str2bool(os.getenv('DEBUG', 'true'))

ALLOWED_HOSTS = get_list_from_env('ALLOWED_HOSTS', '*')  # Comma delimited list of django allowed hosts


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'corsheaders',
    'django_extensions',  # for debugging: shell_plus management command
    'rest_framework',
    'django_filters',
    'rest_framework.authtoken',
    'django_dramatiq',
    'ocs_authentication.auth_profile',
    'datalab.datalab_session',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'datalab.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'datalab.wsgi.application'

DRAMATIQ_BROKER = {
    'BROKER': os.getenv('DRAMATIQ_BROKER', 'dramatiq.brokers.redis.RedisBroker'),
    'OPTIONS': {
        'url': os.getenv('DRAMATIQ_BROKER_URL', 'redis://127.0.0.1:6379'),
    },
    'MIDDLEWARE': [
        'dramatiq.middleware.Prometheus',
        'dramatiq.middleware.AgeLimit',
        'dramatiq.middleware.TimeLimit',
        'dramatiq.middleware.Callbacks',
        'dramatiq.middleware.Retries',
        'django_dramatiq.middleware.DbConnectionsMiddleware',
        'django_dramatiq.middleware.AdminMiddleware',
    ]
}

DRAMATIQ_RESULT_BACKEND = {
    'BACKEND': os.getenv('DRAMATIQ_RESULT_BACKEND', 'dramatiq.results.backends.redis.RedisBackend'),
    'BACKEND_OPTIONS': {
        'url': os.getenv('DRAMATIQ_RESULT_BACKEND_URL', 'redis://localhost:6379'),
    },
    'MIDDLEWARE_OPTIONS': {
        'result_ttl': 1000 * 60 * 10
    }
}

# Defines which database should be used to persist Task objects when the
# AdminMiddleware is enabled.  The default value is 'default'.
DRAMATIQ_TASKS_DATABASE = 'default'

# AWS S3 Bitbucket
DATALAB_OPERATION_BUCKET = os.getenv('DATALAB_OPERATION_BUCKET', 'datalab-operation-output-lco-global')

# Datalab Archive
ARCHIVE_API = os.getenv('ARCHIVE_API', 'https://archive-api.lco.global')
ARCHIVE_API_TOKEN = os.getenv('ARCHIVE_API_TOKEN')
if not ARCHIVE_API_TOKEN:
    print("WARNING: ARCHIVE_API_TOKEN is missing from the environment.")

# Database
# https://docs.djangoproject.com/en/4.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': os.getenv('DB_ENGINE', 'django.db.backends.sqlite3'),
        'NAME': os.getenv('DB_NAME', BASE_DIR / 'db.sqlite3'),
        'USER': os.getenv('DB_USER', 'postgres'),
        'PASSWORD': os.getenv('DB_PASSWORD', 'postgres'),
        'HOST': os.getenv('DB_HOST', '127.0.0.1'),
        'PORT': os.getenv('DB_PORT', '5432'),
    }
}

CACHES = {
    'default': {
        'BACKEND': os.getenv('CACHE_BACKEND', 'django.core.cache.backends.redis.RedisCache'),
        'LOCATION': os.getenv('CACHE_LOCATION', 'redis://127.0.0.1:6379')
    }
}

AUTHENTICATION_BACKENDS = [
    'django.contrib.auth.backends.ModelBackend',
    'ocs_authentication.backends.OAuthUsernamePasswordBackend',
]

# This project now requires connection to an OAuth server for authenticating users to make changes
# In the OCS, this would be the Observation Portal backend
OCS_AUTHENTICATION = {
    'OAUTH_TOKEN_URL': os.getenv('OAUTH_TOKEN_URL', 'http://observation-portal-dev.lco.gtn/o/token/'),
    'OAUTH_PROFILE_URL': os.getenv('OAUTH_PROFILE_URL', 'http://observation-portal-dev.lco.gtn/api/profile/'),
    'OAUTH_CLIENT_ID': os.getenv('OAUTH_CLIENT_ID', ''),
    'OAUTH_CLIENT_SECRET': os.getenv('OAUTH_CLIENT_SECRET', ''),
    'OAUTH_SERVER_KEY': os.getenv('OAUTH_SERVER_KEY', ''),
    'REQUESTS_TIMEOUT_SECONDS': 60
}

# Password validation
# https://docs.djangoproject.com/en/4.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

REST_FRAMEWORK = {
    'TEST_REQUEST_DEFAULT_FORMAT': 'json',
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated',
    ),
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'ocs_authentication.backends.OCSTokenAuthentication',
        'rest_framework.authentication.SessionAuthentication',
    ),
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.LimitOffsetPagination',
    'PAGE_SIZE': 100,
}

# Internationalization
# https://docs.djangoproject.com/en/4.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.2/howto/static-files/

STATIC_URL = os.getenv('STATIC_URL', '/static/')
STATIC_ROOT = os.getenv('STATIC_ROOT', '/static/')

# Default primary key field type
# https://docs.djangoproject.com/en/4.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

CORS_ORIGIN_ALLOW_ALL = True
CORS_ALLOW_CREDENTIALS = True
CSRF_TRUSTED_ORIGINS = get_list_from_env('CSRF_TRUSTED_ORIGINS', 'http://localhost:8080,http://127.0.0.1:8000,http://127.0.0.1:8001,http://localhost:8000,')

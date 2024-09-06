from django.core.cache import cache

class CaptureTokenMiddleware:
    """
    Middleware to capture the Archive Authorization token from the request headers and store it in the cache
    """
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        token = request.headers.get('Authorization')
        if token:
          cache.set('archive_token', token, timeout=None)

        response = self.get_response(request)
        return response

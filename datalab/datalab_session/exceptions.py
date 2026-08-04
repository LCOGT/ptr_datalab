class ClientAlertException(Exception):
  """Custom exception for errors to be shown on the client side."""
  def __init__(self, message):
    self.message = message
    super().__init__(message)


class LightCurveError(ValueError):
  """A light curve could not be produced from the inputs as given."""

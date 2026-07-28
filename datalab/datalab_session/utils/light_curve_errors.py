class LightCurveError(ValueError):
    """
        A light curve could not be produced from the inputs as given.

        Its own module so the pipeline, the calibration strategies and the target locators can all
        raise it without importing each other. Operations catch this and re-raise it as a
        ClientAlertException, so the message reaches the user unchanged: write it for them.
    """

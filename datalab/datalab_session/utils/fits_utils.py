from typing import Any, Mapping, Sequence

import numpy as np
from astropy.io import fits

from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.utils.file_utils import get_hdu


def load_fits_image_data(
    fits_path: str,
    *,
    error_class: type[Exception] = ValueError,
) -> np.ndarray:
    try:
        data = get_hdu(fits_path, extension="SCI").data
    except ClientAlertException as exc:
        raise error_class(f"SCI image HDU is missing for {fits_path}.") from exc
    if data is None:
        raise error_class(f"SCI image HDU is empty for {fits_path}.")
    return np.asarray(data, dtype=float)


def load_fits_primary_header(
    fits_path: str,
    *,
    error_class: type[Exception] = ValueError,
) -> Mapping[str, Any]:
    try:
        sci_header = dict(get_hdu(fits_path, extension="SCI").header)
    except ClientAlertException as exc:
        raise error_class(f"SCI image HDU is missing for {fits_path}.") from exc
    with fits.open(fits_path) as hdul:
        primary_header = dict(hdul[0].header)
    return {**primary_header, **sci_header}


def load_fits_cat_rows(
    fits_path: str,
    *,
    error_class: type[Exception] = ValueError,
) -> Sequence[Mapping[str, Any]]:
    try:
        data = get_hdu(fits_path, extension="CAT").data
    except ClientAlertException as exc:
        raise error_class(f"CAT HDU is missing for {fits_path}.") from exc
    if data is None:
        raise error_class(f"CAT HDU is empty for {fits_path}.")
    names = list(data.names or [])
    return [
        {name: data[name][index].item() if hasattr(data[name][index], "item") else data[name][index] for name in names}
        for index in range(len(data))
    ]

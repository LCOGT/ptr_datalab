import numpy as np
from PIL import Image
from datalab.datalab_session.s3_utils import get_fits
from datalab.datalab_session.file_utils import get_hdu
from fits2image.scaling import extract_samples, calc_zscale_min_max

# TODO: This analysis endpoint assumes the image to be of 16 bitdepth. We should make this agnositc to bit depth in the future

def raw_data(input: dict):
    fits_path = get_fits(input['basename'], input.get('source', 'archive'))

    sci_hdu = get_hdu(fits_path, 'SCI')
    image_data = sci_hdu.data
    
    # Compute the fits2image autoscale params to send with the image
    samples = extract_samples(image_data, sci_hdu.header, 2000)
    median = np.median(samples)
    _, zmax, _ = calc_zscale_min_max(samples, contrast=0.1, iterations=1)

    # resize the image to max. 500 pixels on an axis by default for the UI
    max_size = input.get('max_size', 500)
    image = Image.fromarray(image_data)
    newImage = image.resize((max_size, max_size), Image.LANCZOS)
    bitpix = abs(int(sci_hdu.header.get('BITPIX', 16)))
    match bitpix:
        case 8:
            datatype = np.uint8
        case 16:
            datatype = np.float16
        case 32:
            datatype = np.float32
    scaled_array = np.asarray(newImage).astype(datatype)
    scaled_array_flipped = np.flip(scaled_array, axis=0)

    return {'data': scaled_array_flipped.flatten().tolist(),
            'height': scaled_array.shape[0],
            'width': scaled_array.shape[1],
            'zmin': int(median),
            'zmax': int(zmax),
            'bitdepth': 16
        }

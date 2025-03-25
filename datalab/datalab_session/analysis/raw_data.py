import logging
import numpy as np
import math
from PIL import Image
from datalab.datalab_session.utils.s3_utils import get_fits
from datalab.datalab_session.utils.file_utils import get_hdu
from fits2image.scaling import extract_samples, calc_zscale_min_max

# TODO: This analysis endpoint assumes the image to be of 16 bitdepth. We should make this agnositc to bit depth in the future

def raw_data(input: dict):
    with get_fits(input['basename'], input.get('source', 'archive')) as fits_path:
        sci_hdu = get_hdu(fits_path, 'SCI')
    
    image_data = sci_hdu.data
    
    # Compute the fits2image autoscale params to send with the image
    samples = extract_samples(image_data, sci_hdu.header, 2000)
    median = np.median(samples)
    zmin, zmax, _ = calc_zscale_min_max(samples, contrast=0.1, iterations=1)

    # resize the image to max. 500 pixels on an axis by default for the UI
    max_size = input.get('max_size', 500)
    image = Image.fromarray(image_data)
    bitpix = abs(int(sci_hdu.header.get('BITPIX', 16)))
    max_value = int(sci_hdu.header.get('SATURATE', 0))  # If saturate header is present, use that as max value

    BITPIX_TYPES = {
        8: (np.uint8, np.iinfo(np.uint8).max),
        16: (np.float16, np.finfo(np.float16).max),
        32: (np.float32, np.finfo(np.float32).max)
    }
    
    datatype, max_value_default = BITPIX_TYPES.get(bitpix, (np.uint16, np.iinfo(np.uint16).max))
    max_value = max_value or max_value_default

    image = Image.fromarray(image_data)
    newImage = image.resize((max_size, max_size), Image.LANCZOS)
    scaled_array = np.asarray(newImage).astype(datatype)
    scaled_array_flipped = np.flip(scaled_array, axis=0)

    # Set the zmin/zmax to integer values for calculating bins
    zmin = math.floor(zmin)
    zmax = math.ceil(zmax)

    # Here we do a crazy histogram scaling to stretch the points in between zmin and zmax since that is where most detail is
    # We have 10 bins before zmin, 100 between zmin and zmax and 10 after zmax.
    zero_point = math.floor(min(np.min(samples), 0))  # This is for images whose values go below 0
    lower_bound = int(zmin * 0.8)  # Increase resolution slightly below zmin
    upper_bound = int(zmax*1.2)  # Increase resolution slightly beyond zmax

    def calculate_bins(start, end, num_bins):
        # if start and end are equal, return a single bin
        if start == end:
            return [start]
        # step must be at least 1
        step = max(1, int(abs((end-start) / num_bins)))
        return np.arange(start, end, step).tolist()

    try:
        # Create bins, can fail if trying to create range between two equal numbers
        bins = calculate_bins(zero_point, lower_bound, 10)
        bins += calculate_bins(lower_bound, upper_bound, 100)
        bins += calculate_bins(upper_bound, max_value, 10)
    except Exception as e:
        logging.error(f'error calculating bins: {e}')
        logging.error(f'values: zero_point={zero_point}, lower_bound={lower_bound}, upper_bound={upper_bound}, max_value={max_value}')
        bins = np.linspace(zero_point, max_value, 120).tolist() # Fallback to linear binning
    
    # Calculate histogram
    histogram, bin_edges = np.histogram(samples, bins=bins)
    bin_middles = []
    previous_edge = 0
    for edge in bin_edges:
        if edge != 0:
            bin_middles.append(previous_edge + int((edge-previous_edge) / 2.0))
        previous_edge = edge

    # Using np.log10 on the histogram made some wild results, so just apply log10 to each value
    hist = []
    for h in histogram:
        if h > 0:
            hist.append(math.log10(h))
        else:
            hist.append(0)

    return {'data': scaled_array_flipped.flatten().tolist(),
            'height': scaled_array.shape[0],
            'width': scaled_array.shape[1],
            'histogram': hist,
            'bins': bin_middles,
            'zmin': round(median),
            'zmax': round(zmax),
            'bitdepth': bitpix
        }

import numpy as np
import math
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
    zmin, zmax, _ = calc_zscale_min_max(samples, contrast=0.1, iterations=1)

    # resize the image to max. 500 pixels on an axis by default for the UI
    max_size = input.get('max_size', 500)
    image = Image.fromarray(image_data)
    newImage = image.resize((max_size, max_size), Image.LANCZOS)
    bitpix = abs(int(sci_hdu.header.get('BITPIX', 16)))
    max_value = int(sci_hdu.header.get('SATURATE', 0))  # If saturate header is present, use that as max value
    match bitpix:
        case 8:
            datatype = np.uint8
            if not max_value:
                max_value = np.iinfo(datatype).max
        case 16:
            datatype = np.float16
            if not max_value:
                max_value = np.finfo(datatype).max
        case 32:
            datatype = np.float32
            if not max_value:
                max_value = np.finfo(datatype).max
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
    lower_step = int(abs(lower_bound / 10))
    upper_step = int(abs((max_value - upper_bound) / 10))
    step = int(abs((upper_bound - lower_bound) / 100))
    bins = np.arange(zero_point, lower_bound, lower_step).tolist()
    bins += np.arange(lower_bound, upper_bound, step).tolist()
    bins += np.arange(upper_bound, max_value, upper_step).tolist()
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

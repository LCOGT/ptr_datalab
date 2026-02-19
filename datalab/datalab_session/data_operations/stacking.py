import logging

import numpy as np
from django.contrib.auth.models import User

from datalab.datalab_session.data_operations.input_data_handler import InputDataHandler
from datalab.datalab_session.data_operations.fits_output_handler import FITSOutputHandler
from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.utils.format import Format
from datalab.datalab_session.utils.file_utils import crop_arrays
from reproject import reproject_adaptive
from astropy.io import fits
from reproject.mosaicking import find_optimal_celestial_wcs
from astropy.wcs import WCS

log = logging.getLogger()
log.setLevel(logging.INFO)


class Stack(BaseDataOperation):
    MINIMUM_NUMBER_OF_INPUTS = 2
    MAXIMUM_NUMBER_OF_INPUTS = 999
    PROGRESS_STEPS = {
        'STACKING_MIDPOINT': 0.5,
        'STACKING_PERCENTAGE_COMPLETION': 0.6,
        'STACKING_OUTPUT_PERCENTAGE_COMPLETION': 0.8,
        'OUTPUT_PERCENTAGE_COMPLETION': 1.0
    }
    @staticmethod
    def name():
        return 'Stacking'
    
    @staticmethod
    def description():
        return """The stacking operation takes in 2..n input images and adds the values pixel-by-pixel.

The output is a stacked image for the n input images. This operation is commonly used for improving signal to noise."""

    @staticmethod
    def wizard_description():
        description = {
            'name': Stack.name(),
            'description': Stack.description(),
            'category': 'image',
            'inputs': {
                'input_files': {
                    'name': 'Input Files',
                    'description': 'The input files to operate on',
                    'type': Format.FITS,
                    'minimum': Stack.MINIMUM_NUMBER_OF_INPUTS,
                    'maximum': Stack.MAXIMUM_NUMBER_OF_INPUTS,
                },
                'stacking_mode': {
                    'name': 'Stacking Mode',
                    'description': 'Choose simple stacking or reprojection before stacking',
                    'type': 'select',
                    'options': ['simple', 'reproject'],
                    'default': 'simple'
                }
            }
        }
        return description
    
    def find_optimal_reference(self, images):
        """
        images: list of InputDataHandler
        returns: optimized_wcs, optimized_shape
        """

        image_hdus = [img.sci_hdu for img in images]

        wcs_opt, shape_out = find_optimal_celestial_wcs(image_hdus)
        return wcs_opt, shape_out

    def crop_bbox_from_footprint(self, footprint: np.ndarray):
        """
        Return a bbox (r0,r1,c0,c1) such that the bbox edges contain no invalid pixels.
        footprint: 2D array where non-zero means covered.
        """
        mask = footprint != 0
        if not mask.any():
            raise ValueError("Empty footprint")

        # start with the loose envelope (any valid somewhere)
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        r0, r1 = rows[0], rows[-1] + 1
        c0, c1 = cols[0], cols[-1] + 1

        # now shrink until the *edges* are fully valid
        changed = True
        while changed:
            changed = False

            # top edge
            if r0 < r1 and not mask[r0, c0:c1].all():
                r0 += 1
                changed = True

            # bottom edge
            if r0 < r1 and not mask[r1 - 1, c0:c1].all():
                r1 -= 1
                changed = True

            # left edge
            if c0 < c1 and not mask[r0:r1, c0].all():
                c0 += 1
                changed = True

            # right edge
            if c0 < c1 and not mask[r0:r1, c1 - 1].all():
                c1 -= 1
                changed = True

            if r0 >= r1 or c0 >= c1:
                raise ValueError("No rectangular region without invalid pixels on the edges")

            if not changed:
                break

        log.info(f'crop bbox from footprint (shrunk): {r0}, {r1}, {c0}, {c1}')
        return r0, r1, c0, c1
    
    def intersect_bboxes(self, bboxes):
        """
        bboxes: iterable of (r0,r1,c0,c1). Returns intersection bbox.
        """
        r0 = max(b[0] for b in bboxes)
        r1 = min(b[1] for b in bboxes)
        c0 = max(b[2] for b in bboxes)
        c1 = min(b[3] for b in bboxes)
        if r0 >= r1 or c0 >= c1:
            raise ValueError("No overlapping valid region across images")
        log.info(f'intersection bbox: {r0}, {r1}, {c0}, {c1}')
        return r0, r1, c0, c1
    
    def crop(self, img, bbox):
        r0, r1, c0, c1 = bbox
        return np.ascontiguousarray(img[r0:r1, c0:c1])

    def prepare_for_sum(self, images, footprints):
        """
        images: list of 2D float arrays (len=3)
        footprints: list of 2D {0,1} arrays (len=3)
        Returns: cropped_images, common_bbox
        """


        per_bbox = [self.crop_bbox_from_footprint(fp) for fp in footprints]
        common_bbox = self.intersect_bboxes(per_bbox)
        cropped = [self.crop(im, common_bbox) for im in images]

        log.info(f'cropped: {cropped[0].shape}, common_bbox: {common_bbox}')
        return cropped, common_bbox

    def reproject_images_to_reference(self, input_fits, optimized_wcs, optimized_shape):
        """
        input_fits: list of InputDataHandler
        optimized_wcs: WCS object for the optimal reference frame
        optimized_shape: (ny, nx) shape 

        returns:
            reprojected_arrays: list of 2D float arrays reprojected to the optimal reference frame
            footprints: list of 2D  numpy arrays indicating valid data regions in
        """
        reprojected_arrays = []
        footprints = []
        for img in input_fits:
            array, footprint = reproject_adaptive(
                img.sci_hdu,
                optimized_wcs,
                shape_out=optimized_shape,
                return_footprint=True,
                conserve_flux=True
            )
            reprojected_arrays.append(array)
            footprints.append(footprint)

        return reprojected_arrays, footprints

    def operate(self, submitter: User):
        stacking_mode = self.input_data.get("stacking_mode")
        input_files = self._validate_inputs(
            input_key='input_files',
            minimum_inputs=self.MINIMUM_NUMBER_OF_INPUTS
        )
        comment= f'Datalab Stacking on {", ".join([image["basename"] for image in input_files])}'
        log.info(comment)

        input_fits_list = []
        for index, input in enumerate(input_files, start=1):
            input_fits_list.append(InputDataHandler(submitter, input['basename'], input['source']))
            log.info(f'input fits list in normalization: {input_fits_list}')
            self.set_operation_progress(Stack.PROGRESS_STEPS['STACKING_MIDPOINT'] * (index / len(input_files)))

        if stacking_mode == "reproject":
            optimized_wcs, optimized_shape = self.find_optimal_reference(input_fits_list)
            arrays, footprints = self.reproject_images_to_reference(input_fits_list, optimized_wcs, optimized_shape)
            cropped_data, _ = self.prepare_for_sum(arrays, footprints)

            optimized_header = optimized_wcs.to_header()
            header = input_fits_list[0].sci_hdu.header.copy()
            header.update(optimized_header)

        else:
            arrays = [image.sci_data for image in input_fits_list]
            cropped_data, _ = crop_arrays(arrays)
            header = input_fits_list[0].sci_hdu.header.copy()

        self.set_operation_progress(Stack.PROGRESS_STEPS['STACKING_PERCENTAGE_COMPLETION'])
        
        stacked_sum = np.nansum(np.stack(cropped_data), axis=0)

        self.set_operation_progress(Stack.PROGRESS_STEPS['STACKING_OUTPUT_PERCENTAGE_COMPLETION'])

        output = FITSOutputHandler(self.cache_key, stacked_sum, self.temp, comment, data_header=header).create_and_save_data_products(Format.FITS)
        log.info(f'Stacked output: {output}')

        self.set_output(output)
        self.set_operation_progress(Stack.PROGRESS_STEPS['OUTPUT_PERCENTAGE_COMPLETION'])
        self.set_status('COMPLETED')

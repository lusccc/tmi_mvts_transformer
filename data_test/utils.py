from warnings import warn
import numpy as np
import scipy.ndimage as ndi
from skimage.feature.peak import _get_excluded_border_width, _get_threshold, _get_peak_mask, _exclude_border, \
    _get_high_intensity_peaks


def peak_local_max(image, min_distance=1, threshold_abs=None,
                   threshold_rel=None, exclude_border=True, indices=True,
                   num_peaks=np.inf, footprint=None, labels=None,
                   num_peaks_per_label=np.inf, p_norm=np.inf):
    """Find peaks in an image as coordinate list or boolean mask.

    Peaks are the local maxima in a region of `2 * min_distance + 1`
    (i.e. peaks are separated by at least `min_distance`).

    If both `threshold_abs` and `threshold_rel` are provided, the maximum
    of the two is chosen as the minimum intensity threshold of peaks.

    .. versionchanged:: 0.18
        Prior to version 0.18, peaks of the same height within a radius of
        `min_distance` were all returned, but this could cause unexpected
        behaviour. From 0.18 onwards, an arbitrary peak within the region is
        returned. See issue gh-2592.

    Parameters
    ----------
    image : ndarray
        Input image.
    min_distance : int, optional
        The minimal allowed distance separating peaks. To find the
        maximum number of peaks, use `min_distance=1`.
    threshold_abs : float or None, optional
        Minimum intensity of peaks. By default, the absolute threshold is
        the minimum intensity of the image.
    threshold_rel : float or None, optional
        Minimum intensity of peaks, calculated as
        ``max(image) * threshold_rel``.
    exclude_border : int, tuple of ints, or bool, optional
        If positive integer, `exclude_border` excludes peaks from within
        `exclude_border`-pixels of the border of the image.
        If tuple of non-negative ints, the length of the tuple must match the
        input array's dimensionality.  Each element of the tuple will exclude
        peaks from within `exclude_border`-pixels of the border of the image
        along that dimension.
        If True, takes the `min_distance` parameter as value.
        If zero or False, peaks are identified regardless of their distance
        from the border.
    indices : bool, optional
        If True, the output will be an array representing peak
        coordinates. The coordinates are sorted according to peaks
        values (Larger first). If False, the output will be a boolean
        array shaped as `image.shape` with peaks present at True
        elements. ``indices`` is deprecated and will be removed in
        version 0.20. Default behavior will be to always return peak
        coordinates. You can obtain a mask as shown in the example
        below.
    num_peaks : int, optional
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` peaks based on highest peak intensity.
    footprint : ndarray of bools, optional
        If provided, `footprint == 1` represents the local region within which
        to search for peaks at every point in `image`.
    labels : ndarray of ints, optional
        If provided, each unique region `labels == value` represents a unique
        region to search for peaks. Zero is reserved for background.
    num_peaks_per_label : int, optional
        Maximum number of peaks for each label.
    p_norm : float
        Which Minkowski p-norm to use. Should be in the range [1, inf].
        A finite large p may cause a ValueError if overflow can occur.
        ``inf`` corresponds to the Chebyshev distance and 2 to the
        Euclidean distance.

    Returns
    -------
    output : ndarray or ndarray of bools

        * If `indices = True`  : (row, column, ...) coordinates of peaks.
        * If `indices = False` : Boolean array shaped like `image`, with peaks
          represented by True values.

    Notes
    -----
    The peak local maximum function returns the coordinates of local peaks
    (maxima) in an image. Internally, a maximum filter is used for finding local
    maxima. This operation dilates the original image. After comparison of the
    dilated and original image, this function returns the coordinates or a mask
    of the peaks where the dilated image equals the original image.

    See also
    --------
    skimage.feature.corner_peaks

    Examples
    --------
    >>> img1 = np.zeros((7, 7))
    >>> img1[3, 4] = 1
    >>> img1[3, 2] = 1.5
    >>> img1
    array([[0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 1.5, 0. , 1. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ]])

    >>> peak_local_max(img1, min_distance=1)
    array([[3, 2],
           [3, 4]])

    >>> peak_local_max(img1, min_distance=2)
    array([[3, 2]])

    >>> img2 = np.zeros((20, 20, 20))
    >>> img2[10, 10, 10] = 1
    >>> img2[15, 15, 15] = 1
    >>> peak_idx = peak_local_max(img2, exclude_border=0)
    >>> peak_idx
    array([[10, 10, 10],
           [15, 15, 15]])

    >>> peak_mask = np.zeros_like(img2, dtype=bool)
    >>> peak_mask[tuple(peak_idx.T)] = True
    >>> np.argwhere(peak_mask)
    array([[10, 10, 10],
           [15, 15, 15]])

    """
    if (footprint is None or footprint.size == 1) and min_distance < 1:
        warn("When min_distance < 1, peak_local_max acts as finding "
             "image > max(threshold_abs, threshold_rel * max(image)).",
             RuntimeWarning, stacklevel=2)

    border_width = _get_excluded_border_width(image, min_distance,
                                              exclude_border)

    threshold = _get_threshold(image, threshold_abs, threshold_rel)

    if footprint is None:
        size = 2 * min_distance + 1
        footprint = np.ones((size, ) * image.ndim, dtype=bool)
    else:
        footprint = np.asarray(footprint)

    if labels is None:
        # Non maximum filter
        mask = _get_peak_mask(image, footprint, threshold)

        mask = _exclude_border(mask, border_width)

        # Select highest intensities (num_peaks)
        coordinates = _get_high_intensity_peaks(image, mask,
                                                num_peaks,
                                                min_distance, p_norm)

    else:
        _labels = _exclude_border(labels.astype(int, casting="safe"),
                                  border_width)

        if np.issubdtype(image.dtype, np.floating):
            bg_val = np.finfo(image.dtype).min
        else:
            bg_val = np.iinfo(image.dtype).min

        # For each label, extract a smaller image enclosing the object of
        # interest, identify num_peaks_per_label peaks
        labels_peak_coord = []

        for label_idx, roi in enumerate(ndi.find_objects(_labels)):

            if roi is None:
                continue

            # Get roi mask
            label_mask = labels[roi] == label_idx + 1
            # Extract image roi
            img_object = image[roi].copy()
            # Ensure masked values don't affect roi's local peaks
            img_object[np.logical_not(label_mask)] = bg_val

            mask = _get_peak_mask(img_object, footprint, threshold, label_mask)

            coordinates = _get_high_intensity_peaks(img_object, mask,
                                                    num_peaks_per_label,
                                                    min_distance,
                                                    p_norm)

            # transform coordinates in global image indices space
            for idx, s in enumerate(roi):
                coordinates[:, idx] += s.start

            labels_peak_coord.append(coordinates)

        if labels_peak_coord:
            coordinates = np.vstack(labels_peak_coord)
        else:
            coordinates = np.empty((0, 2), dtype=int)

        if len(coordinates) > num_peaks:
            out = np.zeros_like(image, dtype=bool)
            out[tuple(coordinates.T)] = True
            coordinates = _get_high_intensity_peaks(image, out,
                                                    num_peaks,
                                                    min_distance,
                                                    p_norm)

    if indices:
        return coordinates
    else:
        out = np.zeros_like(image, dtype=bool)
        out[tuple(coordinates.T)] = True
        return out
"""
refpix_numba.py

High-performance, pure-Python Numba implementation of the Reference Pixel
Filter algorithm (originally refpixelfilter.f90).

Replaces legacy Fortran 90 / f2py extension with a memory-safe, multi-threaded
JIT-compiled implementation that runs bit-for-bit identical results on 64-bit platforms.
"""

import numpy as np
from numba import njit, prange


@njit(fastmath=False)
def _bounds(histogram, n, lq, uq):
    """
    Compute quantile bounds from a 1-based histogram matching Fortran bounds().
    histogram is 1-indexed (indices 1 to n).
    """
    cumsum = 0
    for i in range(1, n + 1):
        cumsum += histogram[i]
    lv = int(lq * cumsum)
    uv = int(uq * cumsum)
    uvfound = 0
    b0 = 0
    b1 = 0
    for i in range(n, 0, -1):
        if cumsum < uv and uvfound == 0:
            b1 = i
            uvfound = 1
        if cumsum < lv:
            b0 = i
            break
        cumsum -= histogram[i]
    return b0, b1


@njit(fastmath=False)
def _calcmean(drysum, dryhist, lowerb, upperb):
    """
    Calculate mean groundcover within quantile bounds matching Fortran calcmean().
    Array lookups match Fortran dryhist(i+1) and drysum(i+1).
    """
    nsum = 0
    bigsum = 0
    for i in range(lowerb, upperb + 1):
        if 0 <= i + 1 < len(dryhist):
            nsum += dryhist[i + 1]
            bigsum += drysum[i + 1]
    if nsum == 0:
        return 0
    else:
        return bigsum // nsum


@njit(fastmath=False)
def _refpix_row(cover, ground, out, out_r, r, window, lowerq, upperq, minpercentv_f32, nrow, ncol):
    """
    Process a single row using a 1D sliding 2D window histogram.
    """
    halfwindow = window // 2

    # Fixed buffers of size 103 for Fortran 1-based indexing [1..101] and i+1 lookups
    percentilehist = np.zeros(103, dtype=np.int32)
    drysum = np.zeros(103, dtype=np.int32)
    dryhist = np.zeros(103, dtype=np.int32)

    topindex_f = max(1, (r + 1) - halfwindow)
    bottomindex_f = min(nrow, (r + 1) + halfwindow)
    r_top = topindex_f - 1
    r_bottom = bottomindex_f - 1

    # Left-hand edge initialization (cols 0 to halfwindow - 1)
    for wi in range(r_top, r_bottom + 1):
        for wj in range(0, halfwindow):
            if wj < ncol:
                pixval = cover[wi, wj]
                dryval = ground[wi, wj]
                if pixval > 0:
                    percentilehist[pixval] += 1
                    if dryval > 0:
                        dryhist[pixval] += 1
                        drysum[pixval] += dryval

    # March across columns
    for c in range(ncol):
        j_f = c + 1
        rightindex_f = j_f + halfwindow
        right_c = rightindex_f - 1

        # Add right-hand sliver
        if rightindex_f <= ncol:
            for k in range(r_top, r_bottom + 1):
                pixval = cover[k, right_c]
                dryval = ground[k, right_c]
                if pixval > 0:
                    percentilehist[pixval] += 1
                    if dryval > 0:
                        dryhist[pixval] += 1
                        drysum[pixval] += dryval

        # Approximate window pixel count matching Fortran formula
        if j_f < halfwindow:
            npixels = (bottomindex_f - topindex_f) * rightindex_f
        elif rightindex_f > ncol:
            npixels = (bottomindex_f - topindex_f) * (ncol - j_f)
        else:
            npixels = (bottomindex_f - topindex_f) * window

        validpixels = 0
        for z in range(1, 102):
            validpixels += dryhist[z]

        # Valid fraction check
        if validpixels >= minpercentv_f32 * npixels:
            b0, b1 = _bounds(percentilehist, 101, lowerq, upperq)
            out[out_r, c] = _calcmean(drysum, dryhist, b0, b1)
        else:
            out[out_r, c] = -1

        # Subtract left sliver
        if j_f > halfwindow:
            left_c = (j_f - halfwindow) - 1
            for k in range(r_top, r_bottom + 1):
                pixval = cover[k, left_c]
                dryval = ground[k, left_c]
                if pixval > 0:
                    percentilehist[pixval] -= 1
                    if dryval > 0:
                        drysum[pixval] -= dryval
                        dryhist[pixval] -= 1


@njit(parallel=True, fastmath=False)
def refpixelfilternodata(cover, ground, window=1501, lowerquantile=0.90, upperquantile=0.95, minvalpercent=0.15, r_start=0, r_stop=-1):
    """
    Parallel reference pixel filter replacing Fortran refpixelfilter.refpix.refpixelfilternodata.

    Parameters
    ----------
    cover : np.ndarray (int8, 2D)
        Masked percentile groundcover statistics (values 1..101, 0 is nodata).
    ground : np.ndarray (int8, 2D)
        Seasonal groundcover array (values 1..101, 0 is nodata).
    window : int
        Odd comparison window size (default 1501).
    lowerquantile : float
        Lower quantile bound for finding high persistent cover reference sites (default 0.90).
    upperquantile : float
        Upper quantile bound (default 0.95).
    minvalpercent : float
        Minimum fraction of valid pixels within comparison window to return a result (default 0.15).
    r_start : int
        Start row index in `cover` / `ground` to compute output for (default 0).
    r_stop : int
        Stop row index in `cover` / `ground` to compute output for (default -1 = nrow).

    Returns
    -------
    out : np.ndarray (int8, 2D)
        Reference pixel value array of shape (r_stop - r_start, ncol).
    """
    nrow, ncol = cover.shape
    if r_stop < 0 or r_stop > nrow:
        r_stop = nrow

    n_out = r_stop - r_start
    out = np.empty((n_out, ncol), dtype=np.int8)
    minpercentv_f32 = np.float32(minvalpercent)

    for i in prange(n_out):
        r = r_start + i
        _refpix_row(cover, ground, out, i, r, window, lowerquantile, upperquantile, minpercentv_f32, nrow, ncol)

    return out

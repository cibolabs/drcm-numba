"""
refpix_numba_ibra.py

IBRA-Constrained High-Performance Numba Implementation of Reference Pixel Filter.

Constrains the comparison window so each pixel is compared exclusively against
reference statistics and seasonal groundcover from the same matching IBRA
biogeographic region (IBRA 7.0 regions 1..89).
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
def _refpix_ibra_row(cover, ground, ibra, out, out_r, r, window, lowerq, upperq, minpercentv_f32, nrow, ncol):
    """
    Process a single row using a 1D sliding 2D window histogram partitioned by IBRA region.
    Supports up to 90 regions (0 = NoData/Ocean, 1..89 = IBRA regions).
    """
    halfwindow = window // 2

    # Fixed per-region histogram tables: 90 regions x 103 bins.
    # Total memory footprint per row is ~112 KB (3 * 90 * 103 * 4B + counts),
    # which fits entirely within the CPU core's L2 cache (typically 512 KB - 1 MB),
    # guaranteeing high cache hit rates and zero thread memory contention.
    percentilehist = np.zeros((90, 103), dtype=np.int32)
    drysum = np.zeros((90, 103), dtype=np.int32)
    dryhist = np.zeros((90, 103), dtype=np.int32)
    valid_counts = np.zeros(90, dtype=np.int32)
    total_counts = np.zeros(90, dtype=np.int32)

    topindex_f = max(1, (r + 1) - halfwindow)
    bottomindex_f = min(nrow, (r + 1) + halfwindow)
    r_top = topindex_f - 1
    r_bottom = bottomindex_f - 1

    # Left-hand edge initialization (cols 0 to halfwindow - 1)
    for wi in range(r_top, r_bottom + 1):
        for wj in range(0, halfwindow):
            if wj < ncol:
                reg = ibra[wi, wj]
                if reg > 0:
                    total_counts[reg] += 1
                    pixval = cover[wi, wj]
                    dryval = ground[wi, wj]
                    if pixval > 0:
                        percentilehist[reg, pixval] += 1
                        if dryval > 0:
                            dryhist[reg, pixval] += 1
                            drysum[reg, pixval] += dryval
                            valid_counts[reg] += 1

    # March across columns
    for c in range(ncol):
        j_f = c + 1
        rightindex_f = j_f + halfwindow
        right_c = rightindex_f - 1

        # Add right-hand sliver
        if rightindex_f <= ncol:
            for k in range(r_top, r_bottom + 1):
                reg = ibra[k, right_c]
                if reg > 0:
                    total_counts[reg] += 1
                    pixval = cover[k, right_c]
                    dryval = ground[k, right_c]
                    if pixval > 0:
                        percentilehist[reg, pixval] += 1
                        if dryval > 0:
                            dryhist[reg, pixval] += 1
                            drysum[reg, pixval] += dryval
                            valid_counts[reg] += 1

        # Evaluate target pixel
        target_reg = ibra[r, c]
        if target_reg == 0 or total_counts[target_reg] == 0:
            out[out_r, c] = -1
        else:
            n_reg_pixels = total_counts[target_reg]
            v_reg_pixels = valid_counts[target_reg]

            if v_reg_pixels >= minpercentv_f32 * n_reg_pixels and v_reg_pixels > 0:
                b0, b1 = _bounds(percentilehist[target_reg], 101, lowerq, upperq)
                out[out_r, c] = _calcmean(drysum[target_reg], dryhist[target_reg], b0, b1)
            else:
                out[out_r, c] = -1

        # Subtract left sliver
        if j_f > halfwindow:
            left_c = (j_f - halfwindow) - 1
            for k in range(r_top, r_bottom + 1):
                reg = ibra[k, left_c]
                if reg > 0:
                    total_counts[reg] -= 1
                    pixval = cover[k, left_c]
                    dryval = ground[k, left_c]
                    if pixval > 0:
                        percentilehist[reg, pixval] -= 1
                        if dryval > 0:
                            drysum[reg, pixval] -= dryval
                            dryhist[reg, pixval] -= 1
                            valid_counts[reg] -= 1


@njit(parallel=True, fastmath=False)
def refpixelfilter_ibra(
    cover,
    ground,
    ibra,
    window=1501,
    lowerquantile=0.90,
    upperquantile=0.95,
    minvalpercent=0.15,
    r_start=0,
    r_stop=-1
):
    """
    Parallel IBRA-constrained reference pixel filter.

    Parameters
    ----------
    cover : np.ndarray (int8, 2D)
        Masked percentile groundcover statistics (values 1..101, 0 is nodata).
    ground : np.ndarray (int8, 2D)
        Seasonal groundcover array (values 1..101, 0 is nodata).
    ibra : np.ndarray (uint8, 2D)
        Rasterized IBRA region IDs (values 1..89, 0 is nodata/ocean).
    window : int
        Odd comparison window size (default 1501).
    lowerquantile : float
        Lower quantile bound (default 0.90).
    upperquantile : float
        Upper quantile bound (default 0.95).
    minvalpercent : float
        Minimum fraction of valid pixels within comparison window in matching IBRA (default 0.15).
    r_start : int
        Start row index in input arrays to compute output for (default 0).
    r_stop : int
        Stop row index in input arrays to compute output for (default -1 = nrow).

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
        _refpix_ibra_row(
            cover, ground, ibra, out, i, r,
            window, lowerquantile, upperquantile, minpercentv_f32,
            nrow, ncol
        )

    return out

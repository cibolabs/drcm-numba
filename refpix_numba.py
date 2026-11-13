"""
Numba-accelerated reference pixel filter without IBRA stratification.

Uses a cache-tuned 2D tiled incremental column-histogram sliding window with
L2-friendly uint16 column histograms, empty-block ocean skipping, aligned vector
updates, and optional column slicing (c_start/c_stop) for halo-padded AOI runs.
"""

import numpy as np
from numba import njit, prange


@njit(fastmath=False)
def _bounds_fast(histogram, total_cnt, n, lq, uq):
    """Compute quantile bin bounds using pre-tracked histogram total."""
    cumsum = total_cnt
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
    """Calculate integer mean groundcover within quantile bounds."""
    nsum = 0
    bigsum = 0
    for i in range(lowerb, upperb + 1):
        if 0 <= i + 1 < len(dryhist):
            nsum += dryhist[i + 1]
            bigsum += drysum[i + 1]
    if nsum == 0:
        return 0
    return bigsum // nsum


@njit(parallel=True, fastmath=False)
def _build_valid_blocks(cover, ground, bsize):
    """Precompute coarse block presence mask where both cover and ground are valid."""
    nrow, ncol = cover.shape
    nbr = (nrow + bsize - 1) // bsize
    nbc = (ncol + bsize - 1) // bsize
    blk = np.zeros((nbr, nbc), dtype=np.uint8)
    for br in prange(nbr):
        r0 = br * bsize
        r1 = min(nrow, r0 + bsize)
        for bc in range(nbc):
            c0 = bc * bsize
            c1 = min(ncol, c0 + bsize)
            found = np.uint8(0)
            for r in range(r0, r1):
                for c in range(c0, c1):
                    pv = cover[r, c]
                    if 0 < pv <= 101 and ground[r, c] > 0:
                        found = np.uint8(1)
                        break
                if found != 0:
                    break
            blk[br, bc] = found
    return blk


@njit(fastmath=False)
def _tile_has_valid(blk, r0, r1, c0, c1, halfwindow, nrow, ncol, bsize):
    """Check whether any block intersecting a tile's halo contains valid paired pixels."""
    wr0 = max(0, r0 - halfwindow)
    wr1 = min(nrow, r1 + halfwindow)
    wc0 = max(0, c0 - halfwindow)
    wc1 = min(ncol, c1 + halfwindow)
    br0 = wr0 // bsize
    br1 = (wr1 - 1) // bsize
    bc0 = wc0 // bsize
    bc1 = (wc1 - 1) // bsize
    for br in range(br0, br1 + 1):
        for bc in range(bc0, bc1 + 1):
            if blk[br, bc] != 0:
                return True
    return False


@njit(fastmath=False)
def _fill_empty_tile(out, r0, r1, c0, c1, r_start, c_start, halfwindow, nrow, ncol):
    """Fill an all-empty tile preserving boundary edge zero-npixel behavior."""
    for r in range(r0, r1):
        topindex_f = max(1, (r + 1) - halfwindow)
        bottomindex_f = min(nrow, (r + 1) + halfwindow)
        row_empty_deg = bottomindex_f <= topindex_f
        orow = r - r_start
        for c in range(c0, c1):
            if row_empty_deg or c == ncol - 1:
                out[orow, c - c_start] = 0
            else:
                out[orow, c - c_start] = -1


def _make_kernel(col_hist_dtype):
    """Compile a tiled sliding-window Numba kernel for the requested column histogram dtype."""
    one = col_hist_dtype(1)

    @njit(fastmath=False)
    def _process_tile(
        cover,
        ground,
        out,
        r0,
        r1,
        c0,
        c1,
        r_start,
        c_start,
        window,
        lowerq,
        upperq,
        minpercentv_f32,
        nrow,
        ncol,
    ):
        halfwindow = window // 2
        w_c0 = max(0, c0 - halfwindow)
        w_c1 = min(ncol, c1 + halfwindow)
        tile_cols = w_c1 - w_c0

        # Pad inner dimension to 104 elements (bins 1..101 active; 0/102/103 remain zero)
        # for 16-byte SIMD alignment and L2 cache locality.
        col_perc = np.zeros((tile_cols, 104), dtype=col_hist_dtype)
        col_dryhist = np.zeros((tile_cols, 104), dtype=col_hist_dtype)
        col_drysum = np.zeros((tile_cols, 104), dtype=np.int32)
        col_perc_tot = np.zeros(tile_cols, dtype=np.int32)
        col_valid_tot = np.zeros(tile_cols, dtype=np.int32)

        top_init = max(0, r0 - halfwindow)
        bot_init = min(nrow - 1, r0 + halfwindow)
        for rr in range(top_init, bot_init + 1):
            for cc in range(w_c0, w_c1):
                pv = cover[rr, cc]
                if 0 < pv <= 101:
                    tc = cc - w_c0
                    col_perc[tc, pv] += one
                    col_perc_tot[tc] += 1
                    dv = ground[rr, cc]
                    if dv > 0:
                        col_dryhist[tc, pv] += one
                        col_drysum[tc, pv] += dv
                        col_valid_tot[tc] += 1

        percentilehist = np.zeros(104, dtype=np.int32)
        dryhist = np.zeros(104, dtype=np.int32)
        drysum = np.zeros(104, dtype=np.int32)

        for r in range(r0, r1):
            if r > r0:
                rem_r = r - 1 - halfwindow
                if rem_r >= 0:
                    for cc in range(w_c0, w_c1):
                        pv = cover[rem_r, cc]
                        if 0 < pv <= 101:
                            tc = cc - w_c0
                            col_perc[tc, pv] -= one
                            col_perc_tot[tc] -= 1
                            dv = ground[rem_r, cc]
                            if dv > 0:
                                col_dryhist[tc, pv] -= one
                                col_drysum[tc, pv] -= dv
                                col_valid_tot[tc] -= 1
                add_r = r + halfwindow
                if add_r < nrow:
                    for cc in range(w_c0, w_c1):
                        pv = cover[add_r, cc]
                        if 0 < pv <= 101:
                            tc = cc - w_c0
                            col_perc[tc, pv] += one
                            col_perc_tot[tc] += 1
                            dv = ground[add_r, cc]
                            if dv > 0:
                                col_dryhist[tc, pv] += one
                                col_drysum[tc, pv] += dv
                                col_valid_tot[tc] += 1

            for b in range(104):
                percentilehist[b] = 0
                dryhist[b] = 0
                drysum[b] = 0
            perc_tot = 0
            validpixels = 0

            pre_end = min(ncol, c0 + halfwindow)
            for cc in range(w_c0, pre_end):
                tc = cc - w_c0
                pt = col_perc_tot[tc]
                if pt > 0:
                    perc_tot += pt
                    vt = col_valid_tot[tc]
                    if vt > 0:
                        validpixels += vt
                        for b in range(104):
                            percentilehist[b] += np.int32(col_perc[tc, b])
                            dryhist[b] += np.int32(col_dryhist[tc, b])
                            drysum[b] += col_drysum[tc, b]
                    else:
                        for b in range(104):
                            percentilehist[b] += np.int32(col_perc[tc, b])

            topindex_f = max(1, (r + 1) - halfwindow)
            bottomindex_f = min(nrow, (r + 1) + halfwindow)
            out_r = r - r_start

            for c in range(c0, c1):
                j_f = c + 1
                rightindex_f = j_f + halfwindow
                right_c = rightindex_f - 1
                if right_c < ncol:
                    tc = right_c - w_c0
                    pt = col_perc_tot[tc]
                    if pt > 0:
                        perc_tot += pt
                        vt = col_valid_tot[tc]
                        if vt > 0:
                            validpixels += vt
                            for b in range(104):
                                percentilehist[b] += np.int32(col_perc[tc, b])
                                dryhist[b] += np.int32(col_dryhist[tc, b])
                                drysum[b] += col_drysum[tc, b]
                        else:
                            for b in range(104):
                                percentilehist[b] += np.int32(col_perc[tc, b])

                if j_f < halfwindow:
                    npixels = (bottomindex_f - topindex_f) * rightindex_f
                elif rightindex_f > ncol:
                    npixels = (bottomindex_f - topindex_f) * (ncol - j_f)
                else:
                    npixels = (bottomindex_f - topindex_f) * window

                if validpixels >= minpercentv_f32 * npixels:
                    b0, b1 = _bounds_fast(percentilehist, perc_tot, 101, lowerq, upperq)
                    out[out_r, c - c_start] = _calcmean(drysum, dryhist, b0, b1)
                else:
                    out[out_r, c - c_start] = -1

                if j_f > halfwindow:
                    left_c = (j_f - halfwindow) - 1
                    tc = left_c - w_c0
                    pt = col_perc_tot[tc]
                    if pt > 0:
                        perc_tot -= pt
                        vt = col_valid_tot[tc]
                        if vt > 0:
                            validpixels -= vt
                            for b in range(104):
                                percentilehist[b] -= np.int32(col_perc[tc, b])
                                dryhist[b] -= np.int32(col_dryhist[tc, b])
                                drysum[b] -= col_drysum[tc, b]
                        else:
                            for b in range(104):
                                percentilehist[b] -= np.int32(col_perc[tc, b])

    @njit(parallel=True, fastmath=False)
    def _run_tiled(
        cover,
        ground,
        window,
        lowerquantile,
        upperquantile,
        minvalpercent,
        r_start,
        r_stop,
        c_start,
        c_stop,
        strip_h,
        col_tile_w,
    ):
        nrow, ncol = cover.shape
        n_out_r = max(0, r_stop - r_start)
        n_out_c = max(0, c_stop - c_start)
        out = np.empty((n_out_r, n_out_c), dtype=np.int8)
        if n_out_r == 0 or n_out_c == 0:
            return out

        minpercentv_f32 = np.float32(minvalpercent)
        halfwindow = window // 2
        bsize = 256
        use_blk = minpercentv_f32 > np.float32(0.0) and halfwindow >= 1 and nrow > 1
        if use_blk:
            blk = _build_valid_blocks(cover, ground, bsize)
        else:
            blk = np.ones((1, 1), dtype=np.uint8)

        n_row_strips = (n_out_r + strip_h - 1) // strip_h
        n_col_tiles = (n_out_c + col_tile_w - 1) // col_tile_w
        n_tasks = n_row_strips * n_col_tiles

        for t in prange(n_tasks):
            rs = t // n_col_tiles
            ct = t % n_col_tiles
            r0 = r_start + rs * strip_h
            r1 = min(r_stop, r0 + strip_h)
            c0 = c_start + ct * col_tile_w
            c1 = min(c_stop, c0 + col_tile_w)
            if use_blk and not _tile_has_valid(blk, r0, r1, c0, c1, halfwindow, nrow, ncol, bsize):
                _fill_empty_tile(out, r0, r1, c0, c1, r_start, c_start, halfwindow, nrow, ncol)
                continue
            _process_tile(
                cover,
                ground,
                out,
                r0,
                r1,
                c0,
                c1,
                r_start,
                c_start,
                window,
                lowerquantile,
                upperquantile,
                minpercentv_f32,
                nrow,
                ncol,
            )
        return out

    return _run_tiled


_KERNEL_U16 = _make_kernel(np.uint16)
_KERNEL_I32 = _make_kernel(np.int32)


def refpixelfilternodata(
    cover,
    ground,
    window=1501,
    lowerquantile=0.90,
    upperquantile=0.95,
    minvalpercent=0.15,
    r_start=0,
    r_stop=-1,
    c_start=0,
    c_stop=-1,
    strip_h=256,
    col_tile_w=512,
):
    """
    Numba-accelerated focal reference pixel filter over [r_start:r_stop, c_start:c_stop].

    Bit-for-bit compatible with the scalar sliding-window implementation while using
    L2 cache-tuned uint16 column histograms and empty-block skipping.
    """
    nrow, ncol = cover.shape
    r0 = max(0, int(r_start))
    c0 = max(0, int(c_start))
    r1 = nrow if r_stop < 0 or r_stop > nrow else int(r_stop)
    c1 = ncol if c_stop < 0 or c_stop > ncol else int(c_stop)
    kernel = _KERNEL_U16 if window <= 65535 else _KERNEL_I32
    return kernel(
        cover,
        ground,
        window,
        lowerquantile,
        upperquantile,
        minvalpercent,
        r0,
        r1,
        c0,
        c1,
        strip_h,
        col_tile_w,
    )

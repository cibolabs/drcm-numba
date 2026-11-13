"""
Numba-accelerated reference pixel filter without IBRA stratification.

Uses a Serpentine 2D tiled incremental column-histogram sliding window with
L2-friendly uint16 column histograms (int32 fallback for window > 65535),
empty-block ocean skipping, aligned 104-bin vector updates, adaptive strip
sizing, and optional column slicing (c_start/c_stop) for halo-padded AOI runs.
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
    nsum = np.int64(0)
    bigsum = np.int64(0)
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
                    if 0 < pv <= 101:
                        dv = ground[r, c]
                        if 0 < dv <= 101:
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
def _tile_valid_row_range(blk, r0, r1, c0, c1, halfwindow, nrow, ncol, bsize):
    """Return active row sub-range [act_r0, act_r1) within [r0, r1) whose halo contains valid paired pixels."""
    wr0 = max(0, r0 - halfwindow)
    wr1 = min(nrow, r1 + halfwindow)
    wc0 = max(0, c0 - halfwindow)
    wc1 = min(ncol, c1 + halfwindow)
    br0 = wr0 // bsize
    br1 = (wr1 - 1) // bsize
    bc0 = wc0 // bsize
    bc1 = (wc1 - 1) // bsize
    first_br = br1 + 1
    last_br = br0 - 1
    for br in range(br0, br1 + 1):
        for bc in range(bc0, bc1 + 1):
            if blk[br, bc] != 0:
                if br < first_br:
                    first_br = br
                if br > last_br:
                    last_br = br
                break
    if first_br > last_br:
        return r1, r0
    act_r0 = max(r0, first_br * bsize - halfwindow)
    act_r1 = min(r1, (last_br + 1) * bsize + halfwindow)
    if act_r0 >= act_r1:
        return r1, r0
    return act_r0, act_r1


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


def _make_kernel(col_hist_dtype, sum_dtype=np.int32, tot_dtype=np.int32):
    """Compile a tiled sliding-window Numba kernel for requested histogram/sum dtypes."""
    one = col_hist_dtype(1)
    zero_tot = tot_dtype(0)

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
        col_drysum = np.zeros((tile_cols, 104), dtype=sum_dtype)
        col_perc_tot = np.zeros(tile_cols, dtype=tot_dtype)
        col_valid_tot = np.zeros(tile_cols, dtype=tot_dtype)

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
                    if 0 < dv <= 101:
                        col_dryhist[tc, pv] += one
                        col_drysum[tc, pv] += dv
                        col_valid_tot[tc] += 1

        percentilehist = np.zeros(104, dtype=tot_dtype)
        dryhist = np.zeros(104, dtype=tot_dtype)
        drysum = np.zeros(104, dtype=sum_dtype)

        perc_tot = zero_tot
        validpixels = zero_tot
        act_c0 = max(0, c0 - halfwindow)
        act_c1 = min(ncol, c0 + halfwindow + 1)
        for cc in range(act_c0, act_c1):
            tc = cc - w_c0
            pt = col_perc_tot[tc]
            if pt > 0:
                perc_tot += pt
                vt = col_valid_tot[tc]
                if vt > 0:
                    validpixels += vt
                    for b in range(104):
                        percentilehist[b] += tot_dtype(col_perc[tc, b])
                        dryhist[b] += tot_dtype(col_dryhist[tc, b])
                        drysum[b] += col_drysum[tc, b]
                else:
                    for b in range(104):
                        percentilehist[b] += tot_dtype(col_perc[tc, b])

        # Serpentine row sweep preserves the active 2D histogram at row turnarounds,
        # eliminating per-row column histogram clearing and re-summation.
        forward = True
        for r in range(r0, r1):
            if r > r0:
                if forward:
                    act_c0 = max(0, c0 - halfwindow)
                    act_c1 = min(ncol, c0 + halfwindow + 1)
                else:
                    c_end = c1 - 1
                    act_c0 = max(0, c_end - halfwindow)
                    act_c1 = min(ncol, c_end + halfwindow + 1)

                rem_r = r - 1 - halfwindow
                if rem_r >= 0:
                    for cc in range(w_c0, w_c1):
                        pv = cover[rem_r, cc]
                        if 0 < pv <= 101:
                            tc = cc - w_c0
                            in_act = act_c0 <= cc < act_c1
                            col_perc[tc, pv] -= one
                            col_perc_tot[tc] -= 1
                            if in_act:
                                perc_tot -= 1
                                percentilehist[pv] -= 1
                            dv = ground[rem_r, cc]
                            if 0 < dv <= 101:
                                col_dryhist[tc, pv] -= one
                                col_drysum[tc, pv] -= dv
                                col_valid_tot[tc] -= 1
                                if in_act:
                                    validpixels -= 1
                                    dryhist[pv] -= 1
                                    drysum[pv] -= dv

                add_r = r + halfwindow
                if add_r < nrow:
                    for cc in range(w_c0, w_c1):
                        pv = cover[add_r, cc]
                        if 0 < pv <= 101:
                            tc = cc - w_c0
                            in_act = act_c0 <= cc < act_c1
                            col_perc[tc, pv] += one
                            col_perc_tot[tc] += 1
                            if in_act:
                                perc_tot += 1
                                percentilehist[pv] += 1
                            dv = ground[add_r, cc]
                            if 0 < dv <= 101:
                                col_dryhist[tc, pv] += one
                                col_drysum[tc, pv] += dv
                                col_valid_tot[tc] += 1
                                if in_act:
                                    validpixels += 1
                                    dryhist[pv] += 1
                                    drysum[pv] += dv

            topindex_f = max(1, (r + 1) - halfwindow)
            bottomindex_f = min(nrow, (r + 1) + halfwindow)
            row_span = bottomindex_f - topindex_f
            out_r = r - r_start

            if forward:
                for c in range(c0, c1):
                    j_f = c + 1
                    rightindex_f = j_f + halfwindow
                    if j_f < halfwindow:
                        npixels = row_span * rightindex_f
                    elif rightindex_f > ncol:
                        npixels = row_span * (ncol - j_f)
                    else:
                        npixels = row_span * window

                    if validpixels >= minpercentv_f32 * npixels:
                        b0, b1 = _bounds_fast(percentilehist, perc_tot, 101, lowerq, upperq)
                        out[out_r, c - c_start] = _calcmean(drysum, dryhist, b0, b1)
                    else:
                        out[out_r, c - c_start] = -1

                    if c + 1 < c1:
                        left_c = c - halfwindow
                        if left_c >= 0:
                            tc = left_c - w_c0
                            pt = col_perc_tot[tc]
                            if pt > 0:
                                perc_tot -= pt
                                vt = col_valid_tot[tc]
                                if vt > 0:
                                    validpixels -= vt
                                    for b in range(104):
                                        percentilehist[b] -= tot_dtype(col_perc[tc, b])
                                        dryhist[b] -= tot_dtype(col_dryhist[tc, b])
                                        drysum[b] -= col_drysum[tc, b]
                                else:
                                    for b in range(104):
                                        percentilehist[b] -= tot_dtype(col_perc[tc, b])
                        next_right_c = (c + 1) + halfwindow
                        if next_right_c < ncol:
                            tc = next_right_c - w_c0
                            pt = col_perc_tot[tc]
                            if pt > 0:
                                perc_tot += pt
                                vt = col_valid_tot[tc]
                                if vt > 0:
                                    validpixels += vt
                                    for b in range(104):
                                        percentilehist[b] += tot_dtype(col_perc[tc, b])
                                        dryhist[b] += tot_dtype(col_dryhist[tc, b])
                                        drysum[b] += col_drysum[tc, b]
                                else:
                                    for b in range(104):
                                        percentilehist[b] += tot_dtype(col_perc[tc, b])
                forward = False
            else:
                for c in range(c1 - 1, c0 - 1, -1):
                    j_f = c + 1
                    rightindex_f = j_f + halfwindow
                    if j_f < halfwindow:
                        npixels = row_span * rightindex_f
                    elif rightindex_f > ncol:
                        npixels = row_span * (ncol - j_f)
                    else:
                        npixels = row_span * window

                    if validpixels >= minpercentv_f32 * npixels:
                        b0, b1 = _bounds_fast(percentilehist, perc_tot, 101, lowerq, upperq)
                        out[out_r, c - c_start] = _calcmean(drysum, dryhist, b0, b1)
                    else:
                        out[out_r, c - c_start] = -1

                    if c > c0:
                        right_c = c + halfwindow
                        if right_c < ncol:
                            tc = right_c - w_c0
                            pt = col_perc_tot[tc]
                            if pt > 0:
                                perc_tot -= pt
                                vt = col_valid_tot[tc]
                                if vt > 0:
                                    validpixels -= vt
                                    for b in range(104):
                                        percentilehist[b] -= tot_dtype(col_perc[tc, b])
                                        dryhist[b] -= tot_dtype(col_dryhist[tc, b])
                                        drysum[b] -= col_drysum[tc, b]
                                else:
                                    for b in range(104):
                                        percentilehist[b] -= tot_dtype(col_perc[tc, b])
                        prev_left_c = (c - 1) - halfwindow
                        if prev_left_c >= 0:
                            tc = prev_left_c - w_c0
                            pt = col_perc_tot[tc]
                            if pt > 0:
                                perc_tot += pt
                                vt = col_valid_tot[tc]
                                if vt > 0:
                                    validpixels += vt
                                    for b in range(104):
                                        percentilehist[b] += tot_dtype(col_perc[tc, b])
                                        dryhist[b] += tot_dtype(col_dryhist[tc, b])
                                        drysum[b] += col_drysum[tc, b]
                                else:
                                    for b in range(104):
                                        percentilehist[b] += tot_dtype(col_perc[tc, b])
                forward = True

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

        eff_strip_h = strip_h
        n_col_tiles = (n_out_c + col_tile_w - 1) // col_tile_w
        while eff_strip_h > 128 and ((n_out_r + eff_strip_h - 1) // eff_strip_h) * n_col_tiles < 256:
            eff_strip_h //= 2

        n_row_strips = (n_out_r + eff_strip_h - 1) // eff_strip_h
        n_tasks = n_row_strips * n_col_tiles

        for t in prange(n_tasks):
            rs = t // n_col_tiles
            ct = t % n_col_tiles
            r0 = r_start + rs * eff_strip_h
            r1 = min(r_stop, r0 + eff_strip_h)
            c0 = c_start + ct * col_tile_w
            c1 = min(c_stop, c0 + col_tile_w)
            if use_blk:
                act_r0, act_r1 = _tile_valid_row_range(blk, r0, r1, c0, c1, halfwindow, nrow, ncol, bsize)
                if act_r0 >= act_r1:
                    _fill_empty_tile(out, r0, r1, c0, c1, r_start, c_start, halfwindow, nrow, ncol)
                    continue
                if act_r0 > r0:
                    _fill_empty_tile(out, r0, act_r0, c0, c1, r_start, c_start, halfwindow, nrow, ncol)
                if act_r1 < r1:
                    _fill_empty_tile(out, act_r1, r1, c0, c1, r_start, c_start, halfwindow, nrow, ncol)
            else:
                act_r0, act_r1 = r0, r1
            _process_tile(
                cover,
                ground,
                out,
                act_r0,
                act_r1,
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


_KERNEL_U16 = _make_kernel(np.uint16, np.int32, np.int32)
_KERNEL_U16_I64 = _make_kernel(np.uint16, np.int64, np.int64)
_KERNEL_I32 = _make_kernel(np.int32, np.int64, np.int64)


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
    strip_h=16384,
    col_tile_w=512,
):
    """
    Numba-accelerated focal reference pixel filter over [r_start:r_stop, c_start:c_stop].

    Bit-for-bit compatible with the scalar sliding-window implementation while using
    a Serpentine 2D tiled L2 cache-tuned uint16 histogram sweep and empty-block skipping.
    """
    nrow, ncol = cover.shape
    r0 = max(0, int(r_start))
    c0 = max(0, int(c_start))
    r1 = nrow if r_stop < 0 or r_stop > nrow else int(r_stop)
    c1 = ncol if c_stop < 0 or c_stop > ncol else int(c_stop)
    if r0 > r1:
        r0 = r1
    if c0 > c1:
        c0 = c1
    if window <= 4611:
        kernel = _KERNEL_U16
    elif window <= 65535:
        kernel = _KERNEL_U16_I64
    else:
        kernel = _KERNEL_I32
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

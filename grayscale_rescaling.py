def norm_band(img_band, band_max, band_min):
    vis_band = img_band.copy()
    vis_band = (vis_band - band_min) / (band_max - band_min) * 255.0
    return vis_band.astype(np.uint8)


def img_norm_range(img_band, ratio_th=0.95):
    n_bins = 200
    im_counts, im_ths = np.histogram(img_band, n_bins)

    cum_ratios = np.cumsum(im_counts) * 1.0 / np.sum(im_counts)

    low_thidx = int(np.sum(cum_ratios <= (1.0 - ratio_th)))
    low_th = im_ths[low_thidx]

    hig_thidx = int(np.sum(cum_ratios <= ratio_th))
    hig_th = im_ths[hig_thidx]

    search_step = 1
    low2hig_pxdists = list(); low_hig_idxpairs = list()

    if low_th > (len(im_ths) - hig_th):
        for low_idx in np.arange(0, low_thidx, search_step):
            low_idx_ratioval = cum_ratios[low_idx]
            hig_idx_ratioval = low_idx_ratioval + ratio_th

            hig_idx = np.sum(cum_ratios <= hig_idx_ratioval)
            low_hig_idxpairs.append((low_idx, hig_idx))
            low2hig_pxdists.append(hig_idx - low_idx)
    else:
        for hig_idx in np.arange(hig_thidx, len(im_ths), search_step):
            hig_idx_ratioval = cum_ratios[hig_idx]
            low_idx_ratioval = hig_idx_ratioval - ratio_th

            low_idx = np.sum(cum_ratios <= low_idx_ratioval)
            low_hig_idxpairs.append((low_idx, hig_idx))
            low2hig_pxdists.append(hig_idx - low_idx)

    if len(low2hig_pxdists) > 0:
        mindist_idx = np.argmin(low2hig_pxdists)
        mindist_lowidx, maxdist_higidx = low_hig_idxpairs[mindist_idx]
        low_pxval = im_ths[mindist_lowidx]; hig_pxval = im_ths[maxdist_higidx]
    else:
        low_pxval = im_ths[0]; hig_pxval = im_ths[-1]

    return low_pxval, hig_pxval


def adapt_norm_band(img_band):
    vis_band = img_band.copy()
    l_rng, h_rng = img_norm_range(vis_band)
    vis_band = (vis_band - l_rng) / (h_rng - l_rng) * 255.0
    return vis_band.astype(np.uint8)

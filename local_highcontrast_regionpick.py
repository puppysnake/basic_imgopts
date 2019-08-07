def get_local_maxima(ir_banddata, regsize=64, regstep=16, maxima_rt=0.9):
    ir_h, ir_w = ir_banddata.shape[0:2]

    hor_starts = np.arange(0, ir_w - regsize, regstep)
    if hor_starts[-1] <= ir_w - regsize - regsize / 2:
        hor_starts = np.array(hor_starts.tolist().append(ir_w - regsize))

    ver_starts = np.arange(0, ir_h - regsize, regstep)
    if ver_starts[-1] <= ir_h - regsize - regsize / 2:
        ver_starts = np.array(ver_starts.tolist().append(ir_h - regsize))
    
    local_filtered = np.zeros(ir_banddata.shape[0:2], dtype=np.float32)
    hor_starts_mesh, ver_starts_mesh = np.meshgrid(hor_starts, ver_starts)
    local_nbins = 100
    for hor_start, ver_start in zip(hor_starts_mesh.flatten(), ver_starts_mesh.flatten()):
        if len(ir_banddata.shape) > 2:
            cur_ir_crop = ir_banddata[ver_start:ver_start + regsize, hor_start:hor_start + regsize, 0]
        else:
            cur_ir_crop = ir_banddata[ver_start:ver_start + regsize, hor_start:hor_start + regsize]
        cur_counts, cur_ths = np.histogram(cur_ir_crop.flatten(), local_nbins)   

        cur_cumrts = np.cumsum(cur_counts) * 1.0 / np.sum(cur_counts)
        cur_thpxval = cur_ths[np.sum(cur_cumrts <= maxima_rt) - 1]

        cur_x_idxs, cur_y_idxs = np.meshgrid(np.arange(hor_start, hor_start + regsize), 
                np.arange(ver_start, ver_start + regsize))
        
        cur_ir_rectcrop = cur_ir_crop[:, :]
        cur_ir_rectcrop = cur_ir_rectcrop - cur_thpxval
        cur_ir_rectcrop[cur_ir_rectcrop <= 0] = 0

        local_filtered[cur_y_idxs, cur_x_idxs] += cur_ir_rectcrop

    # convert local_filtered to binary image (the 2-valued image)
    local_filter_min = local_filtered.min()
    local_filter_max = local_filtered.max()

    # import pdb; pdb.set_trace()
    local_filter_th = 0.1 * (local_filter_max - local_filter_min) + local_filter_min
    local_filter_binary = (local_filtered >= local_filter_th).astype(np.float32)
    
    #plt.imshow(local_filter_binary); plt.show()
    return local_filter_binary

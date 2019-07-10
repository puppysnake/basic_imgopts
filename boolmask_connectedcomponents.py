def mask_conncomps_proc(mask_img):
    # convert the image to grayscale image
    bools_img = (mask_img > 0).astype(np.uint8) * 255
    
    cc_sw = 2
    if cc_sw == 1:
        n_cnncomp, cnncomp_mask = cv2.connectedComponents(bools_img)
    elif cc_sw == 2:
        connectivity = 8
        ret, thresh = cv2.threshold(bools_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)

        n_cnncomp = output[0]
        cnncomp_mask = output[1]

    cnncomps_idxslist = list()
    cnncomps_areas = list()
    cnncomps_centers = list()

    for nc_i in range(n_cnncomp - 1):
        cnncomp_bools = cnncomp_mask == (nc_i + 1)
        cnncomp_idxs = np.where(cnncomp_bools)
        
        #print('Current Area[%d]: %d' % (nc_i + 1, np.sum(cnncomp_bools)))
        cnncomps_idxslist.append(cnncomp_idxs)
        cnncomps_areas.append(len(cnncomp_idxs[0]))

        # get the center coordinates
        y_mean = np.mean(cnncomp_idxs[0])
        x_mean = np.mean(cnncomp_idxs[1])
        cnncomps_centers.append(np.array([y_mean, x_mean]))

    return cnncomps_centers, cnncomps_areas, cnncomps_idxslist

def rot90degs_imgdata_irs(in_irsdata):
    irs_h, irs_w = in_irsdata.shape
    deg0_irsdata = in_irsdata.copy()

    deg90_meshY, deg90_meshX = np.meshgrid(np.arange(irs_h -1, -1, -1), np.arange(0, irs_w, 1))
    deg90_irsdata = in_irsdata[deg90_meshY, deg90_meshX].copy()

    deg180_meshY, deg180_meshX = np.meshgrid(np.arange(irs_h - 1, -1, -1), np.arange(irs_w - 1, -1, -1))
    deg180_irsdata = in_irsdata[deg180_meshY, deg180_meshX].copy().T

    deg270_meshY, deg270_meshX = np.meshgrid(np.arange(0, irs_h, 1), np.arange(irs_w - 1, -1, -1))
    deg270_irsdata = in_irsdata[deg270_meshY, deg270_meshX].copy()

    return deg0_irsdata, deg90_irsdata, deg180_irsdata, deg270_irsdata


def rot90degs_imgdata_pms(in_pmsdata):
    pms_h, pms_w = in_pmsdata.shape[0:2]
    deg0_pmsdata = in_pmsdata.copy()

    deg90_meshY, deg90_meshX = np.meshgrid(np.arange(pms_h -1, -1, -1), np.arange(0, pms_w, 1))
    deg90_pmsdata = in_pmsdata[deg90_meshY, deg90_meshX, :].copy()

    deg180_meshY, deg180_meshX = np.meshgrid(np.arange(pms_h - 1, -1, -1), np.arange(pms_w - 1, -1, -1))
    deg180_pmsdata = np.swapaxes(in_pmsdata[deg180_meshY, deg180_meshX, :].copy(), 0, 1)

    deg270_meshY, deg270_meshX = np.meshgrid(np.arange(0, pms_h, 1), np.arange(pms_w - 1, -1, -1))
    deg270_pmsdata = in_pmsdata[deg270_meshY, deg270_meshX, :].copy()

    return deg0_pmsdata, deg90_pmsdata, deg180_pmsdata, deg270_pmsdata

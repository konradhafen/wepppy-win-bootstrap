import os

def is_comment(s):
    return s.startswith('#')


def change_soil_params(fn, anisotropy=None, field_cap=None, pct_rock=None, kr=None, comment_id="#"):
    with open(fn, 'r') as fh:
        dat = fh.readlines()
        nline = 0
        ksat_pass = False
        for line in dat:
            if not line.startswith(comment_id):
                line_dat = list(line.strip().split())
                if len(line_dat) == 11:
                    ksat_pass = True
                    if anisotropy is not None:
                        line_dat[3] = str(anisotropy)
                    if field_cap is not None:
                        line_dat[4] = str(field_cap)
                    if pct_rock is not None:
                        line_dat[10] = str(pct_rock)
                    dat[nline] = '\t' + '\t'.join(line_dat) + '\n'
                elif ksat_pass and len(line_dat) == 3 and kr is not None:
                    # print('hopefully this is restrictive layer data', dat[nline])
                    line_dat[2] = str(kr)
                dat[nline] = '\t'.join(line_dat) + '\n'
                    # print(dat[nline])
            nline += 1
        with open(fn, 'w') as fo:
            fo.writelines(dat)
    return


def soil_prep(wd, anisotropy=None, kr=None, field_cap=None, pct_rock=None):
    comment_id = "#"
    directory = os.fsencode(os.path.join(wd, 'wepp/runs'))
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".sol"):
            change_soil_params(os.path.join(os.fsdecode(directory), filename), anisotropy=anisotropy, field_cap=field_cap, pct_rock=pct_rock, kr=kr, comment_id=comment_id)


# fn = r"E:\konrad\Projects\usgs\hjandrews\wepp\hja-ws1-base\wepp\runs\p1.sol"
# change_soil_params(fn, anisotropy=100.0, kr=1.0)
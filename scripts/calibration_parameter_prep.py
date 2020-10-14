from itertools import dropwhile


def is_comment(s):
    # print(s.startswith('#'))
    return s.startswith('#')


def soil_params(fn, kvmod=2.0, amod=2.0, comment_id="#"):
    fn_mod = r"E:\konrad\Projects\usgs\hjandrews\wepp\hja-ws1-base\wepp\runs\p1_edit.sol"
    with open(fn, 'r') as fh:
        dat = fh.readlines()
        nline = 0
        ksat_pass = False
        for line in dat:
            if not line.startswith(comment_id):
                line_dat = list(line.strip().split())
                if len(line_dat) == 11:
                    line_dat[3] = str(float(line_dat[3]) * amod)
                    dat[nline] = '\t' + '\t'.join(line_dat) + '\n'
                    ksat_pass = True
                elif ksat_pass and len(line_dat) == 3:
                    print('hopefully this is restrictive layer data', dat[nline])
                    line_dat[2] = str(float(line_dat[2]) * kvmod)
                    dat[nline] = '\t'.join(line_dat) + '\n'
                    print(dat[nline])
            nline += 1
        with open(fn_mod, 'w') as fo:
            fo.writelines(dat)
    return

fn = r"E:\konrad\Projects\usgs\hjandrews\wepp\hja-ws1-base\wepp\runs\p1.sol"
soil_params(fn, kvmod=0.5)
import os

class SnowOpts(object):
    def __init__(self, rs_thresh=None, new_dens=None, set_dens=None):
        """
        Stores coeffs that go into snow.txt
        Args:
            rs_thresh: rain/snow threshold
            new_dens: density of new snow
            set_dens: snow settling density
        """
        if rs_thresh is None:
            self.rs_thresh = -2.0
        else:
            self.rs_thresh = rs_thresh
        if new_dens is None:
            self.new_dens = 100.0
        else:
            self.new_dens = new_dens
        if set_dens is None:
            self.set_dens = 250.0
        else:
            self.set_dens = set_dens

    @property
    def contents(self):
        return (
            '{0.rs_thresh}\t#Rain/snow threshold\n'
            '{0.new_dens}\t#Density of new snow\n'
            '{0.set_dens}\t#Snow settling density\n'
            .format(self)
        )


def gwcoeff_prep(runs_dir, rs_thresh=None, new_dens=None, set_dens=None):
    snow_opts = SnowOpts(rs_thresh=rs_thresh, new_dens=new_dens, set_dens=set_dens)

    with open(os.path.join(runs_dir, 'snow.txt'), 'w') as fp:
        fp.write(snow_opts.contents)

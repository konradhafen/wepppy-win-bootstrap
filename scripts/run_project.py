import sys
from glob import glob
import os
from os.path import join as _join
from os.path import split as _split
from os.path import exists
from time import time
import argparse
import subprocess
import multiprocessing
import shutil

from concurrent.futures import (
    ThreadPoolExecutor, 
    as_completed, 
    wait, 
    FIRST_EXCEPTION
)

from wy_calc import wy_calc


NCPU = multiprocessing.cpu_count() - 1
if NCPU < 1:
    NCPU = 1
    
USE_MULTIPROCESSING = True

wepp_exe = "../bin/WEPP2014.exe"

perl_exe = r"C:\Perl64\bin\perl.exe"
daily_hillslopes_pl_path = "../bin/correct_daily_hillslopes.pl"


def run_hillslope(wepp_id, runs_dir):
    t0 = time()

    cmd = [os.path.abspath(wepp_exe)]

    assert exists(_join(runs_dir, 'p%i.man' % wepp_id))
    assert exists(_join(runs_dir, 'p%i.slp' % wepp_id))
    assert exists(_join(runs_dir, 'p%i.cli' % wepp_id))
    assert exists(_join(runs_dir, 'p%i.sol' % wepp_id))

    _run = open(_join(runs_dir, 'p%i.run' % wepp_id))
    _log = open(_join(runs_dir, 'p%i.err' % wepp_id), 'w')

    p = subprocess.Popen(cmd, stdin=_run, stdout=_log, stderr=_log, cwd=runs_dir)
    p.wait()
    _run.close()
    _log.close()

    log_fn = _join(runs_dir, 'p%i.err' % wepp_id)
    with open(log_fn) as fp:
        lines = fp.readlines()
        for L in lines:
            if 'WEPP COMPLETED HILLSLOPE SIMULATION SUCCESSFULLY' in L:
                return True, wepp_id, time() - t0

    raise Exception('Error running wepp for wepp_id %i\nSee %s'
                    % (wepp_id, log_fn))


def run_watershed(runs_dir, output_dir):
    t0 = time()

    cmd = [os.path.abspath(wepp_exe)]

    assert exists(_join(runs_dir, 'pw0.str'))
    assert exists(_join(runs_dir, 'pw0.chn'))
    assert exists(_join(runs_dir, 'pw0.imp'))
    assert exists(_join(runs_dir, 'pw0.man'))
    assert exists(_join(runs_dir, 'pw0.slp'))
    assert exists(_join(runs_dir, 'pw0.cli'))
    assert exists(_join(runs_dir, 'pw0.sol'))
    assert exists(_join(runs_dir, 'pw0.run'))

    _run = open(_join(runs_dir, 'pw0.run'))
    _log = open(_join(runs_dir, 'pw0.err'), 'w')

    p = subprocess.Popen(cmd, stdin=_run, stdout=_log, stderr=_log, cwd=runs_dir)
    p.wait()
    _run.close()
    _log.close()

    log_fn = _join(runs_dir, 'pw0.err')

    for fn in glob(_join(runs_dir, '*.out')):
        dst_path = _join(output_dir, _split(fn)[1])
        shutil.move(fn, dst_path)
        
    with open(_join(runs_dir, 'pw0.err')) as fp:
        stdout = fp.read()
        if 'WEPP COMPLETED WATERSHED SIMULATION SUCCESSFULLY' in stdout:
            return True, time() - t0
            
    raise Exception('Error running wepp for watershed \nSee <a href="browse/wepp/runs/pw0.err">%s</a>' % log_fn)


def isint(x):
    # noinspection PyBroadException
    try:
        return float(int(x)) == float(x)
    except Exception:
        return False


def oncomplete(wepprun):
    status, _id, elapsed_time = wepprun.result()
    assert status
    print('  {} completed run in {}s\n'.format(_id, elapsed_time))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('wd', type=str,   
                        help='path of project')
    parser.add_argument('-n', '--numcpu',   type=int, 
                        help='Number of cpus in pool    (1)')
    parser.add_argument('--wy_calc_start_year',   type=int, 
                        help='run WY Calc postprocessing routine')   
    parser.add_argument('--no_multiprocessing',
                        help='Disable multiprocessing', action='store_true')    
    args = parser.parse_args()

    wd = args.wd
    numcpu = (args.numcpu, NCPU)[args.numcpu is None]
    wy_calc_start_year = args.wy_calc_start_year
    
    no_multiprocessing = (args.no_multiprocessing, False)[args.no_multiprocessing is None]
    if no_multiprocessing:
        USE_MULTIPROCESSING = False
        
    print('USE_MULTIPROCESSING', USE_MULTIPROCESSING)
    
    assert not wd.endswith('.py')
    assert exists(wd)

    print('project run_id', wd)

    runs_dir = _join(wd, 'wepp/runs')
    output_dir = _join(wd, 'wepp/output')

    assert exists(runs_dir)
    assert exists(output_dir)

    hillslope_runs = glob(_join(runs_dir, 'p*.run'))
    hillslope_runs = [run for run in hillslope_runs if 'pw' not in run]
    
    print('cleaning output dir')
    if exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    if USE_MULTIPROCESSING:
        pool = ThreadPoolExecutor(NCPU)
        futures = []

    for hillslope_run in hillslope_runs:
        
        run_fn = _split(hillslope_run)[-1]
        wepp_id = run_fn.replace('p', '').replace('.run', '')
        assert isint(wepp_id), wepp_id
        
        if USE_MULTIPROCESSING:
            futures.append(pool.submit(lambda p: run_hillslope(*p), (int(wepp_id), runs_dir)))
            futures[-1].add_done_callback(oncomplete)
        else:
            status, _id, elapsed_time = run_hillslope(int(wepp_id), runs_dir)
            assert status
            print('  {} completed run in {}s\n'.format(_id, elapsed_time))

    if USE_MULTIPROCESSING:
        wait(futures, return_when=FIRST_EXCEPTION)
    
    run_watershed(runs_dir, output_dir)
    print('completed watershed run')

    totwatsed_pl = _join(output_dir, 'correct_daily_hillslopes.pl')
    if exists(totwatsed_pl):
        os.remove(totwatsed_pl)

    shutil.copyfile(daily_hillslopes_pl_path, totwatsed_pl)
    shutil.copyfile("../bin/ls.bat", _join(output_dir, "ls.bat"))
    shutil.copyfile("../bin/rm.bat", _join(output_dir, "rm.bat"))

    cmd = [perl_exe, 'correct_daily_hillslopes.pl']
    _log = open(_join(output_dir, 'correct_daily_hillslopes.log'), 'w')

    p = subprocess.Popen(cmd, stdout=_log, stderr=_log, cwd=output_dir)
    p.wait()
    _log.close()
    
    if wy_calc_start_year is not None:
        wy_calc(wy_calc_start_year, output_dir)
    
    print('done')

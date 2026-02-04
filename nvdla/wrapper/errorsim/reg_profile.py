
import os
import sys
import argparse
import tempfile
import concurrent.futures as cf

# --- YAML C Interface (when available) --- #
import yaml

try:
    # Per PyYAML docs: use the C-accelerated variants when available.
    from yaml import CSafeLoader as YLoader, CSafeDumper as YDumper
except ImportError:
    from yaml import SafeLoader as YLoader, SafeDumper as YDumper

from pynvdla import nvdla

###################################

units=[
    'top.conv.cdma',
    'top.conv.cbuf',
    'top.conv.cbuf.datbuf',
    'top.conv.cbuf.wtbuf',
    'top.conv.csb',
    'top.conv.csc',
    'top.conv.dl',
    'top.conv.wl',
    'top.conv.cmac',
    'top.conv.cacc',
    'top.conv.dbuf',
    'top.conv',
    'top.sdp.csb',
    'top.sdp.core',
    'top.sdp',
    'top'
]

###################################

# def getRegs(config, workdir):
#     reglog = {}
#     reglog['regs'] = {}

#     workcfg = os.path.join(workdir, 'tmp_regprofile_cfg.yaml')

#     with open(config, 'r') as f:
#         cfg_dict = yaml.load(f, Loader=YLoader)
    
#     # Profile Ctrl Regs
#     print('-I: Profiling Ctrl registers')
#     cfg_dict['inject']['datapath'] = False
#     cfg_dict['inject']['ctrlpath'] = True

#     with open(workcfg, 'w') as f:
#         yaml.dump(cfg_dict, f)
    
#     print('-I: Creating NVDLA config...')
#     dut = nvdla(workcfg, profiler=True)

#     for unit in units:
#         reglog['regs'][unit] = {}
#         print(f'-I: extracting {unit} registers...')
#         reglog['regs'][unit]['ctrl'] = dut.get_regs(unit)

#     # Profile Data Regs
#     print('-I: Profiling Data registers')
#     cfg_dict['inject']['datapath'] = True
#     cfg_dict['inject']['ctrlpath'] = False

#     with open(workcfg, 'w') as f:
#         yaml.dump(cfg_dict, f)
    
#     print('-I: Creating NVDLA config...')
#     dut = nvdla(workcfg, profiler=True)

#     for unit in units:
#         print(f'-I: extracting {unit} registers...')
#         reglog['regs'][unit]['data'] = dut.get_regs(unit)

#     # Profile Total Regs
#     print('-I: Profiling Total registers')
#     cfg_dict['inject']['datapath'] = True
#     cfg_dict['inject']['ctrlpath'] = True

#     with open(workcfg, 'w') as f:
#         yaml.dump(cfg_dict, f)
    
#     dut = nvdla(workcfg, profiler=True)

#     for unit in units:
#         reglog['regs'][unit]['total'] = dut.get_regs(unit)
    
#     # Exit
#     os.remove(workcfg)

#     return reglog
def profile_helper(units, workcfg_path, datapath: bool, ctrlpath: bool, max_jobs = 8):
    """
    One profiling pass: configure inject flags, instantiate DUT once,
    then fetch all unit regs in parallel.
    """

    def _fetch(u):
        # single C-call; ideal for threads if GIL is released inside
        return u, dut.get_regs(u)

    dut = nvdla(workcfg_path, profiler=True)

    # Threading: many C extensions release the GIL (if pynvdla does, threads scale);
    # else fallback to processes via EXECUTOR_MODE="process"
    EXECUTOR_MODE = os.getenv("GETREGS_EXECUTOR", "thread")  # "thread" or "process"
    max_workers = min(len(units), (os.cpu_count() or 8))
    max_workers = min(max_workers, max_jobs)

    if EXECUTOR_MODE == "process":
        # NOTE: picklability of 'dut' may be an issue; processes only if pynvdla is process-safe.
        with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
            return dict(ex.map(lambda u: (u, nvdla(workcfg_path, profiler=True).get_regs(u)), units))
    else:
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            return dict(ex.map(_fetch, units))


def getRegs(config, workdir):
    """
    Optimized getRegs:
    - Loads YAML once
    - Uses a single NamedTemporaryFile and only rewrites inject flags
    - Parallelizes get_regs calls per pass
    """

    reglog = {'regs': {u: {} for u in units}}

    # Load once (safe loader is already used in the original code)
    with open(config, 'r') as f:
        cfg_dict = yaml.load(f, Loader=YLoader)  # same semantics as original

    # Prepare a single temporary work-config file
    os.makedirs(workdir, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=workdir, prefix='regprofile_', suffix='.yaml', delete=False) as tf:
        workcfg = tf.name

    try:
        # Define the three passes compactly
        passes = (
            ("ctrl",  False, True,  "-I: Profiling Ctrl registers"),
            ("data",  True,  False, "-I: Profiling Data registers"),
            ("total", True,  True,  "-I: Profiling Total registers"),
        )

        for key, datapath, ctrlpath, banner in passes:
            print(banner)

            # Flip inject flags and dump quickly (avoid reformatting unrelated YAML)
            inj = cfg_dict.setdefault('inject', {})
            inj['datapath'] = bool(datapath)
            inj['ctrlpath'] = bool(ctrlpath)

            with open(workcfg, 'w') as f:
                yaml.dump(cfg_dict, f, Dumper=YDumper)

            # One DUT instantiation per pass; parallelize unit queries
            results = profile_helper(units, workcfg, datapath, ctrlpath)

            # Fill results without extra dict creations
            for u in units:
                reglog['regs'][u][key] = results[u]

    finally:
        # Ensure temp file is cleaned even on exceptions
        try:
            os.remove(workcfg)
        except OSError:
            pass

    return reglog



def reglog_check(reglog):
    
    # Check single units
    for unit in units:
        unitlog   = reglog['regs'][unit]
        summation = unitlog['data'] + unitlog['ctrl']
        if(unitlog['total'] != summation):
            print(f'-E: Mismatching reg count in {unit}')
            print(f'-E: Data regs:  {unitlog['data']}')
            print(f'-E: Ctrl regs:  {unitlog['ctrl']}')
            print(f'-E: Total regs: {unitlog['total']}')
            print(f'-E: Expected {unitlog['total']-summation} registers')
            sys.exit(1)
    
    # Check hier modules
    if(reglog['regs']['top.conv.cbuf']['data']  != (reglog['regs']['top.conv.cbuf.datbuf']['data'] + reglog['regs']['top.conv.cbuf.wtbuf']['data'])):
        print('-E: Mismatch in top.conv.cbuf data reg count')
        print(f'-E: Expected: {reglog['regs']['top.conv.cbuf']['data']}')
        print(f'-E: Obtained: {(reglog['regs']['top.conv.cbuf.datbuf']['data'] + reglog['regs']['top.conv.cbuf.wtbuf']['data'])}')
        sys.exit(2)

    if(reglog['regs']['top.conv.cbuf']['ctrl']  < (reglog['regs']['top.conv.cbuf.datbuf']['ctrl'] + reglog['regs']['top.conv.cbuf.wtbuf']['ctrl'])):
        print('-E: Mismatch in top.conv.cbuf ctrl reg count')
        print(f'-E: Expected: {reglog['regs']['top.conv.cbuf']['ctrl']}')
        print(f'-E: Obtained: {(reglog['regs']['top.conv.cbuf.datbuf']['ctrl'] + reglog['regs']['top.conv.cbuf.wtbuf']['ctrl'])}')
        sys.exit(2)

    if(reglog['regs']['top.conv.cbuf']['total'] < (reglog['regs']['top.conv.cbuf.datbuf']['total'] + reglog['regs']['top.conv.cbuf.wtbuf']['total'])):
        print('-E: Mismatch in top.conv.cbuf total reg count')
        print(f'-E: Expected: {reglog['regs']['top.conv.cbuf']['total']}')
        print(f'-E: Obtained: {(reglog['regs']['top.conv.cbuf.datbuf']['total'] + reglog['regs']['top.conv.cbuf.wtbuf']['total'])}')
        sys.exit(2)


    if(reglog['regs']['top.conv']['data']       != (reglog['regs']['top.conv.cdma']['data'] +
                                                   reglog['regs']['top.conv.cbuf']['data'] + 
                                                   reglog['regs']['top.conv.csb']['data'] + 
                                                   reglog['regs']['top.conv.csc']['data'] + 
                                                   reglog['regs']['top.conv.dl']['data'] + 
                                                   reglog['regs']['top.conv.wl']['data'] + 
                                                   reglog['regs']['top.conv.cmac']['data'] + 
                                                   reglog['regs']['top.conv.cacc']['data'] + 
                                                   reglog['regs']['top.conv.dbuf']['data'])):
        print('-E: Mismatch in top.conv data reg count')
        print(f'-E: Expected: {reglog['regs']['top.conv']['data']}')
        print(f'-E: Obtained: {(reglog['regs']['top.conv.cdma']['data'] +
                                reglog['regs']['top.conv.cbuf']['data'] + 
                                reglog['regs']['top.conv.csb']['data'] + 
                                reglog['regs']['top.conv.csc']['data'] + 
                                reglog['regs']['top.conv.dl']['data'] + 
                                reglog['regs']['top.conv.wl']['data'] + 
                                reglog['regs']['top.conv.cmac']['data'] + 
                                reglog['regs']['top.conv.cacc']['data'] + 
                                reglog['regs']['top.conv.dbuf']['data'])}')
        sys.exit(2)

    if(reglog['regs']['top.conv']['ctrl']       < (reglog['regs']['top.conv.cdma']['ctrl'] +
                                                   reglog['regs']['top.conv.cbuf']['ctrl'] + 
                                                   reglog['regs']['top.conv.csb']['ctrl'] + 
                                                   reglog['regs']['top.conv.csc']['ctrl'] + 
                                                   reglog['regs']['top.conv.dl']['ctrl'] + 
                                                   reglog['regs']['top.conv.wl']['ctrl'] + 
                                                   reglog['regs']['top.conv.cmac']['ctrl'] + 
                                                   reglog['regs']['top.conv.cacc']['ctrl'] + 
                                                   reglog['regs']['top.conv.dbuf']['ctrl'])):
        print('-E: Mismatch in top.conv ctrl reg count')
        print(f'-E: Expected: {reglog['regs']['top.conv']['ctrl']}')
        print(f'-E: Obtained: {(reglog['regs']['top.conv.cdma']['ctrl'] +
                                reglog['regs']['top.conv.cbuf']['ctrl'] + 
                                reglog['regs']['top.conv.csb']['ctrl'] + 
                                reglog['regs']['top.conv.csc']['ctrl'] + 
                                reglog['regs']['top.conv.dl']['ctrl'] + 
                                reglog['regs']['top.conv.wl']['ctrl'] + 
                                reglog['regs']['top.conv.cmac']['ctrl'] + 
                                reglog['regs']['top.conv.cacc']['ctrl'] + 
                                reglog['regs']['top.conv.dbuf']['ctrl'])}')
        sys.exit(2)

    if(reglog['regs']['top.conv']['total']      < (reglog['regs']['top.conv.cdma']['total'] +
                                                   reglog['regs']['top.conv.cbuf']['total'] + 
                                                   reglog['regs']['top.conv.csb']['total'] + 
                                                   reglog['regs']['top.conv.csc']['total'] + 
                                                   reglog['regs']['top.conv.dl']['total'] + 
                                                   reglog['regs']['top.conv.wl']['total'] + 
                                                   reglog['regs']['top.conv.cmac']['total'] + 
                                                   reglog['regs']['top.conv.cacc']['total'] + 
                                                   reglog['regs']['top.conv.dbuf']['total'])):
        print('-E: Mismatch in top.conv total reg count')
        print(f'-E: Expected: {reglog['regs']['top.conv']['total']}')
        print(f'-E: Obtained: {(reglog['regs']['top.conv.cdma']['total'] +
                                reglog['regs']['top.conv.cbuf']['total'] + 
                                reglog['regs']['top.conv.csb']['total'] + 
                                reglog['regs']['top.conv.csc']['total'] + 
                                reglog['regs']['top.conv.dl']['total'] + 
                                reglog['regs']['top.conv.wl']['total'] + 
                                reglog['regs']['top.conv.cmac']['total'] + 
                                reglog['regs']['top.conv.cacc']['total'] + 
                                reglog['regs']['top.conv.dbuf']['total'])}')
        sys.exit(2)


    if(reglog['regs']['top.sdp']['data']        != (reglog['regs']['top.sdp.csb']['data'] +
                                                   reglog['regs']['top.sdp.core']['data'])):
        print('-E: Mismatch in top.sdp data reg count')
        print(f'-E: Expected: {reglog['regs']['top.sdp']['data']}')
        print(f'-E: Obtained: {(reglog['regs']['top.sdp.csb']['data'] +
                                reglog['regs']['top.sdp.core']['data'])}')
        sys.exit(2)

    if(reglog['regs']['top.sdp']['ctrl']        != (reglog['regs']['top.sdp.csb']['ctrl'] +
                                                   reglog['regs']['top.sdp.core']['ctrl'])):
        print('-E: Mismatch in top.sdp ctrl reg count')
        print(f'-E: Expected: {reglog['regs']['top.sdp']['ctrl']}')
        print(f'-E: Obtained: {(reglog['regs']['top.sdp.csb']['ctrl'] +
                                reglog['regs']['top.sdp.core']['ctrl'])}')
        sys.exit(2)

    if(reglog['regs']['top.sdp']['total']       != (reglog['regs']['top.sdp.csb']['total'] +
                                                   reglog['regs']['top.sdp.core']['total'])):
        print('-E: Mismatch in top.sdp total reg count')
        print(f'-E: Expected: {reglog['regs']['top.sdp']['total']}')
        print(f'-E: Obtained: {(reglog['regs']['top.sdp.csb']['total'] +
                                reglog['regs']['top.sdp.core']['total'])}')
        sys.exit(2)

    if(reglog['regs']['top']['data']           != (reglog['regs']['top.conv']['data'] + reglog['regs']['top.sdp']['data'])):
        print('-E: Mismatch in top data reg count')
        print(f'-E: Expected: {reglog['regs']['top']['data']}')
        print(f'-E: Obtained: {(reglog['regs']['top.conv']['data'] + reglog['regs']['top.sdp']['data'])}')
        sys.exit(2)

    if(reglog['regs']['top']['ctrl']           != (reglog['regs']['top.conv']['ctrl'] + reglog['regs']['top.sdp']['ctrl'])):
        print('-E: Mismatch in top ctrl reg count')
        print(f'-E: Expected: {reglog['regs']['top']['ctrl']}')
        print(f'-E: Obtained: {(reglog['regs']['top.conv']['ctrl'] + reglog['regs']['top.sdp']['ctrl'])}')
        sys.exit(2)

    if(reglog['regs']['top']['total']          != (reglog['regs']['top.conv']['total'] + reglog['regs']['top.sdp']['total'])):
        print('-E: Mismatch in top total reg count')
        print(f'-E: Expected: {reglog['regs']['top']['total']}')
        print(f'-E: Obtained: {(reglog['regs']['top.conv']['total'] + reglog['regs']['top.sdp']['total'])}')
        sys.exit(2)


def units_probability(reglog, check = True, threshold = 1e-6):

    total = float(reglog['regs']['top']['total'])
    seulog = {}

    seulog['top.conv.cacc'] = {}
    seulog['top.conv.cacc']['data'] = float(reglog['regs']['top.conv.cacc']['data']) / total
    seulog['top.conv.cacc']['ctrl'] = float(reglog['regs']['top.conv.cacc']['ctrl']) / total

    seulog['top.conv.cbuf.datbuf'] = {}
    seulog['top.conv.cbuf.datbuf']['data'] = float(reglog['regs']['top.conv.cbuf.datbuf']['data']) / total
    seulog['top.conv.cbuf.datbuf']['ctrl'] = float(reglog['regs']['top.conv.cbuf.datbuf']['ctrl']) / total

    seulog['top.conv.cbuf.wtbuf'] = {}
    seulog['top.conv.cbuf.wtbuf']['data'] = float(reglog['regs']['top.conv.cbuf.wtbuf']['data']) / total
    seulog['top.conv.cbuf.wtbuf']['ctrl'] = float(reglog['regs']['top.conv.cbuf.wtbuf']['ctrl']) / total

    seulog['top.conv.cdma'] = {}
    seulog['top.conv.cdma']['data'] = float(reglog['regs']['top.conv.cdma']['data']) / total
    seulog['top.conv.cdma']['ctrl'] = float(reglog['regs']['top.conv.cdma']['ctrl']) / total

    seulog['top.conv.cmac'] = {}
    seulog['top.conv.cmac']['data'] = float(reglog['regs']['top.conv.cmac']['data']) / total
    seulog['top.conv.cmac']['ctrl'] = float(reglog['regs']['top.conv.cmac']['ctrl']) / total

    seulog['top.conv.csb'] = {}
    seulog['top.conv.csb']['data'] = float(reglog['regs']['top.conv.csb']['data']) / total
    seulog['top.conv.csb']['ctrl'] = float(reglog['regs']['top.conv.csb']['ctrl']) / total

    seulog['top.conv.csc'] = {}
    seulog['top.conv.csc']['data'] = float(reglog['regs']['top.conv.csc']['data']) / total
    seulog['top.conv.csc']['ctrl'] = float(reglog['regs']['top.conv.csc']['ctrl']) / total

    seulog['top.conv.dbuf'] = {}
    seulog['top.conv.dbuf']['data'] = float(reglog['regs']['top.conv.dbuf']['data']) / total
    seulog['top.conv.dbuf']['ctrl'] = float(reglog['regs']['top.conv.dbuf']['ctrl']) / total

    seulog['top.conv.dl'] = {}
    seulog['top.conv.dl']['data'] = float(reglog['regs']['top.conv.dl']['data']) / total
    seulog['top.conv.dl']['ctrl'] = float(reglog['regs']['top.conv.dl']['ctrl']) / total

    seulog['top.conv.wl'] = {}
    seulog['top.conv.wl']['data'] = float(reglog['regs']['top.conv.wl']['data']) / total
    seulog['top.conv.wl']['ctrl'] = float(reglog['regs']['top.conv.wl']['ctrl']) / total

    seulog['top.sdp'] = {}
    seulog['top.sdp']['data'] = float(reglog['regs']['top.sdp']['data']) / total
    seulog['top.sdp']['ctrl'] = float(reglog['regs']['top.sdp']['ctrl']) / total

    if check:
        count = 0
        for key, regs in seulog.items():
            count += regs['data']
            count += regs['ctrl']
        
        if abs(count - 1) > threshold:
            raise ValueError(f'Only counting {count} of registers')

    return seulog


###################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True, help='NVDLA architecture config file')
    parser.add_argument('--workdir', type=str, default='.', help='Profiler working directory')
    parser.add_argument('--output', type=str, default='reglog.yaml', help='Profiler output .yaml file')
    parser.add_argument('--check',  action='store_true', help='Dump flatten .yaml tile log')
    parser.add_argument('--threshold',  type=float, default=1e-6, help='Tolerance applied during the register count check')

    args = parser.parse_args()

    reglog = getRegs(args.config, args.workdir)

    if args.check:
        reglog_check(reglog)

    reglog['seu'] = units_probability(reglog, args.check, args.threshold)

    with open(args.output, 'w') as f:
        yaml.dump(reglog, f, Dumper=YDumper)


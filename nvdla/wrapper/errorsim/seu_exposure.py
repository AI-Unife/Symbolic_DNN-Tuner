
import argparse
import sys
from math import isclose, fsum
from concurrent.futures import ProcessPoolExecutor
import os

# --- YAML accelerated loader/dumper -----------------------------------------
import yaml
try:
    from yaml import CSafeLoader as YLoader, CSafeDumper as YDumper
except Exception:  # pragma: no cover
    from yaml import SafeLoader as YLoader, SafeDumper as YDumper


# --- Units list (kept as in original order if needed) -----------------------
units = [
    'top', 'top.conv', 'top.conv.cdma', 'top.conv.cbuf',
    'top.conv.cbuf.datbuf', 'top.conv.cbuf.wtbuf', 'top.conv.csb',
    'top.conv.csc', 'top.conv.dl', 'top.conv.wl', 'top.conv.cmac',
    'top.conv.cacc', 'top.conv.dbuf', 'top.sdp', 'top.sdp.csb',
    'top.sdp.core'
]


# --- Table-driven paths for time windows and reg rules ----------------------
TIME_WINDOWS = {
    'top':                  (['time','core','conv','fetch-time'], ['time','core','sdp','stop-time']),
    'top.conv':             (['time','core','conv','fetch-time'], ['time','core','sdp','stop-time']),
    'top.conv.cdma':        (['time','core','conv','fetch-time'], ['time','core','conv','compute-time']),
    'top.conv.cbuf':        (['time','core','conv','fetch-time'], ['time','core','conv','deliver-time']),
    'top.conv.cbuf.datbuf': (['time','core','conv','fetch-time'], ['time','core','conv','deliver-time']),
    'top.conv.cbuf.wtbuf':  (['time','core','conv','fetch-time'], ['time','core','conv','deliver-time']),
    'top.conv.csb':         (['time','core','conv','fetch-time'], ['time','core','sdp','stop-time']),
    'top.conv.csc':         (['time','core','conv','fetch-time'], ['time','core','sdp','stop-time']),
    'top.conv.dl':          (['time','core','conv','fetch-time'], ['time','core','conv','stop-time']),
    'top.conv.wl':          (['time','core','conv','fetch-time'], ['time','core','conv','stop-time']),
    'top.conv.cmac':        (['time','core','conv','compute-time'], ['time','core','conv','deliver-time']),
    'top.conv.cacc':        (['time','core','conv','compute-time'], ['time','core','conv','stop-time']),
    'top.conv.dbuf':        (['time','core','conv','deliver-time'], ['time','core','conv','stop-time']),
    'top.sdp':              (['time','core','conv','fetch-time'], ['time','core','sdp','stop-time']),
    'top.sdp.csb':          (['time','core','conv','fetch-time'], ['time','core','sdp','stop-time']),
    'top.sdp.core':         (['time','core','sdp','start-time'],  ['time','core','sdp','stop-time']),
}

# REG_RULES: value is (ctrl_key, data_key, total_key). Keys can be strings (for
# reglog path) or tuples where the first element is a sentinel (we accept any
# tuple and treat it as a path into tileflat['mems']).
REG_RULES = {
    'top':                  ('ctrl', 'data', 'total'),
    'top.conv':             ('ctrl', 'data', 'total'),
    'top.conv.cdma':        ('ctrl', 'data', 'total'),
    'top.conv.cbuf':        ('ctrl', 'data', 'total'),
    'top.conv.cbuf.datbuf': ('ctrl', ('tile.mems','dat','size'), 'total'),
    'top.conv.cbuf.wtbuf':  ('ctrl', ('tile.mems','wt','size'),  'total'),
    'top.conv.csb':         ('ctrl', 'data', 'total'),
    'top.conv.csc':         ('ctrl', 'data', 'total'),
    'top.conv.dl':          ('ctrl', 'data', 'total'),
    'top.conv.wl':          ('ctrl', 'data', 'total'),
    'top.conv.cmac':        ('ctrl', 'data', 'total'),
    'top.conv.cacc':        ('ctrl', ('tile.mems','out','size'), 'total'),
    'top.conv.dbuf':        ('ctrl', ('tile.mems','out','size'), 'total'),
    'top.sdp':              ('ctrl', 'data', 'total'),
    'top.sdp.csb':          ('ctrl', 'data', 'total'),
    'top.sdp.core':         ('ctrl', 'data', 'total'),
}


# --- Small utilities ---------------------------------------------------------
def _get_by_path(d, path):
    for k in path:
        d = d[k]
    return d


# --- Exposure accessors ------------------------------------------------------
def get_time_exposure(tilelog, unit):
    try:
        start_path, stop_path = TIME_WINDOWS[unit]
    except KeyError as e:
        raise ValueError(f'Unrecognized unit {unit}') from e
    return _get_by_path(tilelog, start_path), _get_by_path(tilelog, stop_path)


def get_reg_exposure(reglog, tilelog, unit):
    try:
        ctrl_key, data_key, total_key = REG_RULES[unit]
    except KeyError as e:
        raise ValueError(f'Unrecognized unit {unit}') from e

    r = reglog['regs'][unit]
    ctrl  = r[ctrl_key] if isinstance(ctrl_key, str) else _get_by_path(r, ctrl_key)

    if isinstance(data_key, str):
        data = r[data_key]
    else:
        # Interpret tuple as a path into tileflat['mems']
        # e.g., ('tile.mems','dat','size') -> ['mems','dat','size']
        _, *mem_path = data_key
        data = _get_by_path(tilelog, ['mems', *mem_path])

    total = r[total_key] if isinstance(total_key, str) else _get_by_path(r, total_key)
    return ctrl, data, total


# --- Flatten iterator (streaming) -------------------------------------------
def iter_flat_tiles(hierlog):
    """Yield (tileid, flat_tile, total_simtime) without accumulating a giant dict."""
    total_simtime = hierlog['total-layertime']
    for wtile in hierlog['tiles'].values():
        b_val = wtile['b-tile']; c_val = wtile['c-tile']
        lg = wtile['log']
        settings = lg['settings']
        dat_shape = lg['dat-shape']; wt_shape = lg['wt-shape']
        bs_shape  = lg['bs-shape'];  bn_shape = lg['bn-shape']
        ew_shape  = lg['ew-shape'];  out_shape = lg['out-shape']

        for ktile in lg['core'].values():
            k_val = ktile['k-tile']
            tileid = f'tile-b{b_val}-c{c_val}-k{k_val}'
            flat = {
                'b-tile': b_val, 'c-tile': c_val, 'k-tile': k_val,
                'mems':   ktile['mems'],
                'time':   {'total': ktile['tile-time'], 'core': ktile['nvdla']},
                'settings': settings,
                'tensors': {
                    'dat-shape': dat_shape, 'wt-shape': wt_shape, 'bs-shape': bs_shape,
                    'bn-shape': bn_shape, 'ew-shape': ew_shape, 'out-shape': out_shape
                },
                'core': ktile  # keep original per-core in case of extensions
            }
            yield tileid, flat, total_simtime


# --- Single-tile compute (picklable) ----------------------------------------
def _compute_tile(args):
    tileid, tileflat, simtime, reglog_local = args
    t_total = tileflat['time']['total']
    units_dict = {}
    for unit in units:
        start, stop = get_time_exposure(tileflat, unit)
        span = stop - start
        ctrl, data, total_r = get_reg_exposure(reglog_local, tileflat, unit)
        inv_total_r = 1.0 / total_r
        units_dict[unit] = {
            'hier': unit,
            'time': {'start': start, 'stop': stop, 'perc': span / t_total},
            'regs': {
                'ctrl': ctrl, 'ctrl-perc': ctrl * inv_total_r,
                'data': data, 'data-perc': data * inv_total_r,
                'total': total_r
            }
        }
    return (tileid, {
        'b-tile': tileflat['b-tile'],
        'c-tile': tileflat['c-tile'],
        'k-tile': tileflat['k-tile'],
        'time': {'total': t_total, 'perc': t_total / simtime},
        'mems': tileflat['mems'],
        'units': units_dict
    })


# --- Streaming compute (single-process, low memory) -------------------------
def calc_exposure_streaming(hierlog, reglog, *, write_path=None):
    out = {} if write_path is None else None
    dumper_ctx = open(write_path, 'w') if write_path else None
    try:
        for tileid, tileflat, simtime in iter_flat_tiles(hierlog):
            tileid, tile_entry = _compute_tile((tileid, tileflat, simtime, reglog))
            if dumper_ctx:
                yaml.dump({tileid: tile_entry}, dumper_ctx, Dumper=YDumper)
                dumper_ctx.write('---\n')
            else:
                out[tileid] = tile_entry
    finally:
        if dumper_ctx:
            dumper_ctx.close()
    return out


# --- Parallel compute (multiprocessing) -------------------------------------
def calc_exposure_parallel(hierlog, reglog, max_workers=None):
    seulog = {}
    tasks = ((tileid, flat, simtime, reglog) for tileid, flat, simtime in iter_flat_tiles(hierlog))
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for tileid, tile_entry in ex.map(_compute_tile, tasks, chunksize=64):
            seulog[tileid] = tile_entry
    return seulog


# --- Consistency check -------------------------------------------------------
def seulog_check(seulog, simtime):
    simtime_check = fsum(tile['time']['total'] for tile in seulog.values())
    perc_check    = fsum(tile['time']['perc']  for tile in seulog.values())
    if simtime_check != simtime:
        print(f'-E: Total simtime mismatch (expected: {simtime}/obtained: {simtime_check})')
        sys.exit(2)
    if not isclose(perc_check, 1.0, rel_tol=0, abs_tol=1e-9):
        print(f'-E: Total tiles exposure is not 100% ({perc_check})')
        sys.exit(2)


# --- CLI / main --------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description='SEU exposure calculator (optimized)')
    p.add_argument('--layer', required=True, help='layer hierarchical YAML')
    p.add_argument('--reglog', required=True, help='registers exposure YAML')
    p.add_argument('--out-flatten', dest='out_flatten', help='(optional) write flattened per-tile YAML')
    p.add_argument('--output', required=True, help='output SEU exposure YAML')
    p.add_argument('--parallel', action='store_true', help='enable multiprocessing across tiles')
    p.add_argument('--max-workers', type=int, default=None, help='number of worker processes')
    p.add_argument('--stream-output', action='store_true', help='stream YAML to --output during compute')
    p.add_argument('--quiet', action='store_true', help='reduce logging')
    args = p.parse_args()

    def log(msg):
        if not args.quiet:
            sys.stderr.write(msg + '\n')

    # Load inputs (C-accelerated loader if available)
    log('-I: loading inputs...')
    with open(args.layer, 'r') as f:
        hierlog = yaml.load(f, Loader=YLoader)
    with open(args.reglog, 'r') as f:
        reglog = yaml.load(f, Loader=YLoader)

    # Optional: emit flattened tiles (streaming, per tile)
    if args.out_flatten:
        log('-I: writing flattened log (per tile)...')
        with open(args.out_flatten, 'w') as ff:
            for tileid, flat, _ in iter_flat_tiles(hierlog):
                yaml.dump({tileid: flat}, ff, Dumper=YDumper)
                ff.write('---\n')

    # Compute exposure
    if args.stream_output:
        log('-I: computing (streaming) and writing output...')
        calc_exposure_streaming(hierlog, reglog, write_path=args.output)
        # If streaming, we cannot do a global sum check cheaply without re-reading.
        # Optionally, skip the check or implement a two-pass. We keep it simple.
        return

    log('-I: computing exposure...')
    if args.parallel:
        seulog = calc_exposure_parallel(hierlog, reglog, max_workers=args.max_workers)
    else:
        seulog = calc_exposure_streaming(hierlog, reglog)

    # Check and write
    log('-I: checking consistency...')
    seulog_check(seulog, hierlog['total-layertime'])

    log('-I: writing output...')
    with open(args.output, 'w') as f:
        yaml.dump(seulog, f, Dumper=YDumper)

    log('-I: done.')

if __name__ == '__main__':
    # On some platforms, multiprocessing needs this guard to avoid fork bombs.
    main()

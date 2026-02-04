
import os
import csv
import numpy as np
import argparse

##########################################################
## Merge Routines

def merge_tensors(tensorlist, output):

    for it in range(len(tensorlist)):
        tensor = np.load(tensorlist[it])

        if (it == 0):
            res = tensor
        else:
            res = np.concatenate((res, tensor), axis=0)
    
    np.save(output, res)
    

def merge_logs(loglist, output):

    with open(output, 'w') as fd:

        for it in range(len(loglist)):
            with open(loglist[it], 'r') as fs:
                fd.writelines(fs)


def filter_results(goldenfile, errorfile, logfile, threshold):

    # Reads    
    golden = np.load(goldenfile)
    errors = np.load(errorfile)

    with open(logfile, 'r') as f:
        csvreader = csv.reader(f)
        data = [tuple(row) for row in csvreader] 

    # Checks
    if (golden.ndim +1) != (errors.ndim):
        raise ValueError(f'Mismatching ndims G = {golden.ndim}, E = {errors.ndim}')

    if golden.ndim == 2:
        E, B, C = errors.shape
        B_, C_  = golden.shape

        if((B_ != B) or (C_ != C)):
            raise ValueError('Mismatching shapes')

    elif golden.ndim == 4:
        E, B, C, H, W  = errors.shape
        B_, C_, H_, W_ = golden.shape

        if((B_ != B) or (C_ != C) or (H_ != H) or (W_ != W)):
            raise ValueError('Mismatching shapes')
    
    if(len(data) != E):
        raise ValueError(f'Erroneous error count (expected {E}, obtained {len(data)})')

    # Filtering
    cnt = 0
    arrays = []

    for it in range(len(data)):
        location, time, model, _, outcome = data[it]

        if(outcome == 'Masked/Silent'):
            if((golden.shape == errors[it].shape) and np.allclose(golden, errors[it], rtol=threshold)):
                data[it] = (location, time, model, -1, 'Masked')
            else:
                data[it] = (location, time, model, cnt, 'Silent')
                arrays.append(errors[it])
                cnt += 1
        else:
            data[it] = (location, time, model, -1, outcome)

    if (len(arrays) > 0):
        outs = np.stack(arrays, axis=0)
    else:
        outs = None

    # Save
    with open(logfile, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data)
    
    if outs is None:
        with open(errorfile, 'w') as f:
            pass
    else:
        np.save(errorfile, outs)


##########################################################
## Main

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--inputnpy',  type=str, nargs='+', required=True, help='Input numpy files')
    parser.add_argument('--inputlog',  type=str, nargs='+', required=True, help='Input csv files')
    parser.add_argument('--outnpy',    type=str, default='./results.npy',  help='Output numpy file')
    parser.add_argument('--outlog',    type=str, default='./results.csv',  help='Output csv file')
    parser.add_argument('--cleanup',   action='store_true',                help='Remove input files after merging')
    parser.add_argument('--filter',    action='store_true',                help='Remove not corrupted tensors')
    parser.add_argument('--golden',    type=str,                           help='Golden out for filtering')
    parser.add_argument('--threshold', type=float, default=1e-3,           help='Relative threshold for error check')

    args = parser.parse_args()

    merge_logs(args.inputlog, args.outlog)
    merge_tensors(args.inputnpy, args.outnpy)

    if args.filter:
        if args.golden is None or (args.golden == ''):
            raise ValueError('--golden option is required for filtering')
        filter_results(args.golden, args.outnpy, args.outlog, args.threshold)

    if args.cleanup:
        for file in args.inputnpy:
            os.remove(file)

        for file in args.inputlog:
            os.remove(file)
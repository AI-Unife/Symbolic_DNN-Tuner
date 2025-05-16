from nas_201_api import NASBench201API as API
api = API('$path_to_meta_nas_bench_file')
# Create an API without the verbose log
api = API('NAS-Bench-201-v1_1-096897.pth', verbose=False)
# The default path for benchmark file is '{:}/{:}'.format(os.environ['TORCH_HOME'], 'NAS-Bench-201-v1_1-096897.pth')
api = API(None)
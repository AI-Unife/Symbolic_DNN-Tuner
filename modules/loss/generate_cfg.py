import yaml
from pytorch_implementation.module_backend import isConv2d

def generate_minimal_cfg(self, model):
    """
    Genera una configurazione hardware minimale basata sulle caratteristiche del modello."""
    max_c = 0
    max_k = 0

    # Allineamento multipli di 16
    def align(x, base=16):
        return int(((x + base - 1) // base) * base)

    for layer in model.modules():
        if isConv2d(layer):
            C = layer.in_channels
            K = layer.out_channels

            max_c = max(max_c, C)
            max_k = max(max_k, K)

    atomic_c = max(16, align(max_c))
    atomic_k = max(16, align(max_k))

    dtype = "int8"
    bytes_per_elem = 1  # int8

    max_dat_buf = 0
    max_wt_buf = 0

    for layer in model.modules():
        if isConv2d(layer):

            W, H = layer.kernel_size
            C = layer.in_channels
            K = layer.out_channels

            dat_buff_logico = (W * H * C) + (W * H * C * min(atomic_k, K))
            wt_buff_logico  = (W * H * K) // atomic_k

            max_dat_buf = max(max_dat_buf, dat_buff_logico)
            max_wt_buf  = max(max_wt_buf, wt_buff_logico)

    datmem_size = max(max_dat_buf * bytes_per_elem, 1024 * 1024)
    wtmem_size  = max(max_wt_buf * bytes_per_elem, 1024 * 1024)

    outmem_size = max(wtmem_size, 1024 * 1024)

    sdpmem_size = max(datmem_size // 4, 1024 * 1024)

    bank_depth_dat = max(max_dat_buf // atomic_c, 1024)
    bank_depth_wt  = max(max_wt_buf // atomic_c, 1024)
    bank_depth_out = max(max_wt_buf // atomic_k, 1024)

    config_name = f"minimal_nv_{atomic_c}x{atomic_k}_b1_{dtype}.yaml"

    minimal_config = {
        'name': config_name,

        'axi-dbb': {
            'latency': 1200,
            'wordsize': 512
        },

        'cmac': {
            'atomic-c': atomic_c,
            'atomic-k': atomic_k,
            'batch-size': 1,
            'mac-pipelined': False
        },

        'cbuf': {
            'datbuf': {
                'num-banks': 1,
                'bank-depth': int(bank_depth_dat),
                'wordsize': 0
            },
            'wtbuf': {
                'num-banks': 1,
                'bank-depth': int(bank_depth_wt),
                'wordsize': 0
            }
        },

        'cacc': {
            'num-banks': 1,
            'bank-depth': int(bank_depth_out),
            'bitwidth': 31,
            'wordsize': 0
        },

        'sdp': {
            'alu-pipelined': False,
            'num-stages': 1
        },

        'host-system': {
            'datmem-size': datmem_size,
            'outmem-size': outmem_size,
            'sdpmem-size': sdpmem_size,
            'wtmem-size': wtmem_size
        },

        'inject': {
            'ctrlpath': True,
            'datapath': True
        },

        'sim-timeout': 0,
        'truncate': True,

        'dat-type': dtype,
        'wt-type': dtype
    }

    minimal_config_path = self.specs_dir / config_name

    return minimal_config, minimal_config_path
    try:
        with open(minimal_config_path, 'w') as f:
            yaml.dump(minimal_config, f)

        print(f"[OK] Minimal hardware configuration saved to: {minimal_config_path}")

    except Exception as e:
        print(f"[FAIL] Error saving minimal configuration: {e}")
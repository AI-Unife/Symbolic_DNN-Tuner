import yaml
import math
from tensorflow.keras.layers import Conv2D
from pytorch_implementation.module_backend import isConv2d


def generate_minimal_cfg(self, model):
    """
    Generate a hardware-safe minimal configuration based on model structure.
    This version is EMBER-safe (prevents register/buffer overflow crashes).
    """

    def align(x, base=16):
        return int(math.ceil(x / base) * base)

    def safe_div(a, b):
        return max(1, math.ceil(a / b))

    max_c = 0
    max_k = 0

   
    # 1. EXTRACT MODEL LIMITS
   
    try:
        for layer in model.modules():
            if isConv2d(layer):
                max_c = max(max_c, layer.in_channels)
                max_k = max(max_k, layer.out_channels)

    except AttributeError:
        for layer in model.layers:
            if isinstance(layer, Conv2D):
                max_c = max(max_c, layer.input_shape[-1])
                max_k = max(max_k, layer.filters)

   
    # 2. HARDWARE ATOMIC UNITS
   
    atomic_c = min(align(max_c, 16), 128)
    atomic_k = min(align(max_k, 16), 128)
    

    batch_size = 1
    dtype = "int8"
    bytes_per_elem = 1

   
    # 3. BUFFER ESTIMATION (SAFE TILE MODEL)
   
    max_dat_buf = 0
    max_wt_buf = 0

    try:
        for layer in model.modules():
            if isConv2d(layer):

                C = layer.in_channels
                K = layer.out_channels
                W, H = layer.kernel_size

                tiles_c = math.ceil(C / atomic_c)
                tiles_k = math.ceil(K / atomic_k)

                # Feature map memory (input activations)
                dat_mem = C * H * W * batch_size * tiles_c

                # Weight memory (kernel blocks)
                wt_mem = C * K * W * H * tiles_k

                max_dat_buf = max(max_dat_buf, dat_mem)
                max_wt_buf = max(max_wt_buf, wt_mem)

    except AttributeError:
        for layer in model.layers:
            if isinstance(layer, Conv2D):

                C = layer.input_shape[-1]
                K = layer.filters
                W, H = layer.kernel_size

                tiles_c = math.ceil(C / atomic_c)
                tiles_k = math.ceil(K / atomic_k)

                dat_mem = C * H * W * batch_size * tiles_c
                wt_mem = C * K * W * H * tiles_k

                max_dat_buf = max(max_dat_buf, dat_mem)
                max_wt_buf = max(max_wt_buf, wt_mem)

   
    # 4. MEMORY SIZING (SAFE FLOOR)
   
    MIN_MEM = 1024 * 1024  # 1MB safety floor

    datmem_size = max(max_dat_buf * bytes_per_elem, MIN_MEM)
    wtmem_size  = max(max_wt_buf * bytes_per_elem, MIN_MEM)
    outmem_size = max(wtmem_size, MIN_MEM)
    sdpmem_size = max(datmem_size // 4, MIN_MEM)

   
    # 5. BANK DEPTH (SAFE)
   
    bank_depth_dat = max(safe_div(max_dat_buf, atomic_c), 1024)
    bank_depth_wt  = max(safe_div(max_wt_buf, atomic_c), 1024)
    bank_depth_out = max(safe_div(max_wt_buf, atomic_k), 1024)

   
    # 6. CONFIG NAME
   
    config_name = f"minimal_nv_{atomic_c}x{atomic_k}_b{batch_size}_{dtype}.yaml"

   
    # 7. FINAL CONFIG
   
    minimal_config = {
        "name": config_name,

        "axi-dbb": {
            "latency": 1200,
            "wordsize": 512
        },

        "cmac": {
            "atomic-c": atomic_c,
            "atomic-k": atomic_k,
            "batch-size": batch_size,
            "mac-pipelined": False
        },

        "cbuf": {
            "datbuf": {
                "num-banks": 1,
                "bank-depth": int(bank_depth_dat),
                "wordsize": 0
            },
            "wtbuf": {
                "num-banks": 1,
                "bank-depth": int(bank_depth_wt),
                "wordsize": 0
            }
        },

        "cacc": {
            "num-banks": 1,
            "bank-depth": int(bank_depth_out),
            "bitwidth": 31,
            "wordsize": 0
        },

        "sdp": {
            "alu-pipelined": False,
            "num-stages": 1
        },

        "host-system": {
            "datmem-size": datmem_size,
            "outmem-size": outmem_size,
            "sdpmem-size": sdpmem_size,
            "wtmem-size": wtmem_size
        },

        "inject": {
            "ctrlpath": True,
            "datapath": True
        },

        "sim-timeout": 0,
        "truncate": True,

        "dat-type": dtype,
        "wt-type": dtype
    }

    minimal_config_path = self.specs_dir / config_name

    return minimal_config, minimal_config_path
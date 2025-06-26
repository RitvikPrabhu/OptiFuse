import json
import pathlib
import pynvml
import torch

pynvml.nvmlInit()

def get_device_info(idx=0):
    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
    name = pynvml.nvmlDeviceGetName(handle)
    major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
    sm_count = torch.cuda.get_device_properties(idx).multi_processor_count

    width_bits = pynvml.nvmlDeviceGetMemoryBusWidth(handle)
    clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
    bw_gbps = round(width_bits * clock_mhz * 2 / 1e3 / 8, 1)

    smem_kib = torch.cuda.get_device_properties(idx).shared_memory_per_multiprocessor // 1024
    reg_per_sm = torch.cuda.get_device_properties(idx).regs_per_multiprocessor

    return {
        "name": name,
        "cc": f"{major}{minor}",
        "sm": sm_count,
        "smem_kib": smem_kib,
        "reg_per_sm": reg_per_sm,
        "bw_gbps": bw_gbps,
    }


def _write_cache(info):
    cache_dir = pathlib.Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / f"device_{info['cc']}.json").write_text(json.dumps(info, indent=2))


# if __name__ == "__main__":
#     info = get_device_info()
#     print(json.dumps(info, indent=2))
#     _write_cache(info)

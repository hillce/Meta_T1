import torch
import re
import numpy as np

def get_device_metrics():
    deviceDict = {}
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        totMem = float(prop.total_memory)
        totMem /= (1024*1024)

        itm = torch.cuda.list_gpu_processes(i)
        itm = re.findall("\d+.\d+ MB",itm)
        itm = [float(num[:-3]) for num in itm]
        memUsage = sum(itm)
        perUsage = 100*memUsage/totMem

        deviceDict[i] = {"Memory Free (MB)":totMem-memUsage,"Total Memory (MB)":totMem,"Percentage Usage":perUsage}
    key = "Memory Free (MB)"
    suggestedDevice = np.argmax([deviceDict[k][key] for k in deviceDict.keys()])

    return deviceDict, suggestedDevice
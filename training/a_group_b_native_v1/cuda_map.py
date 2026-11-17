"""Map visible CUDA ordinals to driver UUID without creating contexts."""
import ctypes, json, os, re, subprocess, sys, uuid

_UUID_TEXT = re.compile(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$')

def parse_gpu_uuid(value):
    """Parse exactly one complete physical-GPU UUID, with optional GPU- prefix.

    PyTorch renders CUuuid without the NVML GPU- prefix. Compare all 128 bits;
    truncated UUIDs, other prefixes (including MIG), and non-strings are invalid.
    """
    if not isinstance(value, str):
        raise ValueError('GPU UUID 类型无效')
    text = value[4:] if value[:4].lower() == 'gpu-' else value
    if not _UUID_TEXT.fullmatch(text):
        raise ValueError('GPU UUID 格式无效')
    return uuid.UUID(text)

class UUID(ctypes.Structure):
    _fields_=[('data',ctypes.c_ubyte*16)]

def query():
    lib=ctypes.CDLL('libcuda.so.1')
    def checked(code):
        if code!=0:raise RuntimeError('CUDA driver call failed')
    checked(lib.cuInit(0));count=ctypes.c_int();checked(lib.cuDeviceGetCount(ctypes.byref(count)))
    fn=getattr(lib,'cuDeviceGetUuid_v2',None) or lib.cuDeviceGetUuid
    rows={}
    for ordinal in range(count.value):
        dev=ctypes.c_int();checked(lib.cuDeviceGet(ctypes.byref(dev),ordinal))
        obj=UUID();checked(fn(ctypes.byref(obj),dev))
        rows[str(ordinal)]='GPU-'+str(uuid.UUID(bytes=bytes(obj.data)))
    if len(set(rows.values()))!=len(rows):raise RuntimeError('CUDA UUID 非唯一')
    return rows

def verified(python, expected, base_env, runner=subprocess.run):
    # CUDA Driver APIs do not establish that inherited runtime restrictions
    # were respected. Never remove or reinterpret a pre-existing restriction.
    if 'CUDA_VISIBLE_DEVICES' in base_env:
        raise RuntimeError('已设置 CUDA_VISIBLE_DEVICES，无法安全确认卡映射')
    try:
        if not isinstance(expected, (list, tuple)) or not expected:
            raise ValueError('GPU UUID 列表无效')
        wanted = [parse_gpu_uuid(x) for x in expected]
        if len(set(wanted)) != len(wanted):
            raise ValueError('GPU UUID 重复')
    except (ValueError, TypeError) as exc:
        raise RuntimeError('选中卡的CUDA UUID复核不一致') from exc
    allowed_env=dict(base_env,CUDA_DEVICE_ORDER='PCI_BUS_ID')
    env=dict(allowed_env)
    cmd=[str(python),'-B',__file__,'--show']
    def run(which):
        out=runner(cmd,env=which,capture_output=True,text=True,timeout=30,check=True)
        return json.loads(out.stdout)
    allowed=run(allowed_env)
    try:
        available = {parse_gpu_uuid(v) for v in allowed.values()}
    except (AttributeError, ValueError, TypeError) as exc:
        raise RuntimeError('CUDA 映射变化') from exc
    if any(x not in available for x in wanted):
        raise RuntimeError('所选卡不在继承的CUDA可见范围')
    a,b=run(env),run(env)
    if a!=b or not isinstance(a,dict):raise RuntimeError('CUDA 映射变化')
    try:
        inverse={parse_gpu_uuid(v):int(k) for k,v in a.items()}
    except (ValueError, TypeError) as exc:
        raise RuntimeError('CUDA 映射变化') from exc
    if len(inverse)!=len(a):raise RuntimeError('CUDA 映射非唯一')
    try:result=[inverse[x] for x in wanted]
    except KeyError as e:raise RuntimeError('卡身份不在CUDA可见列表') from e
    if len(set(result))!=len(result):raise RuntimeError('CUDA 卡身份重复')
    # A fresh torch runtime must resolve the intended UUIDs in the precise
    # visible-ordinal environment that will be given to the vLLM process.
    selected_env=dict(env,CUDA_VISIBLE_DEVICES=','.join(map(str,result)))
    out=runner([str(python),'-B',__file__,'--selected'],env=selected_env,
               capture_output=True,text=True,timeout=60,check=True)
    selected=json.loads(out.stdout)
    try:
        if not isinstance(selected, list):
            raise ValueError('GPU UUID 列表无效')
        resolved = [parse_gpu_uuid(x) for x in selected]
    except (ValueError, TypeError) as exc:
        raise RuntimeError('选中卡的CUDA UUID复核不一致') from exc
    if resolved != wanted:
        raise RuntimeError('选中卡的CUDA UUID复核不一致')
    return result

if __name__=='__main__':
    if sys.argv[1:]==['--show']:
        print(json.dumps(query(),sort_keys=True))
    elif sys.argv[1:]==['--selected']:
        import torch
        result=[]
        for ordinal in range(torch.cuda.device_count()):
            ident=getattr(torch.cuda.get_device_properties(ordinal),'uuid',None)
            if ident is None:raise RuntimeError('运行时没有暴露 GPU UUID')
            result.append(str(ident))
        print(json.dumps(result))
    else:raise SystemExit(2)

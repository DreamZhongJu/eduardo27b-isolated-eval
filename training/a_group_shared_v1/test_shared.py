import importlib.util
from pathlib import Path
p = Path(__file__).with_name('pilot.py')
spec = importlib.util.spec_from_file_location('pilot', p)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
samples=[(67265,83),(75739,100),(7468,0),(8984,0),(68059,78),(62879,97),(54941,96),(75765,0)]
s = {i:(81920,free,util,True) for i,(free,util) in enumerate(samples)}
assert m.choose_roles(s,s) == [4,0,6,7]
assert m.choose_roles({0:s[0]}, {0:s[0]}) == []
assert m.role_eligible(s[7],72) and not m.role_eligible(s[2],40)
s2 = dict(s); s2[7] = (81920, 1000, 0, False)
assert m.choose_roles(s,s2)[-1] == 1
m.ACTIVE_GPUS[:] = [4,0,6,7]
m.gpu_snapshot = lambda: dict(s)
m.monitor_cards()
s[6] = (81920, 8191, 100, True)
try: m.monitor_cards()
except RuntimeError: pass
else: raise AssertionError('missing low-memory stop')
print('共享选卡、内存阈值、低内存停止检查通过')

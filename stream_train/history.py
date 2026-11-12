"""Conservatively hold every source in a historical shard that was started."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-root',action='append',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();held=set();shards=0
    for root in a.old_root:
        if not (root/'runs').is_dir() or not (root/'data'/'shards').is_dir():
            raise ValueError('old root must contain runs and data/shards')
        for run in sorted((root/'runs').glob('shard_[0-9][0-9][0-9]')):
            if not run.is_dir() or not any(run.iterdir()):continue
            shard=root/'data'/'shards'/(run.name+'.jsonl')
            if not shard.is_file():raise ValueError('missing source shard: '+run.name)
            for line in shard.read_text().splitlines():
                if line.strip():held.add(json.loads(line)['source_id'])
            shards+=1
    with a.out.open('x',encoding='utf-8') as f:
        for sid in sorted(held):f.write(sid+'\n')
    print(json.dumps({'held_sources':len(held),'started_shard_directories':shards}))

if __name__=='__main__':main()

"""Restore checksummed full run receipts for optional detailed report reproduction."""
import hashlib
import json
import tarfile
from pathlib import Path

PACKAGE=Path(__file__).resolve().parent
ROOT=PACKAGE.parents[1]
ALLOWED={'table3-reconstruction-20261118-olama-v3','table3-reconstruction-20261118-recovery-r1'}

def main():
    checks=json.loads((PACKAGE/'SHA256SUMS.json').read_text())
    for archive in sorted((PACKAGE/'archives').glob('*.tar.xz')):
        relative=str(archive.relative_to(PACKAGE))
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==checks[relative],relative
        with tarfile.open(archive,'r:xz') as tar:
            for member in tar:
                parts=Path(member.name).parts
                if not member.isfile() or len(parts)<3 or parts[0]!='results' or parts[1] not in ALLOWED or '..' in parts:
                    raise ValueError('Unexpected archive member: '+member.name)
                destination=ROOT/member.name
                if not destination.resolve().is_relative_to(ROOT/'results'):
                    raise ValueError('Destination escapes results')
                stream=tar.extractfile(member)
                data=stream.read()
                if destination.exists():
                    if hashlib.sha256(destination.read_bytes()).digest()!=hashlib.sha256(data).digest():
                        raise ValueError('Refusing to overwrite changed artifact: '+member.name)
                else:
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    destination.write_bytes(data)
        print('Verified and restored '+archive.name)

if __name__=='__main__':main()

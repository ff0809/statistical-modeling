import os,json,sys,hashlib,subprocess
from pathlib import Path
import requests
META=Path(__file__).resolve().parent
ROOT=META.parent
HEADERS={'Accept':'application/octet-stream','User-Agent':'ff0809-archive-restore'}
credential=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',capture_output=True,text=True,env={**os.environ,'GIT_TERMINAL_PROMPT':'0','GCM_INTERACTIVE':'Never'},timeout=30)
values=dict(x.split('=',1) for x in credential.stdout.splitlines() if '=' in x)
if values.get('password'):HEADERS['Authorization']='Bearer '+values['password']
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as data:
  for b in iter(lambda:data.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def safe(path):
 result=(ROOT/path).resolve()
 if ROOT not in result.parents:raise RuntimeError('Invalid archive path')
 return result
for record in json.loads((META/'assets.json').read_text(encoding='utf-8-sig')):
 target=safe(record['path'])
 if target.exists():
  if digest(target)==record['sha256']:continue
  raise RuntimeError('Existing file differs: '+record['path'])
 target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_name(target.name+'.restore-partial')
 with temporary.open('wb') as output:
  for part in record['parts']:
   h=hashlib.sha256();size=0
   with requests.get(part.get('api_url',part['url']),headers=HEADERS,stream=True,timeout=(30,180)) as response:
    response.raise_for_status()
    for b in response.iter_content(4*1024**2):h.update(b);size+=len(b);output.write(b)
   if size!=part['bytes'] or h.hexdigest()!=part['sha256']:raise RuntimeError('Release part mismatch')
 if temporary.stat().st_size!=record['bytes'] or digest(temporary)!=record['sha256']:raise RuntimeError('Full file mismatch')
 temporary.rename(target);print('Restored:',record['path'])
metadata=json.loads((META/'excluded.json').read_text(encoding='utf-8-sig'))
for index,bundle in enumerate(metadata['original_git_history'],1):
 target=META/('restored-history-'+str(index)+'.bundle')
 with target.open('wb') as output:
  for part in bundle['parts']:
   with (META/part).open('rb') as data:
    for b in iter(lambda:data.read(4*1024**2),b''):output.write(b)
 if digest(target)!=bundle['sha256']:raise RuntimeError('History bundle mismatch')
 subprocess.run(['git','bundle','verify',str(target)],cwd=str(ROOT),check=True)
for record in json.loads((META/'manifest.json').read_text(encoding='utf-8-sig')):
 if record.get('storage')=='external':
  original=Path(record['source_path'])
  if not original.is_file():print('External dataset location:',record['source_path']);continue
  if digest(original)!=record['sha256']:raise RuntimeError('External dataset mismatch')
  continue
 if digest(safe(record['path']))!=record['sha256']:raise RuntimeError('File mismatch: '+record['path'])
print('GitHub content and Git history verified. External datasets use the separate E-drive archive.')

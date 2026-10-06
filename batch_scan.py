import sys, glob, os, requests
folder, tag = sys.argv[1], sys.argv[2]
for f in sorted(glob.glob(os.path.join(folder, '*.mp4'))):
    with open(f, 'rb') as fh:
        d = requests.post('http://127.0.0.1:5050/api/scan', files={'file': fh}).json()
    print(tag, os.path.basename(f), d.get('label'), d.get('real_probability'), d.get('fake_probability'))

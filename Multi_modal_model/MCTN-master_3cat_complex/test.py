import os

while True:
    fname=input('fname>')
    if os.path.exists(fname):
        print("Error:'%s' already exists" % fname)
    else:
        break
fobj = open(fname, 'w')
fobj.close()

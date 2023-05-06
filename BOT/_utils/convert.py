with open('_tmp/tmp.py', 'r') as f:
    lines = [str(repr(line).replace('\\n', '#').replace('    ', '\\t')) for line in f.readlines()]
out = ''
for line in lines: out+=line[1:-1]
print(out)

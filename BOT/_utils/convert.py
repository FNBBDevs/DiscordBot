nl_sep = input('Newline seperator: ')
with open('_tmp/tmp.py', 'r', encoding='utf-8') as f:
    lines = [str(repr(line).replace('\\n', nl_sep).replace('    ', '\\t')) for line in f.readlines()]
out = ''
for line in lines: out+=line[1:-1]
print(out)

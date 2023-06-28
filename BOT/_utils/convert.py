nl_sep = input("Newline seperator: ")
tb_sep = input("tab sep: ")
with open("_tmp/tmp.py", "r", encoding="utf-8") as f:
    lines = [
        str(repr(line).replace("\\n", nl_sep).replace("    ", tb_sep))
        for line in f.readlines()
    ]
out = ""
for line in lines:
    out += line[1:-1]
print(out)
asdfasdfasdfasdfasdf
d

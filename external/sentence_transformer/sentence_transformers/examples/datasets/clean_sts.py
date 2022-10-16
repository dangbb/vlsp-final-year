from pathlib import Path


# filter out all csv files
working_dir = Path('/workspace/sentence-transformers/examples/datasets/stsbenchmark/').glob('*_vi.csv')

for f in working_dir:
    # read all lines to memory and use map for efficiency
    print(f"Working on {f.name}.")
    with open(f) as fi:
        data = fi.readlines()

    data = [line.split('\t') for line in data]
    ident_sents = [idx for idx, line in enumerate(data) if line[4] == '5.000' and line[5] == line[6]]
    print("line(s)", ident_sents)
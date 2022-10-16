import requests
import re
import os
from ast import literal_eval
from pathlib import Path
from typing import List, Callable
from tqdm.auto import tqdm
# from functools import reduce

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/translate-stsbenchmark-ed18aff9282f.json'

from google.cloud import translate_v2 as translate
translate_client = translate.Client(target_language="vi")

def gg_translate(text):
    return translate_client.translate(text, format_='text', source_language='en', model='nmt')['translatedText']

# class Translator(object):
#     def __init__(self):
#         super().__init__()

#         self.query = {
#             'client': 'gtx',
#             'sl': 'en',
#             'tl': 'vi',
#             'hl': 'vi',
#             'dt': ['at', 'bd', 'ex', 'ld', 'md', 'qca', 'rw', 'rm', 'ss', 't'],
#             'ie': 'UTF-8',
#             'oe': 'UTF-8',
#             'otf': 1,
#             'ssel': 0,
#             'tsel': 0,
#             'kc': 7
#         }

#     def translate(self, text):
#         result = ['']
#         try:
#             r = requests.post('https://translate.google.com/translate_a/single', params=self.query, data={'q': text})
#         except requests.RequestException as e:
#             print(e)

#         # replace all keywords that doesn't exist in python
        
#         try:
#             result = literal_eval(re.sub(r'Array|null|true|false', '0', r.text))
#         except SyntaxError as e:
#             print(e)
#             print(r.text)

#         # concat sentences
#         translated = ' '.join([sent[0] for sent in result[0]])

#         return translated


def translate_sentences_in_stsbenchmark_line(f: Path, line: str, translator: Callable[[str], str]) -> str:
    err_basket = []
    new_line = line.split('\t')
    try: 
        # in stsbenchmark, field 5 and 6 (0 indexed) are 2 sentences
        new_line[5] = translator(new_line[5].strip())
        new_line[6] = translator(new_line[6].strip())

        # sometimes, two sentences are identical after translating to Vietnamese so we change the similarity score
        if new_line[5] == new_line[6]:
            new_line[4] = '5.000'
        
        new_line = '\t'.join(new_line)
    except:
        err_basket.append(line)
    
    if len(err_basket):
        log_path = ''.join([f.stem, '.log'])
        with open(log_path, 'w') as fo:
            fo.write('\n'.join(err_basket))

    return new_line if type(new_line) == str else "@@ERR@@"

# filter out all csv files
working_dir = Path('/workspace/sentence-transformers/examples/datasets/stsbenchmark/').glob('*.csv')

# open each file and translate 2 sentences in each line to vietnamese
for f in tqdm(working_dir):
    # read all lines to memory and use map for efficiency
    print(f"Working on {f.name}.")
    with open(f) as fi:
        data = fi.readlines()

    print(f"Translating {len(data)} lines.")    
    data = [translate_sentences_in_stsbenchmark_line(f, line, gg_translate) for line in tqdm(data)]
    
    new_name = ''.join([f.stem, '_vi', f.suffix])

    print(f"|--> Saving to {new_name}.")
    f.with_name(new_name).write_text('\n'.join(data))

print("gimme a breakpoint :))")
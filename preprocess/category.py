import sys
import json
from collections import OrderedDict

name = 'book'
if len(sys.argv) > 1:
    name = sys.argv[1]

item_map = {}
item_cate = {}
item_title = {}

with open('./data/%s_data/%s_item_map.txt' % (name, name), 'r') as f:
    for line in f:
        conts = line.strip().split(',')
        item_map[conts[0]] = int(conts[1])

with open('./data/book_data/meta_Books.json', 'r') as f:
    for line in f:
        r = eval(line.strip())
        iid = r['asin']
        cates = r['category']
        title = r['title']
        if iid not in item_map:
            continue

        item_cate[item_map[iid]] = cates
        item_title[item_map[iid]] = title

new_item_title = sorted(item_title.items(),key=lambda x:x[0],reverse=False)
new_item_cate = sorted(item_cate.items(),key=lambda x:x[0],reverse=False)

with open('./data/%s_data/%s_item_title.txt' % (name, name), 'w') as f:
    for item in new_item_title:
        f.write('%s,%s\n' % (str(item[0]), item[1]))
with open('./data/%s_data/%s_item_cate.txt' % (name, name), 'w') as f:
    for item in new_item_cate:
        f.write('%s,%s\n' % (str(item[0]), item[1]))

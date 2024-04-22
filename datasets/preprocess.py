import argparse
import time
import csv
import pickle
import operator
import datetime
import os

dataset = 'F:\data\amz\merged_data.csv'
print("-- Starting @ %ss" % datetime.datetime.now())
with open(dataset, "r") as f:
    reader = csv.DictReader(f, delimiter=',')
    reviewer_id = {}
    rating_date = {}
    ctr = 0
    curid = -1
    curdate = None
    for data in reader:
        reviewerid =  data['reviewerID']
    
        if curdate and not curid == reviewerid:   
            date = ''
            date = curdate
            rating_date[curid] = date

        curid = reviewerid
        item = data['product_id'], int(data['rating'])
        curdate = ''
        curdate = data['date']
        if reviewerid in reviewer_id:
            reviewer_id[reviewerid] += [item]
        else:
            reviewer_id[reviewerid] = [item]
        ctr += 1
    date = ''
    date = curdate
    rating_date[curid] = date

print("-- Reading data @ %ss" % datetime.datetime.now())

# Filter out length 1 sessions
for s in list(reviewer_id):
    if len(reviewer_id[s]) <= 1:
        del reviewer_id[s]
        del rating_date[s]

# Count number of times each item appears
iid_counts = {}
for s in reviewer_id:
    seq = reviewer_id[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

length = len(reviewer_id)
for s in list(reviewer_id):
    curseq = reviewer_id[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
    if len(filseq) < 2:
        del reviewer_id[s]
        del rating_date[s]
    else:
        reviewer_id[s] = filseq

# Split out test set based on dates
dates = list(rating_date.items())
maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date
splitdate = 0
splitdate = int(maxdate) - 86400 * 7


print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
tra_sess = filter(lambda x: x[1] < splitdate, map(lambda x: (x[0], int(x[1])), dates))
tes_sess = filter(lambda x: x[1] > splitdate, map(lambda x: (x[0], int(x[1])), dates))

# Sort sessions by date
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
print(len(tra_sess))    # 186670    # 7966257
print(len(tes_sess))    # 15979     # 15324
print(tra_sess[:3])
print(tes_sess[:3])
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    train_ids = []
    train_seqs = []
    train_dates = []
    item_ctr = 1
    for s, date in tra_sess:
        rev = reviewer_id[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2: 
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
    print(item_ctr)    
    return train_ids, train_dates, train_seqs

# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in tes_sess:
        rev = reviewer_id[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
    return test_ids, test_dates, test_seqs

tra_ids, tra_dates, tra_seqs = obtian_tra()
tes_ids, tes_dates, tes_seqs = obtian_tes()

def process_seqs(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [id]
    return out_seqs, out_dates, labs, ids

tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
tra = (tr_seqs, tr_labs)
tes = (te_seqs, te_labs)
print(len(tr_seqs))
print(len(te_seqs))
print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])
print(te_seqs[:3], te_dates[:3], te_labs[:3])
all = 0

for seq in tra_seqs:
    all += len(seq)
for seq in tes_seqs:
    all += len(seq)
print('avg length: ', all/(len(tra_seqs) + len(tes_seqs) * 1.0))

if not os.path.exists('sample'):
    os.makedirs('sample')
pickle.dump(tra, open('sample/train.txt', 'wb'))
pickle.dump(tes, open('sample/test.txt', 'wb'))
pickle.dump(tra_seqs, open('sample/all_train_seq.txt', 'wb'))

print('Done.')
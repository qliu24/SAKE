import re
import numpy as np
import collections
from nltk.corpus import wordnet as wn
import pickle
import os

verbal = True
similarity_thrh = 0.2
wnid_pattern = re.compile("^n\d{8}_[0-9]+")
zero_dir = '../dataset/Sketchy/zeroshot1/'

filetmp = os.path.join(zero_dir, 'all_photo_filelist_train.txt')
with open(filetmp, 'r') as fh:
    file_content = fh.readlines()
    
wnid_dict=dict()
for ll in file_content:
    fname = ' '.join(ll.strip().split()[:-1])
    cid = int(ll.strip().split()[-1])
    cname,filename = fname.split('/')[-2:]
    assert(filename.split('.')[1] in ['JPEG','jpg'])
    
    if wnid_pattern.match(filename.split('.')[0]):
        wnid = filename.split('_')[0]
        if cname not in wnid_dict.keys():
            wnid_dict[cname]=wnid
        else:
            if not wnid in wnid_dict[cname]:
                # print('new wnid')
                # print(cname, wnid, wnid_dict[cname])
                wnid_dict[cname] = ','.join([wnid_dict[cname], wnid])
    else:
        if cname not in wnid_dict.keys():
            # print('new category with no wnid')
            # print(cname, filename)
            wnid_dict[cname]='???'
            
if verbal:
    print('no wnid found for:')
    for cname in wnid_dict.keys():
        if '???' in wnid_dict[cname]:
            print(cname)
    
    print('more than one wnid found for:')
    for cname in wnid_dict.keys():
        if ',' in wnid_dict[cname]:
            print(wnid_dict[cname])
            
synset_dict = dict()
for cname in wnid_dict.keys():
    if wnid_dict[cname]=='???':
        if cname == 'present':
            synset_dict[cname] = [wn.synsets(cname)[1]]
        elif cname == 'teddy-bear':
            synset_dict[cname] = [wn.synsets('teddy')[0]]
        elif cname == 'flying saucer':
            synset_dict[cname] = [wn.synsets('ufo')[0]]
        elif cname == 'potted plant':
            synset_dict[cname] = [wn.synsets('pot_plant')[0]]
        elif cname == 'santa claus':
            synset_dict[cname] = [wn.synsets('santa_claus')[0]]
        elif cname in ['person walking', 'person sitting']:
            synset_dict[cname] = [wn.synsets('person')[0]]
        elif cname == 'fire hydrant':
            synset_dict[cname] = [wn.synsets('fire_hydrant')[0]]
        elif cname == 'outlet':
            synset_dict[cname] = [wn.synsets('wall_socket')[0]]
        elif cname == 'flower with stem':
            synset_dict[cname] = [wn.synsets('flower')[0]]
        elif cname == 'human-skeleton':
            synset_dict[cname] = [wn.synsets('skeleton')[2]]
        elif len(wn.synsets(cname))==0:
            synset_dict[cname] = None
        else:
            synset_dict[cname] = [wn.synsets(cname)[0]]
    elif ',' in wnid_dict[cname]:
        synset_dict[cname] = []
        for wnid_i in wnid_dict[cname].split(','):
            synset_dict[cname].append(wn.synset_from_pos_and_offset('n', int(wnid_i[1:])))
    else:
        synset_dict[cname] = [wn.synset_from_pos_and_offset('n', int(wnid_dict[cname][1:]))]
        
        
# read in imagenet index and synsets
imagenet_dict_file = '../dataset/imagenet_label_to_wordnet_synset.txt'
with open(imagenet_dict_file, 'r') as fh:
    file_content = fh.readlines()
    
imagenet_dict = dict()
for li in range(0,len(file_content),3):
    wnid = file_content[li].strip().split('\'')[-2]
    imagenet_dict[li//3] = wn.synset_from_pos_and_offset('n', int(wnid.split('-')[0]))
    

# make correspondance matrix from sketchy categories to imagenet classes
wn_matrix = dict()
hypo = lambda s: s.hyponyms()
for cname in synset_dict.keys():
    wn_matrix[cname] = np.zeros(len(imagenet_dict))
    if synset_dict[cname] is None:
        continue
        
    for ss in synset_dict[cname]:
        for ik,iss in imagenet_dict.items():
            if ss == iss:
                wn_matrix[cname][ik] = 1
            elif iss in list(ss.closure(hypo)) or ss in list(iss.closure(hypo)):
#             elif iss in ss.hyponyms():
                wn_matrix[cname][ik] = 1
            elif ss.path_similarity(iss) > similarity_thrh:
                wn_matrix[cname][ik] = ss.path_similarity(iss)
                # wn_matrix[cname][ik] = 1
                

one_to_one=0
for cname in synset_dict.keys():
    if synset_dict[cname] is None:
        continue
        
    for ss in synset_dict[cname]:
        if ss in imagenet_dict.values():
            one_to_one += 1
            
print('number of exact matches:')
print(one_to_one)

hist_ls = []
for cname in list(wn_matrix.keys()):
    hist_ls.append(np.sum(wn_matrix[cname]))
    
cname_most = list(wn_matrix.keys())[np.where(np.array(hist_ls)==max(hist_ls))[0][0]]
print('the category that has the most corresponding classes is: {}, {}'.format(synset_dict[cname_most], max(hist_ls)))
iilist = np.where(wn_matrix[cname_most]==1)[0]
for ii in iilist:
    print(file_content[ii*3+1].strip())
    
cnter = collections.Counter(hist_ls)
total_c = 0
for ck,cv in cnter.items():
    total_c += int(cv)*int(ck)
    print('{} category corresponds to {} classes in imagenet'.format(int(cv),int(ck)))
    
print(total_c)

wn_matrix_np = np.zeros((len(wn_matrix),1000))
for ci,cname in enumerate(list(wn_matrix.keys())):
    wn_matrix_np[ci] = wn_matrix[cname]

print('Number of ImageNet classes that are not matched:')
print(np.sum(np.sum(wn_matrix_np, axis=0) == 0))


# mapping cid to 1000d vector of correspondance
cname_cid_file = os.path.join(zero_dir, 'cname_cid.txt')
with open(cname_cid_file, 'r') as fh:
    file_content = fh.readlines()
    
cname_cid_dict = dict()
for ff in file_content:
    cname = ' '.join(ff.strip().split()[0:-1])
    cid = int(ff.strip().split()[-1])
    cname_cid_dict[cname] = cid
    
cid_matrix = dict()
for cname in wn_matrix.keys():
    assert cname in cname_cid_dict.keys()
    cid_matrix[cname_cid_dict[cname]] = wn_matrix[cname]
    
to_save = os.path.join(zero_dir, 'cid_mask.pickle')
with open(to_save, 'wb') as fh:
    pickle.dump(cid_matrix, fh)

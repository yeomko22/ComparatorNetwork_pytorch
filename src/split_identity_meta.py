# 파일 개요 : VGGFace2 데이터 셋에 포함된 인물 목록은 identity_meta.csv 파일 안에 저장되어 있다.
#           이는 학습용 데이터와 테스트용 데이터가 분리되어 있지 않아서 불편한 측면이 있다.
#           때문에 이를 분리시키려 한다.

import csv

base_dir = '/usr/junny/VGGFace2/'
identity_meta_path = base_dir+'identity_meta.csv'
identity_meta_reader = csv.reader(open(identity_meta_path, 'r'))

identity_train_path = '../labels/identity_train.csv'
identity_train_writer = csv.writer(open(identity_train_path, 'w'))

identity_test_path = '../labels/identity_test.csv'
identity_test_writer = csv.writer(open(identity_test_path, 'w'))

train_path = base_dir+'train_list.txt'
train_file = open(train_path, 'r')

test_path = base_dir+'test_list.txt'
test_file = open(test_path, 'r')

train_id_set = set()
test_id_set = set()

for line in train_file :
    train_id_set.add(line.split('/')[0])

for line in test_file :
    test_id_set.add(line.split('/')[0])

for i, line in enumerate(identity_meta_reader) :
    if i==0 :
        identity_train_writer.writerow(line)
        identity_test_writer.writerow(line)
    else :
        if train_id_set.__contains__(line[0]) :
            identity_train_writer.writerow(line)
        else :
            identity_test_writer.writerow(line)

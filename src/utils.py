# 파일 개요 : Comparator Network 학습에 필요한 기타 함수들이 작성되어 있는 파일
# 참고 링크 : https://github.com/cydonia999/VGGFace2-pytorch

import csv
import argparse
import os

# 인물 아이디(n000002)를 신경망이 classification 하기에 용이한
# 정수 형태의 아이디로 변환하여 매핑한 딕셔너리로 생성하여 리턴하는 함수
# dataloader.py 에서 호출한다.
def get_id_label_map(meta_file):
    meta_reader = csv.reader(open(meta_file))
    label_count = 0
    label_dict = {}

    for i, row in enumerate(meta_reader):
        label_dict.update({row[0]:label_count})
        label_count+=1

    return label_dict

# 모델 학습 코드를 실행할 때 필요한 파라미터들을 입력받고 파싱해주는
# 파서를 생성 및 리턴하는 함수
def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        dest="input_dir",
                        default="none",
                        help="Directory path which contains VGGFace2 dataset train images")

    parser.add_argument('-t', '--test',
                        dest="test_dir",
                        default="none",
                        help="Directory path which contains VGGFace2 dataset test images")

    parser.add_argument('-a', '--inputA',
                        dest="input_imgA",
                        default="none",
                        help="imageA that conduct test")

    parser.add_argument('-b', '--inputB',
                        dest="input_imgB",
                        default="none",
                        help="imageB that conduct test")

    return parser.parse_args()

# 모델을 저장할 디렉터리가 없을 경우 생성
def checkpoint_create():
    if not os.path.exists('../checkpoint') :
        os.mkdir('../checkpoint')
# 파일 개요 : 파일 형태로 저장되어 있는 이미지들과 메타 데이터를 읽어온다.
#           그 다음, 학습에 적합한 형태로 전처리하여 신경망에 전달하는 역할을 한다.

# 학습 데이터 형식 : 구현하고자 하는 Comparator Network의 경우 template라 불리는 이미지 묶음을 한번에 2개 입력 받는다.
#                하나의 template는 동일 인물 이미지를 묶어서 구성한다.
#                두 template가 같은 인물에 대한 이미지면 positive 라벨을, 다른 인물이면 negative 라벨을 매겨준다.
#                이렇게 이미지 6장 (2개의 template)와 1개의 라벨 정보가 한 쌍을 이루게 된다.
#                이를 다시 batch size로 묶어서 신경망에 전달하게 된다.

import torch
from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import os
import utils
import pandas as pd
import time

class CustomDataset(Dataset) :
    def __init__(self, train_dir=None, test_dir=None):
        self.label_path = '../labels/'
        self.identity_train = self.label_path+'identity_train.csv'
        self.identity_test = self.label_path+'identity_test.csv'

        self.id_label_dict = utils.get_id_label_map(self.identity_train)

        if train_dir is not None :
            # 학습용 데이터 셋 경로 지정
            self.img_label = pd.read_csv(self.identity_train, delimiter=',')
            self.img_dir = train_dir
        else :
            # 테스트용 데이터 셋 경로 지정
            self.img_label = pd.read_csv(self.identity_test, delimiter=',')
            self.img_dir = test_dir

        # 모션 블러 적용 시에 필요한 필터 생성
        self.motion_filter_size = 15
        self.motion_blur_filter = np.zeros((self.motion_filter_size, self.motion_filter_size))
        self.motion_blur_filter[int((self.motion_filter_size - 1) / 2), :] = np.ones(self.motion_filter_size)
        self.motion_blur_filter = self.motion_blur_filter / self.motion_filter_size


    # 데이터 로더가 데이터를 읽어올 때 호출되는 함수
    # 한 쌍의 템플릿 이미지와 라벨을 묶어서 반환한다.
    # 템플릿을 만들 때에는 데이터 어그멘테이션을 거친다.
    def __getitem__(self, index):
        # positive 템플릿을 생성할 것인지, negative 템플릿을 생성할 지 결정한다.
        # 0 이면 서로 다른 인물, 1 이면 동일한 인물로 두 쌍의 템플릿을 구성한다.
        label = random.randint(0,1)
        # label = 1

        template1 = set()
        template2 = set()

        identity_one = self.img_label.iloc[index, 0]
        identity_two = ''

        # 서로 다른 인물로 구성된 템플릿 생성
        if label == 0:
            # 현재 인물과 다른 인물을 랜덤하게 선택
            # 추후에 여기 부분에 hard sampling 부분을 추가할 것
            while True :
                other_identity_index = random.randint(0, len(self.img_label)-1)
                if other_identity_index != index :
                    break
            identity_two = self.img_label.iloc[other_identity_index, 0]

        # 동일인으로 구성된 템플릿 생성
        else :
            identity_two = identity_one

        # 현재 인물의 클래스 값을 가져온다.
        class1 = self.id_label_dict.get(identity_one)
        class2 = self.id_label_dict.get(identity_two)

        # 현재 인물과 다른 인물의 이미지가 저장된 폴더를 설정한 뒤, 이미지 목록을 가져온다.
        cur_img_dir = self.img_dir+identity_one+'/'
        cur_img_list = os.listdir(cur_img_dir)
        other_img_dir = self.img_dir+identity_two+'/'
        other_img_list = os.listdir(other_img_dir)

        # identity1 인물의 이미지 3장을 읽어와 텐서형식으로 변환한 다음, template1 안에 추가
        while len(template1) < 3 :
            # 이미지를 읽어와 데이터 어그멘테이션 적용
            cur_img_path = cur_img_dir+cur_img_list[random.randint(1, len(cur_img_list)-1)]
            cur_img = cv2.imread(cur_img_path)
            cur_img = self.transform_img(cur_img)

            # 이미지를 텐서 형식으로 변한한 뒤 템플릿에 저장
            cur_img = cur_img.transpose((2, 0, 1))
            cur_img = torch.from_numpy(np.flip(cur_img, axis=0).copy()).float()
            template1.add(cur_img)

        # identity2 인물의 이미지 3장을 읽어와 텐서형식으로 변환한 다음, template2 안에 추가
        while len(template2) < 3 :
            # 이미지를 읽어와 데이터 어그멘테이션 적용
            cur_img_path=other_img_dir + other_img_list[random.randint(1, len(other_img_list) - 1)]
            cur_img = cv2.imread(cur_img_path)
            cur_img = self.transform_img(cur_img)

            # 이미지를 텐서 형식으로 변한한 뒤 템플릿에 저장
            cur_img = cur_img.transpose((2, 0, 1))
            cur_img = torch.from_numpy(np.flip(cur_img, axis=0).copy()).float()

            template2.add(cur_img)

        template1 = list(template1)
        template2 = list(template2)

        # 라벨 값을 텐서 형식으로 변환해준다.
        label = torch.LongTensor(np.array([label], dtype=np.int64))

        # 템플릿 두 개, 각 템플릿 별 클래스, 전체 라벨을 하나의 샘플로 묶어서 리턴한다.
        sample = {"template1": template1, "template2": template2, "class1": class1, "class2": class2,
                      "label": label}
        return sample

    def __len__(self):
        return len(self.img_label)

    # 이미지 전처리 함수
    # 1. 이미지의 높이, 너비 중 짧은 쪽을 144 크기로 변경하며, 나머지는 가운데를 중심으로 크롭한다.
    # 2. 각각의 이미지들은 20 % 확률로 좌우 반전 가우시안 블러, 모션 블러, 흑백 변환을 거친다.
    def transform_img(self, cur_img):
        cur_img = self.img_resize(cur_img)

        # 20% 확률로 좌우 반전 적용
        if random.randint(1, 10) < 3:
            cur_img = cur_img[:, ::-1]

        # 20% 확률로 가우시안 블러 적용
        if random.randint(1, 10) < 3:
            cur_img = cv2.GaussianBlur(cur_img, (5, 5), 0)

        # 20% 확률로 모션 블러 적용
        if random.randint(1, 10) < 3:
            cur_img = cv2.filter2D(cur_img, -1, self.motion_blur_filter)

        # 20% 확률로 흑백 변환 적용
        if random.randint(1, 10) < 3:
            cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
            cur_img = np.stack((cur_img,)*3, -1)

        # 이미지의 각 체널에서 127.5를 빼주며, 0보다 작은 값은 0으로 치환해준다.
        subtract_value = np.full((144, 144, 3), 127.5)
        cur_img = (cur_img - subtract_value).clip(min=0)

        return cur_img

    # 입력 이미지의 크기를 144x144로 맞춰주는 함수
    def img_resize(self, cur_img):
        height, width = cur_img.shape[:2]

        # 너비가 높이보다 크면 너비를 144에 맞춰준다.
        if width > height:
            transform_ratio = 144 / height
            new_width = int(width * transform_ratio)
            resized_img = cv2.resize(cur_img, (new_width, 144))

            # 그 다음 높이를 가운데를 기준으로 크롭하여 144 크기로 맞춰준다.
            if new_width != 144:
                crop_size = int((new_width - 144) / 2)

                if new_width % 2 == 0:
                    resized_img = resized_img[0:144, crop_size:new_width - crop_size]
                else:
                    resized_img = resized_img[0:144, crop_size:new_width - crop_size - 1]


        # 높이가 너비보다 크면 너비를 144에 맞춰준다.
        else:
            transform_ratio = 144 / width
            new_height = int(height * transform_ratio)
            resized_img = cv2.resize(cur_img, (144, new_height))

            # 그 다음 높이를 가운데를 기준으로 크롭하여 144 크기로 맞춰준다.
            if new_height != 144:
                crop_size = int((new_height - 144) / 2)

                if new_height % 2 == 0:
                    resized_img = resized_img[crop_size:new_height - crop_size, 0:144]
                else:
                    resized_img = resized_img[crop_size:new_height - crop_size - 1, 0:144]

        return resized_img

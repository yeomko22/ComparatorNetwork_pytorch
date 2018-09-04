# 파일 개요 : ComparatorNetwork을 구성하는 신경망 모델이 담겨있는 파일
#           Detector, Attender, Comparator 클래스,
#           이들을 조합한 ComparatorNetwork 클래스가 작성되어 있다.

import torch.nn as nn
import torch
from torch.autograd import Variable

# ComparatorNetwork 모델
# Detector, Attender, Comparator 세 부분이 합쳐진 형태로 구성되어 있다.
class ComparatorNetwork(nn.Module):
    def __init__(self, batch_size=None, K=None):
        super(ComparatorNetwork, self).__init__()
        self.K = K
        self.batch_size = batch_size
        self.detector = Detector(Bottleneck, [3, 4, 1], K=self.K, batch_size=self.batch_size, regularization='diversity')
        self.attender = Attender(K=self.K, batch_size=self.batch_size)
        self.comparator = Comparator(K=self.K, batch_size=self.batch_size)

    def detect(self, template1, template2, class1, class2, label):
        return self.detector(template1, template2, class1, class2, label)

    def attend(self, local_landmarks, global_map):
        return self.attender(local_landmarks, global_map)

    def compare(self, temp1_attended_vector, temp2_attended_vector):
        return self.comparator(temp1_attended_vector, temp2_attended_vector)

# Detector
# 기본적으로 ResNet 50의 구조를 따른다.
# 차이점은 마지막 FC 레이어의 크기를 identity 수에 맞게 8631로 설정해준 부분과
# 미리 설정한 K개 만큼 로컬 스코어 맵을 추출하는 부분,
# 로컬 스코어 맵들을 max projection으로 합치는 부분,
# 마지막 레이어 이전 feature map을 리턴하는 부분이다.
class Detector(nn.Module):
    def __init__(self, block, layers, num_classes=8651, K=None, batch_size=None, regularization=None):
        super(Detector, self).__init__()
        self.K=K
        self.batch_size=batch_size
        self.regularization = regularization

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residaul 레이어들
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)

        # 클래시피케이션을 위한 avgpool과 fc
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(9216, num_classes)

        # K 개 로컬 피쳐맵을 뽑아내기 위한 1x1 컨볼루션
        self.conv_1x1_K = nn.Conv2d(1024, self.K, kernel_size=1, stride=1, padding=0, bias=False)

        # Diversity Regularizer 필요 요소들
        self.softmax = nn.Softmax2d()
        self.criterion = nn.CrossEntropyLoss()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, template1, template2, class1, class2, label):
        # 각 템플릿 별로 안에 묶여있는 이미지들을 각각 identity classify를 진행한다.
        # 이를 통해 1개의 이미지에서 1개의 global feature, K+1 local feature maps, classify_output을 추출한다.
        temp1_classify_outputs = []
        temp1_local_landmarks = []
        temp1_K_local_maps = []
        temp1_global_maps=[]

        for img_tensor in template1:
            img_tensor = Variable(img_tensor).cuda()
            global_map, classify_output, local_landmarks, K_local_maps  = self.identity_classify(img_tensor)
            temp1_classify_outputs.append(classify_output)
            temp1_local_landmarks.append(local_landmarks)
            temp1_K_local_maps.append(K_local_maps)
            temp1_global_maps.append(global_map)

        # 이미지 3장의 결과 행렬의 평균을 내어 최종 클래시피케이션 결과를 구한다.
        # 이를 클레스 라벨과 비교하여 로스를 구해준다.
        temp1_classify_avg = (temp1_classify_outputs[0]+temp1_classify_outputs[1]
                             +temp1_classify_outputs[2])/3

        loss_cls1 = self.criterion(temp1_classify_avg, class1)

        # 템플릿 2에 대해서도 같은 작업을 반복해준다.
        temp2_classify_outputs = []
        temp2_local_landmarks = []
        temp2_K_local_maps = []
        temp2_global_maps = []
        for img_tensor in template2:
            img_tensor = Variable(img_tensor).cuda()
            global_map, classify_output, local_landmarks, K_local_maps = self.identity_classify(img_tensor)
            temp2_classify_outputs.append(classify_output)
            temp2_local_landmarks.append(local_landmarks)
            temp2_K_local_maps.append(K_local_maps)
            temp2_global_maps.append(global_map)

        temp2_classify_avg = (temp2_classify_outputs[0] + temp2_classify_outputs[1] + temp2_classify_outputs[2]) / 3
        loss_cls2 = self.criterion(temp2_classify_avg, class2)


        # -------------landmark regulization------------- #
        # 논문 상에는 정규화 로스 값이 0 ~ 1 사이로 기재가 되어 있었는데
        # 현재 구현 상으로는 -300에 가까운 숫자가 리턴되는 현상 발생
        # 문제는 파악중이며 현재 모델 학습에는 정규화를 하지 않는다.

        loss_reg = 0
        if self.regularization=='diversity' :
            max_projection1_array = []
            max_projection2_array = []

            for n in range(3) :
                # 템플릿 1 정규화 로스 계산
                # softmax normalization
                cur_K_local_maps = temp1_K_local_maps[n]
                cur_K_local_maps = self.softmax(cur_K_local_maps)

                # max projection
                cur_max_projection = torch.max(cur_K_local_maps, 1, False)[0]
                cur_max_projection = torch.unsqueeze(cur_max_projection, dim=1)
                max_projection1_array.append(cur_max_projection)

                # 템플릿 2 정규화 로스 계산
                # softmax normalization
                cur_K_local_maps = temp2_K_local_maps[n]
                cur_K_local_maps = self.softmax(cur_K_local_maps)

                # max projection
                cur_max_projection = torch.max(cur_K_local_maps, 1, False)[0]
                cur_max_projection = torch.unsqueeze(cur_max_projection, dim=1)
                max_projection2_array.append(cur_max_projection)

            merged_max_map1 = torch.stack(max_projection1_array)
            merged_max_map2 = torch.stack(max_projection2_array)

            final_max_projcetion1 = torch.max(merged_max_map1, 0, False)[0]
            final_max_projcetion2 = torch.max(merged_max_map2, 0, False)[0]

            sum1 = ((final_max_projcetion1.sum(1)).sum(1)).sum(1).sum(0)
            sum2 = ((final_max_projcetion1.sum(1)).sum(1)).sum(1).sum(0)

            loss_reg1 = self.batch_size * 3 * self.K - sum1
            loss_reg2 = self.batch_size * 3 * self.K - sum2

            # 여기까지 계산한 정규화 로그 값의 범위가 비정상적으로 크므로
            # 모델 학습에는 적용하지 않는다.
            # loss_reg = loss_reg1 + loss_reg2

        # Key point regulization 추가될 영역
        else :
            loss_reg=0

        return temp1_local_landmarks, temp2_local_landmarks, temp1_global_maps, temp2_global_maps, loss_cls1, loss_cls2, loss_reg


    def identity_classify(self, x):
        # input_size : [batch_size, 3, 144, 144]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # output_size : [batch_size, 64, 72, 72]

        x = self.maxpool(x)
        x = self.layer1(x)
        # output_size : [batch_size, 256, 36, 36]

        x = self.layer2(x)
        # output_size : [batch_size, 512, 18, 18]

        # Attender로 넘길 피쳐맵
        global_map = self.layer3(x)
        # output_size : [batch_size, 1024, 18, 18]

        # K개 로컬 피쳐 맵 추출
        K_local_maps = self.conv_1x1_K(global_map)
        # output_size : [batch_size, K, 18, 18]

        # 가장 최대 값만 뽑아낸 피쳐맵을 생성한다.
        max_projection = torch.max(K_local_maps, 1, False)[0]
        max_projection = torch.unsqueeze(max_projection, dim=1)
        # output_size : [batch_size, 1, 18, 18]

        # 이를 기존 K개 피쳐맵에 덧붙여주어 K+1 차원의 로컬 랜드마크를 생성한다.
        local_landmarks = torch.cat((K_local_maps, max_projection), 1)
        # output_size : [batch_size, K+1, 18, 18]

        # 나머지 얼굴 이미지 클래시피케이션 진행
        x = self.maxpool(global_map)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # output_size : [batch_size, 9216]

        # 얼굴 이미지 클래시피케이션 결과 행렬
        classify_output = self.fc(x)
        # output_size : [9216, 8651]

        return global_map, classify_output, local_landmarks, K_local_maps

# Detector가 기반으로하는 ResNet 구성 요소들
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# Attender
# 각 로컬 영역별 피쳐맵들에 대하여 recalibration, attion pooling을 수행한다.
# 이는 템플릿 내에서 더 품질이 좋은 학습 이미지를 선별하는 과정에 해당한다.
class Attender(nn.Module) :
    def __init__(self, K=None, batch_size=None):
        super(Attender, self).__init__()
        self.recalibrate = nn.Softmax2d()
        self.K=K
        self.batch_size=batch_size
        return

    # 템플릿 안의 3장의 이미지들에 대한 로컬 피쳐맵 벡터와 글로벌 덴스 피쳐를 전달받는다.
    def forward(self, local_landmarks, global_maps):
        # 배치 사이즈 만큼 반복
        batch_tensor_list=[]
        for b in range(self.batch_size):
            # 로컬 피쳐맵의 개수 만큼 반복
            # 각 로컬 영역별로 피쳐 디스크립터를 저장할 배열 생성
            feature_descs = []

            for k in range(self.K+1):
                # 각 이미지 별로 어텐셔널 풀링을 진행하여 결과 값을 배열에 저장
                attention_pooled_values = []
                for n in range(len(local_landmarks)) :
                    # 먼저 해당 이미지의 로컬 피쳐맵의 recalibrate 수행
                    cur_recalibrated_maps=self.recalibrate(local_landmarks[n])

                    # 1024x18x18 차원의 글로벌 피쳐맵의 각 차원을 반복하면서
                    # recalibrate를 거친 로컬 피쳐맵을 element-wise로 곱해준 다음, 총합을 구한다.
                    temp_map = (global_maps[n][b] * cur_recalibrated_maps[b][k])
                    temp_sum = (temp_map.sum(1)).sum(1)
                    attention_pooled_values.append(temp_sum)

                # 각 이미지 별 피쳐 디스크립터를 하나로 합쳐서 템플릿 단위의 피쳐 디스크립터 생성
                feature_descs.append(sum(attention_pooled_values))

            # 각 로컬 영역들의 피쳐 디스크립터를 하나로 모아서 (K+1)x1024 피쳐맵을 생성
            # 그 뒤에 L2 normalization 수행
            merged_feature = torch.stack(feature_descs)
            merged_feature = nn.functional.normalize(input=merged_feature, p=2, dim=1)

            # 이를 배치 연산을 위해서 배치 텐서 리스트에 저장
            batch_tensor_list.append(merged_feature)

        # 배치 텐서 리스트를 다시 텐서로 변환해주어 리턴
        # Attend 과정 종료
        result_tensor =  torch.stack(batch_tensor_list)
        return result_tensor

# Comparator
# 템플릿 별로 추출한 벡터들을 주요 영역별로 합쳐준다.
# 또한 어느 부위를 나타내는지 표시하는 one-hot 벡터도 이어준다.
# 이를 각 영역별 fc, maxpool, 마지막 fc를 통과시킨다.
# 이렇게 구한 최종 유사도 백터를 배치 크기만큼 묶어서 리턴한다.
class Comparator(nn.Module) :
    def __init__(self,  K=None, batch_size=None):
        super(Comparator, self).__init__()
        self.K = K
        self.batch_size = batch_size
        self.fc_dict = {}
        for k in range(self.K) :
            self.fc_dict.update({k:nn.Linear(2061, 2048).cuda()})
        self.last_classifier = nn.Linear(2048, 2)
        return

    def forward(self, temp1_attended_vector, temp2_attended_vector, batch_size=None, K=None):
        # 어느 부위인지 나타내는 one_hot_vector 생성
        one_hot_2d = torch.zeros((self.K + 1), (self.K + 1))
        for i in range(self.K + 1):
            one_hot_2d[i][i] = one_hot_2d[i][i] + 1

        # 배치 사이즈 만큼 one_hot_vector를 쌓아준 뒤, 다시 텐서로 변환
        one_hot_list=[]
        for i in range(self.batch_size):
            one_hot_list.append(one_hot_2d)

        # 템플릿 1, 2 피쳐 벡터와 one-hot 벡터를 합쳐준다.
        one_hot_tensor = Variable(torch.stack(one_hot_list)).cuda()
        concat_templates_partid = torch.cat((temp1_attended_vector, temp2_attended_vector, one_hot_tensor), dim=2)

        # 각 영역별 fc를 통과한다.
        similarity_vector_list = []
        for b in range(self.batch_size):
            local_vector_list = []
            for k in range(self.K):
                cur_local_expert = self.fc_dict.get(k)
                local_vector_list.append(cur_local_expert(concat_templates_partid[b][k]))

            # (K+1)x2048 피쳐맵을 생성한다.
            local_tensor = torch.stack(local_vector_list)

            # maxpooling을 거쳐 1x2048 벡터를 추출한다.
            max_pooled_tensor = torch.max(input=local_tensor, dim=0, keepdim=False)[0]
            max_pooled_tensor = torch.unsqueeze(max_pooled_tensor, dim=0)

            # 마지막 fc를 통과하여 유사도를 판별할 최종 벡터를 추출하여 배열에 저장.
            similarity_vector = self.last_classifier(max_pooled_tensor)
            similarity_vector = similarity_vector.view(2)
            similarity_vector_list.append(similarity_vector)

        # 배치 크기 만큼 최종 유사도 판별 벡터를 묶어서 반환
        similarity_vector = torch.stack(similarity_vector_list)
        return similarity_vector
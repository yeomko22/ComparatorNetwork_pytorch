# 파일 개요 : 데이터 로더와 모델들을 불러와 데이터 학습을 진행하는 파일
import models
import dataloader
import torch
import torch.nn as nn
import utils
import sys
from torch.autograd import Variable

# 코드 실행 시 전달되는 파라미터를 파싱하는 파서 설정
args = utils.get_arg_parser()

# 학습용 얼굴 이미지들이 담겨있는 디렉터리 경로 전달받음
input_dir = args.input_dir
if input_dir is 'none' :
    print('Please input -i train_img_dir_path')
    sys.exit()

if input_dir[-1] != '/':
    input_dir+='/'

# 학습에 필요한 기본적인 변수 설정
num_epochs = 500
learning_rate = 0.0001
num_identities = 8651

# 로스 펑션에 적용되는 가중치
a1 = 2
a2 = 5
a3 = 30

# 배치 크기는 현재 사용 중인 하드웨어 환경에 따라 10으로 설정
# 논문에서는 64로 설정되어 있으니, 이 부분은 자신의 환경에 따라 조절할 것
batch_size = 10

# 얼굴 이미지에서 뽑아낼 주요 특징 부위 수 설정
K = 12

# 학습된 모델을 저장할 디렉터리 생성
utils.checkpoint_create()

# 디텍터 모델 객체 생성
# GPU 활용을 위해 쿠다 설정
comparator_network = models.ComparatorNetwork(batch_size=batch_size, K=K)
comparator_network.cuda()

# 소프트 맥스 학습을 위해서 크로스 엔트로피 로스 함수를 사용
# 옵티마이저로는 아담을 사용
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(comparator_network.parameters(), lr=learning_rate)

# 더 이상 에러율이 감소하지 않는 error plateau 현상이 10번 발생할 경우
# 학습율을 줄여주도록 lr_scheduler 설정
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# 학습과 테스트에 필요한 데이터 셋 객체를 만든다.
# 그리고 데이터 셋 객체를 활용해 데이터 로더 객체를 만든다.
train_data = dataloader.CustomDataset(train_dir=input_dir)
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

# 모델 학습 부분
# 먼저 전체 에포크 만큼 반복 설정
for epoch in range(num_epochs):

    # 학습용 데이터 로더를 순차적으로 읽어오면서 학습 진행
    iter_count = 0
    for i, sample in enumerate(train_loader):
        iter_count += 1
        # 이미지 텐서가 3개씩 묶여있는 템플릿을 읽어온다.
        template1 = sample['template1']
        template2 = sample['template2']

        # 라벨과 클래스 정보를 GPU 학습이 가능하게끔 쿠다 설정을 해준다
        class1 = Variable(sample['class1']).cuda()
        class1 = class1.squeeze()
        class2 = Variable(sample['class2']).cuda()
        class2 = class2.squeeze()

        label = Variable(sample['label']).cuda()
        label = label.squeeze()

        optimizer.zero_grad()

        # ---------- Detect 과정 ---------- #
        temp1_local_landmarks, temp2_local_landmarks, temp1_global_maps, temp2_global_maps, loss_cls1, loss_cls2, loss_reg \
            = comparator_network.detect(template1, template2, class1, class2, label)

        # ---------- Attend 과정 ---------- #
        temp1_attended_vector = comparator_network.attend(temp1_local_landmarks, temp1_global_maps)
        temp2_attended_vector = comparator_network.attend(temp2_local_landmarks, temp2_global_maps)

        # ---------- Compare 과정 ---------- #
        similarity_vector = comparator_network.compare(temp1_attended_vector, temp2_attended_vector)
        loss_sim = criterion(similarity_vector, label)

        # 클래시피케이션 로스, 유사도 측정 로스, 정규화 로스를 합쳐 전체 로스를 구한다.
        # 이를 백프로퍼게이션하여 신경망을 학습시킨다.
        total_loss = a1*(loss_cls1+loss_cls2) + a2*(loss_sim) + a3*(loss_reg)
        total_loss.backward()
        optimizer.step()

        if (i + 1) % 20 == 0:
            print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" % (epoch + 1, num_epochs, (i + 1), int(num_identities / batch_size), total_loss.item()))

    # 매 60000 iteration 마다 a3 절반으로 감소
    if iter_count == 60000:
        a3 *= 0.5

    # 매 에포크마다 모델 저장
    torch.save(comparator_network.state_dict(), '../checkpoint/comparator_netork.pkl')


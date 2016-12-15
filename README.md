Deep Learning을 이용한 Image Super Resolution.

============
국민대학교 컴퓨터공학부 권영훈

>###Contributor 국민대학교 컴퓨터공학부 교수. 김준호, 국민대학교 비주얼컴퓨팅 랩. 이진우

Image Super Resolution 관련 논문 Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang: Image Super-Resolution Using Deep Convolutional Networks (2015). 와
Jiwon Kim, Jung Kwon Lee, and Kyoung Mu Lee.: Accurate Image Super-Resolution Using Very Deep Convolutional Networks (2016).를 참고하여 구현.

##Github 구성 내용
+ Explanations - 구현 관련 설명, 보고서
+ SRCNN.py - CNN Learning
+ generate_test.py, generate_train.py - 트레이닝, 테스트 데이터 생성
+ full_image_saver.py - 트레이닝 완료 후 테스트 코드
+ train_epoch_cost - 수행횟수와 cost 값 저장
+ train_num_epoch - 총 수행 횟수 저장
+ Test, Train folder - 테스트와 트레이닝 이미지 파일


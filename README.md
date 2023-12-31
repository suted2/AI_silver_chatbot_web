# AI_silver_chatbot_web // Github URL : [https://github.com/suted2/AI_video_chatbot](https://github.com/suted2/AI_video_chatbot)
AI_silver_chatbot_web 구현을 위한 레포입니다. 
![flow](https://github.com/suted2/AI_silver_chatbot_web/assets/121469546/a5151761-a64d-4c7a-b9ec-ee8549f3334a)

## Web 첫 페이지

![슬라이드55](https://github.com/suted2/AI_silver_chatbot_web/assets/121469546/3ab61c58-b433-47ae-ab9b-8f1622e593a0)

링크 1 : [arsalpacos5.site](http://arsalpacos5.site/)

링크 2 : [alpaco5.site](http://alpaco5.site/) 

### 위의 링크는 모바일로 연결하면 더 원활하게 이용이 가능합니다.

( Bert 모델을 위한 파일, DataBase 연결, 영상 및 음성 파일이 없어 원활하게 구동이 되지 않습니다. 페이지를 이용하고 싶으시다면 @Kihoon9498로 메일주시면 감사하겠습니다.)

시작 첫 페이지 구성입니다.

해당 페이지에서 원하는 상담원을 선택하여 상담을 진행할 수 있습니다.


## Web 카테고리 선택 페이지 

![슬라이드56](https://github.com/suted2/AI_silver_chatbot_web/assets/121469546/87c9d19b-26d4-4957-86c5-708c145f13e6)

현재 카테고리 선택 페이지의 구성입니다.

해당 페이지에서 인사 영상이 자동으로 실행되고, 원하는 주제를 선택하여 상담을 진행할 수 있습니다.


## Web 질문 페이지

![슬라이드57](https://github.com/suted2/AI_silver_chatbot_web/assets/121469546/00848475-cd1a-49eb-aae9-7c8bb979931a)

1번 버튼 : 녹음 버튼을 눌러 질문을 시작하고, 한번 더 눌러 녹음을 종료할 수 있습니다.
          녹음된 파일은 STT 모델과 Toxic-Check 모델을 거쳐 질문 Text가 전달될 계획입니다.
          전달된 Text에 맞는 답변을 얻어, DataBase에 저장된 영상이 출력됩니다.

2번 버튼 : 시민의 상황에 맞게 Text로 질문도 가능한 버튼으로, 버튼을 누르면 Text를 입력할 수 있어, 일반 챗봇과 같이 Text로 질문을 할 수 있습니다.

3번 버튼 : 자주 묻는 질문(FAQ) 버튼으로, 시민들이 질문 한 횟수를 Count하여 DataBase에 실시간으로 적용되며 선택하면 답변을 볼 수 있습니다.

4번 버튼 : 상담원 연결 버튼으로, 선택한 카테고리와 관련된 부서로 연결할 수 있게 준비 할 예정입니다.
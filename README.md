# AI_silver_chatbot_web
AI_silver_chatbot_web 구현을 위한 레포입니다. [( 실버 AI 화상 상담 챗봇 Git )](https://github.com/suted2/AI_video_chatbot)
![flow](https://github.com/suted2/AI_silver_chatbot_web/assets/121469546/a5151761-a64d-4c7a-b9ec-ee8549f3334a)

( Bert 모델을 위한 파일, DataBase 연결, 영상 및 음성 파일이 없어 원활하게 구동이 되지 않습니다. )

## Web 첫 페이지 -> 링크 : [arsalpacos5.site](http://arsalpacos5.site/)

![슬라이드55](https://github.com/suted2/AI_silver_chatbot_web/assets/121469546/28d4d462-db32-439d-8ba5-5d95133d34e1)

시작 첫 페이지 구성입니다.

해당 페이지에서 원하는 상담원을 선택하여 상담을 진행할 수 있습니다.


## Web 메인 페이지 

![슬라이드56](https://github.com/suted2/AI_silver_chatbot_web/assets/121469546/99fc7e33-ce8a-4f3e-9067-6f241e869837)

현재 메인 페이지의 구성입니다. 

해당 메인 페이지에서 인사 영상이 자동으로 실행되고, 원하는 주제를 선택하여 상담을 진행할 수 있습니다.


## Web 질문 페지

![슬라이드57](https://github.com/suted2/AI_silver_chatbot_web/assets/121469546/e9761f14-349c-4eda-85d4-7d0e90e5419f)

1번 버튼 : 녹음 버튼을 눌러 질문을 시작하고, 한번 더 눌러 녹음을 종료할 수 있습니다.
          녹음된 파일은 STT 모델과 Toxic-Check 모델을 거쳐 질문 Text가 전달될 계획입니다.
          전달된 Text에 맞는 답변을 얻어, DataBase에 저장된 영상이 출력됩니다.

2번 버튼 : 시민의 상황에 맞게 Text로 질문도 가능한 버튼으로, 버튼을 누르면 Text를 입력할 수 있어, 일반 챗봇과 같이 Text로 질문을 할 수 있습니다.

3번 버튼 : 자주 묻는 질문(FAQ) 버튼으로, 시민들이 질문 한 횟수를 Count하여 DataBase에 실시간으로 적용되며 선택하면 답변을 볼 수 있습니다.

4번 버튼 : 상담원 연결 버튼으로, 선택한 카테고리와 관련된 부서로 연결할 수 있게 준비 할 예정입니다.

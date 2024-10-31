import cv2
import mediapipe as mp
import json
import time
import random
import os
import numpy as np
from screeninfo import get_monitors

# Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# 웹캠 초기화 및 해상도 가져오기
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("카메라에서 영상을 가져올 수 없습니다.")
    cap.release()
    exit()

webcam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
webcam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 실제 화면 해상도 가져오기
monitor = get_monitors()[0]  # 첫 번째 모니터 사용
screen_width = monitor.width
screen_height = monitor.height

# 사용자 ID 설정
user_id = input("사용자 ID를 입력하세요: ")

# 경로 설정 (상대 경로 사용)
image_save_path = "./image"
label_save_path = "./label"
os.makedirs(image_save_path, exist_ok=True)
os.makedirs(label_save_path, exist_ok=True)

# 변환 비율 계산 (웹캠 해상도 -> 화면 해상도)
width_ratio = webcam_width / screen_width
height_ratio = webcam_height / screen_height

# 눈꺼풀과 홍채 인덱스
left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
left_iris_indices = [468, 469, 470, 471]
right_iris_indices = [473, 474, 475, 476]

# 전체 화면 모드로 창 설정
cv2.namedWindow("Eye Tracking Data Collection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Eye Tracking Data Collection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 데이터 수집 루프 시작
while cap.isOpened():
    # 배경을 흰색으로 설정
    background = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255  # 흰색 배경

    # 화면 해상도 기준으로 랜덤 좌표 생성
    target_x_screen = random.randint(0, screen_width - 1)
    target_y_screen = random.randint(0, screen_height - 1)
    display_duration = 2  # 시간 설정

    # 화면 좌표를 웹캠 해상도로 변환하여 저장할 때 사용
    target_x_webcam = int(target_x_screen * width_ratio)
    target_y_webcam = int(target_y_screen * height_ratio)

    # JSON 파일 저장 형식 설정
    json_data = {
        "user_id": user_id,
        "target_screen_coordinates": {"x": target_x_screen, "y": target_y_screen},
        "target_webcam_coordinates": {"x": target_x_webcam, "y": target_y_webcam},
        "eye_center_coordinates": {"left_eye": None, "right_eye": None},
        "iris_info": {"left_iris": None, "right_iris": None},
        "eyelid_shape": {"left_eyelid": None, "right_eyelid": None},
        "head_rotation": None,
        "distance_from_screen": None
    }

    # 타겟 좌표 표시 시간 초기화
    start_time = time.time()
    data_captured = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라에서 영상을 가져올 수 없습니다.")
            break

        # 화면을 좌우 반전
        frame = cv2.flip(frame, 1)
        original_frame = frame.copy()  # 웹캠 이미지를 변환 없이 저장하기 위해 복사

        # 배경에 타겟 점 그리기
        cv2.circle(background, (target_x_screen, target_y_screen), 10, (0, 0, 255), -1)  # 빨간색 점

        # 화면 표시
        cv2.imshow("Eye Tracking Data Collection", background)

        # 이미지 캡처
        elapsed_time = time.time() - start_time
        if elapsed_time >= display_duration and not data_captured:
            # Mediapipe를 사용해 얼굴 랜드마크 탐지
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 1. 눈 중심 좌표 계산
                    left_eye_center = face_landmarks.landmark[468]  # 왼쪽 홍채 중심
                    right_eye_center = face_landmarks.landmark[473]  # 오른쪽 홍채 중심

                    json_data["eye_center_coordinates"] = {
                        "left_eye": {
                            "x": left_eye_center.x,
                            "y": left_eye_center.y,
                            "z": left_eye_center.z
                        },
                        "right_eye": {
                            "x": right_eye_center.x,
                            "y": right_eye_center.y,
                            "z": right_eye_center.z
                        }
                    }

                    # 2. 홍채 위치 및 윤곽
                    left_iris_points = [
                        {
                            "x": face_landmarks.landmark[i].x,
                            "y": face_landmarks.landmark[i].y
                        } for i in left_iris_indices
                    ]
                    right_iris_points = [
                        {
                            "x": face_landmarks.landmark[i].x,
                            "y": face_landmarks.landmark[i].y
                        } for i in right_iris_indices
                    ]

                    json_data["iris_info"] = {
                        "left_iris": left_iris_points,
                        "right_iris": right_iris_points
                    }

                    # 3. 눈꺼풀 형상 좌표
                    left_eyelid_points = [
                        {
                            "x": face_landmarks.landmark[i].x,
                            "y": face_landmarks.landmark[i].y
                        } for i in left_eye_indices
                    ]
                    right_eyelid_points = [
                        {
                            "x": face_landmarks.landmark[i].x,
                            "y": face_landmarks.landmark[i].y
                        } for i in right_eye_indices
                    ]

                    json_data["eyelid_shape"] = {
                        "left_eyelid": left_eyelid_points,
                        "right_eyelid": right_eyelid_points
                    }

                    # 4. 머리 회전 각도 (간단한 추정 방식)
                    nose_tip = face_landmarks.landmark[1]
                    left_eye = face_landmarks.landmark[468]
                    right_eye = face_landmarks.landmark[473]
                    head_rotation = [
                        nose_tip.x - (left_eye.x + right_eye.x) / 2,
                        nose_tip.y - (left_eye.y + right_eye.y) / 2,
                        nose_tip.z - (left_eye.z + right_eye.z) / 2
                    ]

                    json_data["head_rotation"] = head_rotation

                    # 5. 화면과의 거리 수정 (멀어질수록 값이 커지도록 설정)
                    distance_from_screen = (1 / abs(nose_tip.z)) * 1000  # 거리 반비례 관계 설정
                    json_data["distance_from_screen"] = distance_from_screen

                    # JSON 데이터 저장 (파일 이름에 사용자 ID 포함)
                    label_filename = f"{user_id}_data_{int(time.time())}.json"
                    label_path = os.path.join(label_save_path, label_filename)
                    with open(label_path, "w") as json_file:
                        json.dump(json_data, json_file, indent=4)

                    # 웹캠 이미지를 변환 없이 저장 (사용자 ID와 타임스탬프 포함)
                    image_filename = f"{user_id}_image_{int(time.time())}.jpg"
                    image_path = os.path.join(image_save_path, image_filename)
                    cv2.imwrite(image_path, original_frame)

                    print("이미지와 JSON 데이터가 저장되었습니다.")
                    data_captured = True  # 데이터가 캡처되었음을 표시
                    break  # 랜드마크 처리가 완료되었으므로 루프 종료

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print("데이터 수집을 종료합니다.")
            exit()

        # 데이터가 캡처되었으면 다음 타겟으로 이동
        if data_captured:
            time.sleep(1)  # 잠시 대기 후 다음 타겟으로 이동
            break  # 내부 루프를 빠져나와 새로운 타겟 생성

cap.release()
cv2.destroyAllWindows()

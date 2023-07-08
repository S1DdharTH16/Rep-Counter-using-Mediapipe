from flask import Flask, render_template, Response, request
import cv2
import math
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

app = Flask(__name__)
exercise = 'pushups'  # Default exercise selection

@app.route('/', methods=['GET', 'POST'])
def index():
    global exercise
    if request.method == 'POST':
        exercise = request.form.get('exercise')
    return render_template('index.html')

def count_reps(angles, threshold):
    reps = 0
    prev_angle = angles[0]

    for angle in angles[1:]:
        if angle < threshold and prev_angle >= threshold:
            reps += 1
        prev_angle = angle

    return reps


def calculate_angle(a, b, c):
    # Calculate the angle between three joints
    radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    degrees = math.degrees(radians)
    return degrees


def generate_frames():
    cap = cv2.VideoCapture(0)  # Capture video from webcam
    threshold = 90  # Angle threshold for rep counting
    exercise_data = exercises[exercise]
    angles = []

    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmark_a = results.pose_landmarks.landmark[exercise_data['landmark_a']]
                landmark_b = results.pose_landmarks.landmark[exercise_data['landmark_b']]
                landmark_c = results.pose_landmarks.landmark[exercise_data['landmark_c']]

                angle = calculate_angle(landmark_a, landmark_b, landmark_c)
                angles.append(angle)

                cv2.putText(image, f'{exercise.capitalize()} Angle: {int(angle)}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                reps = count_reps(angles, threshold)

                cv2.putText(image, f'{exercise.capitalize()} Reps: {reps}', (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

exercises = {
    'pushups': {
        'landmark_a': mp_pose.PoseLandmark.LEFT_SHOULDER,
        'landmark_b': mp_pose.PoseLandmark.LEFT_ELBOW,
        'landmark_c': mp_pose.PoseLandmark.LEFT_WRIST,
        'threshold': 90
    },
    'squats': {
        'landmark_a': mp_pose.PoseLandmark.LEFT_HIP,
        'landmark_b': mp_pose.PoseLandmark.LEFT_KNEE,
        'landmark_c': mp_pose.PoseLandmark.LEFT_ANKLE,
        'threshold': 120
    },
    'lunges': {
        'landmark_a': mp_pose.PoseLandmark.RIGHT_HIP,
        'landmark_b': mp_pose.PoseLandmark.RIGHT_KNEE,
        'landmark_c': mp_pose.PoseLandmark.RIGHT_ANKLE,
        'threshold': 120
    }
}

@app.route('/video_feed', methods=['POST'])
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

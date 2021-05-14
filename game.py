from keras.models import load_model
import cv2
import numpy as np
from random import choice

REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "nothing"
}


def mapper(val):
    return REV_CLASS_MAP[val]


def calculate_winner(user_move, Pi_move):
    if user_move == Pi_move:
        return "Tie"

    elif user_move == "rock" and Pi_move == "scissors":
        return "You" 

    elif user_move == "rock" and Pi_move == "paper":
        return "Pi"

    elif user_move == "scissors" and Pi_move == "rock":
        return "Pi"

    elif user_move == "scissors" and Pi_move == "paper":
        return "You"

    elif user_move == "paper" and Pi_move == "rock":
        return "You"

    elif user_move == "paper" and Pi_move == "scissors":
        return "Pi"


model = load_model("game-model.h5")

cap = cv2.VideoCapture(0)

prev_move = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.rectangle(frame, (10, 70), (300, 340), (0, 255, 0), 2)
    cv2.rectangle(frame, (330, 70), (630, 370), (255, 0, 0), 2)

    # extract the region of image within the user rectangle
    roi = frame[70:300, 10:340]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (227, 227))

    # predict the move made
    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0])
    user_move_name = mapper(move_code)

    # predict the winner (human vs computer)
    if prev_move != user_move_name:
        if user_move_name != "nothing":
            computer_move_name = choice(['rock', 'paper', 'scissors'])
            winner = calculate_winner(user_move_name, computer_move_name)
        else:
            computer_move_name = "nothing"
            winner = "Waiting..."
    prev_move = user_move_name

    # display the information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Your Move: " + user_move_name,
                (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Pi's Move: " + computer_move_name,
                (330, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Winner: " + winner,
                (100, 450), font, 2, (0, 255, 0), 4, cv2.LINE_AA)

    if computer_move_name != "nothing":
        icon = cv2.imread(
            "test_img/{}.png".format(computer_move_name))
        icon = cv2.resize(icon, (300, 300))
        frame[70:370, 330:630] = icon

    cv2.imshow("Rock Paper Scissors", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

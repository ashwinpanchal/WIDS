import cv2

# Load the video
cap = cv2.VideoCapture("messi.mp4")

# Load the reference image of the ball
ball = cv2.imread("ball.png", cv2.IMREAD_GRAYSCALE)

# Set up the detector and matcher
detector = cv2.ORB_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Get the keypoints and descriptors of the reference image
ball_kp, ball_des = detector.detectAndCompute(ball, None)

while True:
    # Read a frame from the video
    _, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get the keypoints and descriptors of the frame
    frame_kp, frame_des = detector.detectAndCompute(gray_frame, None)

    # Match the keypoints and descriptors of the frame and the reference image
    matches = matcher.match(frame_des, ball_des)

    # Sort the matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw the matches on the frame
    match_img = cv2.drawMatches(frame, frame_kp, ball, ball_kp, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the frame with matches
    cv2.imshow("Frame", match_img)

    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()

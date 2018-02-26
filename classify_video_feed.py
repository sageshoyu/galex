import cv2
import cube_classifier
import os

THRESHOLD = 0.97

input_path = input("Input video directory: ")
if os.path.exists(input_path):
    vidcap = cv2.VideoCapture(input_path)
    success,image = vidcap.read()
    success = True
    classifier = cube_classifier.PowerCubeClassifier()
    while success:
        success,image = vidcap.read()
        if success:
            image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            boxes, scores, classes, nums = classifier.get_classification(image)

            index = 0
            while scores[0][index] > THRESHOLD and index < scores.shape[1]:
                min_point = (int(boxes[0][index][1]*image.shape[1]),int(boxes[0][index][0]*image.shape[0]))
                max_point = (int(boxes[0][index][3]*image.shape[1]),int(boxes[0][index][2]*image.shape[0]))
                cv2.rectangle(image, min_point, max_point, (0,255,0))
                cv2.putText(image, str(scores[0][index]), (min_point[0],max_point[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),thickness=2)
                index+=1
            cv2.imshow('frame', image)
            cv2.waitKey(2)
    vidcap.release()
    cv2.destroyAllWindows()
else:
    print("Invalid input.")
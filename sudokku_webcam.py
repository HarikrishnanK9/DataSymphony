import cv2
from solver import *
import numpy as np
import operator
import tensorflow
import matplotlib.pyplot as plt
import easyocr

def pre_process_image(img, skip_dilate=False):
	#converting the original image to gray image
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#Uses a blurring function, adaptive thresholding and dilation to expose the main features of an image."""
  
	# Gaussian blur with a kernal size (height, width) of 9.
	# Note that kernal sizes must be positive and odd and the kernel must be square.
    blur_img1 = cv2.GaussianBlur(gray_img, (9, 9), 0)
	
    proc = cv2.adaptiveThreshold(blur_img1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # Invert colours, so gridlines have non-zero pixel values.

    proc = cv2.bitwise_not(proc, proc)
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
    proc = cv2.dilate(proc, kernel)

    return proc
def find_contours(img):
    # Load the thresholded image (proc)
    # Make sure 'proc' is a grayscale thresholded image in CV_8UC1 format

    # Find contours in the image
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Get the largest contour (assuming it represents the outer boundary of the Sudoku grid)
    polygon = contours[0]

    #print(polygon)
    return polygon

def distance_between(p1, p2): 
    a = p2[0] - p1[0] 
    b = p2[1] - p1[1] 
    return np.sqrt((a ** 2) + (b ** 2))

def crop_and_wrap(img,polygon):

    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in
                      polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in
                  polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in
                     polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in
                   polygon]), key=operator.itemgetter(1))
    crop_rect=[polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32') 
    side = max([distance_between(bottom_right, top_right), 
            distance_between(top_left, bottom_left),
            distance_between(bottom_right, bottom_left),   
            distance_between(top_left, top_right) ])
    dst = np.array([[0, 0], [side-1, 0], [side-1, side-1], [0, side-1]], dtype='float32')
    m = cv2.getPerspectiveTransform(src, dst)
    wrap_sud=cv2.warpPerspective(img, m, (int(side), int(side)))
    return wrap_sud,crop_rect,side
    

def find_squares(img):
    squares = [] 
    side = img.shape[:1] 
    side = side[0] / 9
    for j in range(9):
        for i in range(9):
            p1 = (i * side, j * side)  #Top left corner of a box   
            p2 = ((i + 1) * side, (j + 1) * side)  #Bottom right corner         
            squares.append((p1, p2))

    #print(len(squares))
    return squares


cap = cv2.VideoCapture("/home/harikrishnan/VSCode/OpenCV/sudokku/WhatsApp Video 2023-11-06 at 10.35.30 AM.mp4")
reader = easyocr.Reader(['en'])



while True:
    ret, frame = cap.read()
    frame=cv2.resize(frame,(800,600))
    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame1=pre_process_image(frame)
    corners=find_contours(frame1)
    #cv2.drawContours(frame, [polygon], -1, (255,0, 0), 2)
    #wrap_sud=crop_rect_grid(frame,polygon)
    cropped,crop_rect,side = crop_and_wrap(frame, corners)

    # same crop apply to gray image
    cropped_gray,crop_rect_gray,side_gray=crop_and_wrap(frame_gray,corners)

    # this finds the coordinates of diagonals of each of the 81 squares in the sudokku grid

    squares = find_squares(cropped)

    




    values=[]
    # loop for each square in the sudokku grid and is passes to reader to extract the digit in it
    for i in squares:
        x1, y1, x2, y2 = int(i[0][0]), int(i[0][1]), int(i[1][0]), int(i[1][1])
        square_image = cropped_gray[y1:y2, x1:x2]
        results = reader.readtext(square_image, detail=0, mag_ratio=2, allowlist='123456789')
    
    # if results is empty, 0 is added to the list values; otherwise, the obtained digit is appended.
        if not results:
            values.append('0')
        else:
            values.append(results[0])
    #values copy to grid as array
    grid=np.array(values)
    grid_final=grid.reshape(9,9)
    # change the datatype to int
    grid_final =  grid_final.astype(int)
    # we copy the grid to pass to the solver function so that the original extracted grid remains unchanged.
    pass_grid=grid_final.copy()
    print(grid_final)
    sol=solve_wrapper(pass_grid)
    #sol is a tuple containing 2 elemnets . one is solution grid and other is the time taken to solve.
    #we extract only the solution grid
    solution=sol[0]





    # Calculate the position for each number and overlay it on the cropped image

    solution_image=frame.copy()

    cell_width=side//9
    cell_height=side//9

    for i in range(9):
        for j in range(9):
            if (grid_final[i][j]==0 ):
			
			
            # Calculate the center point of the cell
                center_x = int(crop_rect[0][0])+(j * cell_width + (j + 1) * cell_width) // 2
                center_y = int(crop_rect[0][1])+(i * cell_height + (i + 1) * cell_height) // 2

            # Place the number at the center of the cell (you can adjust these coordinates)
                x = center_x - 10 # Adjust the positioning
                y = center_y + 10  # Adjust the positioning

            # Convert the digit to a string and draw it on the image
                solution_image=cv2.putText(solution_image, text=str(solution[i][j]), org=(int(x), int(y)), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0,255,0), thickness=3)



    # Display the modified image with the solution
    cv2.imshow("Sudoku Solution", solution_image)




    if cv2.waitKey(1) & 0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()

    
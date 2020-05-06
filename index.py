import numpy as np
import cv2
from pynput.mouse import Button, Controller
import wx

# create mouse object
mouse = Controller()

# initialise wx
app = wx.App(False)


(sx, sy) = wx.GetDisplaySize()  # get mac screen size

(camx, camy) = (640, 480)  # set cam screen or img reso size


cap = cv2.VideoCapture(0)
cap.set(3, camx)
cap.set(4, camy)

# lower and upper bound to detect green objects
lower_bound = np.array([33, 80, 40])
upper_bound = np.array([102, 255, 255])

# kernel size for morphology open and close
kernel_open = np.ones((5, 5))
kernel_close = np.ones((20, 20))

# fix shaky mouse movement using dampening
mouse_loc_prev = np.array([0, 0])
mouse_loc = np.array([0, 0])
damping_factor = 2  # should be more than 1

# mouse_loc = mouse_loc_prev + (target_loc - mouse_loc_prev) / damping_factor

# selectFlag
selectFlag = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (340, 220))

    # Convert RGB color to HSV
    img_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Filter green portion of image
    mask = cv2.inRange(img_HSV, lower_bound, upper_bound)

    # Remove noise from mask
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_close)

    # Final masking
    mask_final = mask_close

    # Draw contour around the object detected (i.e. green object)
    # RETR_External only takes outer most border
    # CHAIN_APPROX_NONE draws the whole border line
    contours, h = cv2.findContours(
        mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # find contours

    # draw rectangle on contour

    if len(contours) == 1:
        if selectFlag == 1:
            selectFlag = 0
            mouse.press(Button.left)

        x, y, w, h = cv2.boundingRect(contours[0])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cntx = int(x+w/2)
        cnty = int(y+h/2)
        cv2.circle(frame, (cntx, cnty), 2, (0, 0, 255), 2)

        # move mouse from cam to mac screen. sx - (cntx) is to invert mouse movement
        # mouse.position = (sx - (cntx*sx/camx), cnty*sy/camy)

        # including dampening_factor to avoid shaky mouse
        mouse_loc = mouse_loc_prev + \
            (np.int64((cntx, cnty)-mouse_loc_prev)/damping_factor)

        mouse.position = (sx-(mouse_loc[0]*sx/camx), mouse_loc[1]*sy/camy)

        # when mouse not moving, pass
        while mouse.position != (sx-(mouse_loc[0]*sx/camx), mouse_loc[1]*sy/camy):
            pass

        mouse_loc_prev = mouse_loc

    elif len(contours) == 2:
        if selectFlag == 0:
            selectFlag = 1
            mouse.press(Button.left)

        x1, y1, w1, h1 = cv2.boundingRect(contours[0])
        x2, y2, w2, h2 = cv2.boundingRect(contours[1])

        cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), (0, 0, 255), 2)
        cv2.rectangle(frame, (x2, y2), (x2+w2, y2+h2), (0, 0, 255), 2)

        cntx1 = int(x1+w1/2)
        cnty1 = int(y1+h1/2)
        cntx2 = int(x2+w2/2)
        cnty2 = int(y2+h2/2)

        cntx = int((cntx1+cntx2)/2)
        cnty = int((cnty1+cnty2)/2)

        cv2.line(frame, (cntx1, cnty1), (cntx2, cnty2), (255, 0, 0), 2)
        cv2.circle(frame, (cntx, cnty), 2, (0, 0, 255), 2)

        # move mouse from cam to mac screen. sx - (cntx) is to invert mouse movement
        # mouse.position = (sx - (cntx*sx/camx), cnty*sy/camy)

        # including dampening_factor to avoid shaky mouse
        mouse_loc = mouse_loc_prev + \
            (np.int64((cntx, cnty)-mouse_loc_prev)/damping_factor)

        print(mouse_loc)

        mouse.position = (sx-(mouse_loc[0]*sx/camx), mouse_loc[1]*sy/camy)

        # when mouse not moving, pass
        while mouse.position != (sx-(mouse_loc[0]*sx/camx), mouse_loc[1]*sy/camy):
            pass

        mouse_loc_prev = mouse_loc

        # cv2.drawContours(frame, contours, -1, (255, 0, 0), 3)  # draw contours

        ################################################
        # For Drawing All Contours and Number in boxes #
        ################################################

        #########
        # START #
        #########

        # # params for put_text fonts
        # font_face = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        # font_scale = 1
        # font_color = (0, 255, 255)

        # # draw rectangles for all contours found
        # for i in range(len(contours)):
        #     x, y, w, h = cv2.boundingRect(contours[i])  # find rectangle
        #     cv2.rectangle(frame, (x, y), (x+w, y+h),
        #                   (0, 0, 255), 2)  # draw rectangle
        #     cv2.putText(frame, str(i+1), (x, y+h),
        #                 font_face, font_scale, font_color)  # put numbering for all rectangle

        #######
        # END #
        #######

    # Display the resulting frame
    cv2.imshow('frame', frame)  # normal color
    # cv2.imshow('mask', mask)  # mask to detect green color
    # cv2.imshow('mask_open', mask_open)  # mask to remove white noises
    # cv2.imshow('mask_close', mask_close)  # mask to remove black noises

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

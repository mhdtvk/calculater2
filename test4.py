from json_report_md import *
from moviepy.editor import VideoFileClip
from PIL import Image
from ultralytics import YOLO

import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import os




class ChestDetecting :
    def __init__(self) -> None:
        self.model = None
        self.model_gen()

    def model_gen(self):
        self.model = YOLO('yolov8n-pose.pt')  # load an official model

    def chest_det(self, frame) :
        
        result = self.model(frame)
        if result[0].keypoints.xy.shape[1] != 0 :
            pixel_keypoints = result[0].keypoints.xy
            # Convert keypoints tensor to numpy array
            pixel_keypoints_numpy = pixel_keypoints.numpy()
            x1,y1 = pixel_keypoints_numpy[0][2][:2].astype(int)
            x2,y2 = pixel_keypoints_numpy[0][1][:2].astype(int)
            x3,y3 = pixel_keypoints_numpy[0][5][:2].astype(int)
            x4,y4 = pixel_keypoints_numpy[0][11][:2].astype(int)
            x5,y5 = pixel_keypoints_numpy[0][12][:2].astype(int)
            x6,y6 = pixel_keypoints_numpy[0][6][:2].astype(int)
            roi_points = np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6)], dtype=np.int32)
            roi_points = [roi_points]
            return roi_points
        else :
            print("Chest Not Detected!")
            return 0
            

class BreathDetection :
    def __init__(self) :
        print("\nThe BreathDetection init.\n")
        self.md_breath = MotionDetectionABMM()
        self.chest_points = ChestDetecting()
        self.lr = 0.1
        self.trsh = 25
        self.br_trsh = 50
        self.md_trsh = 2500
        self.json_report = JSONReportMD() 
        self.breath_detected = False
        self.motion_detected = False
        
    def optimize_motion_mask(self, fg_mask, min_thresh=0):

        _, thresh = cv2.threshold(fg_mask,min_thresh,255,cv2.THRESH_BINARY)
        motion_mask = cv2.medianBlur(thresh, 3)
        motion_mask = cv2.medianBlur(motion_mask, 3)
        # morphological operations
        kernel=np.array((9,9), dtype=np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return motion_mask
    
    # Function to calculate centroid
    def calculate_contours(self, binary_mask, roi_points):
        # Create a mask for the ROI
        mask = np.zeros_like(binary_mask)
        cv2.fillPoly(mask, roi_points, 255)

        # Apply the mask to the binary mask
        roi_masked = cv2.bitwise_and(binary_mask, mask)

        # Find contours within the region of interest
        contours, _ = cv2.findContours(roi_masked, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        areas = 0
        if contours:
            for contour in contours:
                M = cv2.moments(contour)
                if M['m00']:
                    area = cv2.contourArea(contour)
                    areas += area
            return areas, contours
        else:
            return 0, []

    def motion_detection(self, video_path : str, ylim = 10000, show = True, json_gen = False):
        print("\nThe motion_detection starts.\n")
        video = VideoFileClip(video_path)
        frame_rate = video.fps
        tot_num_frm = int(video.duration * frame_rate)
        video_time = 0.0
        video_name = os.path.basename(video_path)
        breath = 0
        bps = 0
        on_breath = False
        # Initialize variables for tracking movement
        areas = 0
        frame_number = -1
        motion_frames_count = 0

        if show :
            showmotions = ShowBreathFrames(self.md_trsh, tot_num_frm, ylim)
        # Initialize background model
        bg_model = None
        
        for frame_idx, frame_org in enumerate(video.iter_frames()) :           
            frame_number += 1
            frame = frame_org.copy()
                    
            video_time = frame_number / frame_rate
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi = self.chest_points.chest_det(frame)
            
            color = (0, 255, 0)  # Green color in BGR format
            thickness = 2  # Thickness of the square's edges
            


            if gray is not None and gray.size != 0 and roi:
                cv2.polylines(frame, roi, isClosed=True, color=color, thickness=thickness)

                # Initialize background model with the first frame
                if bg_model is None:
                    bg_model = gray.copy().astype("float")
                    continue
                # Update background model using running average
                # Resize the image
                cv2.accumulateWeighted(gray, bg_model, self.lr)
                # Compute absolute difference between current frame and background model
                diff = cv2.absdiff(gray, cv2.convertScaleAbs(bg_model))
                # Apply thresholding to obtain binary motion mask
                _, thresh = cv2.threshold(diff, self.trsh, 255, cv2.THRESH_BINARY)
                motion_frame = self.optimize_motion_mask(thresh,min_thresh=0)
                # Calculate centroid of motion mask
                areas, contours = self.calculate_contours(motion_frame, roi)

                if contours :
                    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)


                if areas > self.md_trsh :
                    self.motion_detected = True
                    self.breath_detected = True

                elif areas > self.br_trsh :
                    self.motion_detected = False
                    self.breath_detected = True

                else : 
                    self.motion_detected = False
                    self.breath_detected = False

                # Sending frame data to the JSON logger Class
                if json_gen:
                    self.json_report.frame_info_gen(frame_num= frame_number, time_stmp= video_time, motion_amount= areas, motion_detected=self.motion_detected, breath_detected=self.breath_detected)

                if self.breath_detected:
                    on_breath = True
                    # Counting the number of frames that has movment
                    motion_frames_count += 1

                if on_breath and self.breath_detected == False :
                    breath += 1
                    on_breath = False

                bps = (breath/2) / video_time * 60

                if show:
                    showmotions.show(frame, motion_frame, video_name, frame_number, areas, video_time, bps, md=self.motion_detected, bd=self.breath_detected)

        # Sending the video data to Json Logger
        if json_gen:
            self.json_report.alghorithm_info_gen(alg_name= "ABMM", lr= self.lr, md_trsh= self.md_trsh)
            self.json_report.video_info_gen(video_path = video_path, frames_with_motion = motion_frames_count)
            self.json_report.save_json_file()
            self.json_report.plot_json()
        # close all windows
        cv2.destroyAllWindows() 

class ShowBreathFrames :
    def __init__(self, md_trsh, xlim, ylim) -> None:
        self.md_trsh = md_trsh
        # Create a new figure with subplots
        self.fig, self.axes = plt.subplots(figsize=(14, 4))
        self.threshold_line = self.axes.axhline(y=self.md_trsh, color='blue', linestyle='--', label='Movement Threshold')
        self.threshold_line = self.axes.axhline(y=50, color='red', linestyle='--', label='Breath Threshold')
        self.plot_line, = self.axes.plot([], [],lw=1, color='red')
        self.axes.set_title('Frame/Motion amount')
        self.axes.set_xlim(0, xlim)  
        self.axes.set_ylim(0, ylim)  
        plt.tight_layout()
        # Define the font, position, and scale of the text
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.position1 = (50, 50)  # Position of the text (x, y)
        self.position2 = (50, 100)  # Position of the text (x, y)
        self.font_scale = 1  # Font scale (size)
        self.font_color = (255, 255, 255)  # Font color in BGR format
        self.thickness = 2  # Thickness of the text




    def show(self, orgframe, motionmask, videoname = 'no_name', frame_number = 0, areas = 0, video_time = 0.0, bps :int = 0, md= False, bd= False):
        text = f'Video name: {videoname} Frame: {frame_number} Motion amount: {areas} Time(s) : {video_time}'
        cv2.putText(orgframe, text, self.position1, self.font, self.font_scale, self.font_color, self.thickness)
        if md :
            text_mood = f'Limit: {self.md_trsh} Motion Detected : "Yes" ,Breath Detected : "Yes", BPM: {bps}'
            cv2.putText(orgframe, text_mood, self.position2, self.font, self.font_scale, (0,128,0), self.thickness)
        elif bd : 
            text_mood = f'Limit: {self.md_trsh} Motion Detected : "No" ,Breath Detected : {"Yes" if bd else "No"} BPM: {bps}'
            cv2.putText(orgframe, text_mood, self.position2, self.font, self.font_scale, (255, 255, 255), self.thickness)
        else :
            text_mood = f'Limit: {self.md_trsh} Motion Detected : "No" ,Breath Detected : "No", BPM: {bps}'
            cv2.putText(orgframe, text_mood, self.position2, self.font, self.font_scale, (0, 0, 255), self.thickness)

        self.update_mf_plot ((frame_number, areas))

        x1, y1 = 100, 100  # Top-left corner of the rectangle
        x2, y2 = 300, 300  # Bottom-right corner of the rectangle

        # Extract the ROI using slicing
        #roi = orgframe[y1:y2, x1:x2]

        # Perform analysis on the ROI (for demonstration, let's convert it to grayscale)
        #roi_gray = cv2.cvtColor(orgframe, cv2.COLOR_BGR2GRAY)

        cv2.namedWindow(f'Original Frame {videoname}', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Motion Mask', cv2.WINDOW_NORMAL)
        # Show the frames with the specified window sizes
        cv2.imshow(f'Original Frame {videoname}', orgframe)
        cv2.imshow('Motion Mask', motionmask)
        # Set the desired window sizes
        width, height = 900 , 500
        # Adjust the window sizes
        cv2.resizeWindow(f'Original Frame {videoname}', width, height)
        cv2.resizeWindow('Motion Mask', width, height)
        
        # Check for key press and break loop if 'q' is pressed
        key = cv2.waitKey(30)
        if key & 0xFF == ord('q'):
            exit(0)
        
    def update_mf_plot(self,new_data_point ):
        x_data, y_data = self.plot_line.get_data()
        x_data = list(x_data) + [new_data_point[0]]
        y_data = list(y_data) + [new_data_point[1]]
        self.plot_line.set_data(x_data, y_data)
        
        self.axes.relim()  # Recalculate limits
        self.axes.autoscale_view()  # Autoscale
        plt.draw()
        plt.pause(0.00005)
        

class ShowFrames :
    def __init__(self, md_trsh, xlim, ylim) -> None:
        self.md_trsh = md_trsh
        # Create a new figure with subplots
        self.fig, self.axes = plt.subplots(figsize=(14, 4))
        self.threshold_line = self.axes.axhline(y=self.md_trsh, color='blue', linestyle='--', label='Threshold')
        self.plot_line, = self.axes.plot([], [],lw=1, color='red')
        self.axes.set_title('Frame/Motion amount')
        self.axes.set_xlim(0, xlim)  
        self.axes.set_ylim(0, ylim)  
        plt.tight_layout()
        # Define the font, position, and scale of the text
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.position1 = (50, 50)  # Position of the text (x, y)
        self.position2 = (50, 100)  # Position of the text (x, y)
        self.font_scale = 1  # Font scale (size)
        self.font_color = (255, 255, 255)  # Font color in BGR format
        self.thickness = 2  # Thickness of the text

    def show(self, orgframe, motionmask, videoname = 'no_name', frame_number = 0, areas = 0, video_time = 0.0):
        text = f'Video name: {videoname} Frame: {frame_number} Motion amount: {areas} Time(s) : {video_time}'
        cv2.putText(orgframe, text, self.position1, self.font, self.font_scale, self.font_color, self.thickness)
        text_mood = f'Limit: {self.md_trsh} Motion Detected : {"Yes" if areas >= self.md_trsh else "No"}'
        cv2.putText(orgframe, text_mood, self.position2, self.font, self.font_scale, (0, 0, 255) if areas >= self.md_trsh else (255, 255, 255), self.thickness)

        self.update_mf_plot ((frame_number, areas))

        cv2.namedWindow(f'Original Frame {videoname}', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Motion Mask', cv2.WINDOW_NORMAL)
        # Show the frames with the specified window sizes
        cv2.imshow(f'Original Frame {videoname}', orgframe)
        cv2.imshow('Motion Mask', motionmask)
        # Set the desired window sizes
        width, height = 900 , 500
        # Adjust the window sizes
        cv2.resizeWindow(f'Original Frame {videoname}', width, height)
        cv2.resizeWindow('Motion Mask', width, height)
        
        # Check for key press and break loop if 'q' is pressed
        key = cv2.waitKey(30)
        if key & 0xFF == ord('q'):
            exit(0)
        
    def update_mf_plot(self,new_data_point ):
        x_data, y_data = self.plot_line.get_data()
        x_data = list(x_data) + [new_data_point[0]]
        y_data = list(y_data) + [new_data_point[1]]
        self.plot_line.set_data(x_data, y_data)
        
        self.axes.relim()  # Recalculate limits
        self.axes.autoscale_view()  # Autoscale
        plt.draw()
        plt.pause(0.00005)


class MotionDetectionABMM :
    def __init__(self) :
        self.md_trsh = None
        self.json_report = JSONReportMD() 
        self.motion_detected = False

    def optimize_motion_mask(self, fg_mask, min_thresh=0):

        _, thresh = cv2.threshold(fg_mask,min_thresh,255,cv2.THRESH_BINARY)
        motion_mask = cv2.medianBlur(thresh, 3)
        motion_mask = cv2.medianBlur(motion_mask, 3)
        # morphological operations
        kernel=np.array((9,9), dtype=np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return motion_mask
    
    # Function to calculate centroid
    def calculate_contours(self, binary_mask):
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        areas = 0
        if contours:
            for contour in contours:
                M = cv2.moments(contour)
                if M['m00']:
                    area = cv2.contourArea(contour)
                    areas += area
            return areas, contours
        else:
            return 0, []

    def motion_detection(self, video_path : str, lr = 0.09, trsh = 25, md_trsh = 2500, ylim = 10000, show = True, json_gen = True):
        self.md_trsh = md_trsh

        video = VideoFileClip(video_path)
        frame_rate = video.fps
        tot_num_frm = int(video.duration * frame_rate)
        video_time = 0.0
        video_name = os.path.basename(video_path)
        # Initialize variables for tracking movement
        areas = 0
        frame_number = -1
        motion_frames_count = 0

        if show :
            showmotions = ShowFrames(md_trsh, tot_num_frm, ylim)
        # Initialize background model
        bg_model = None
        for frame_idx, frame_org in enumerate(video.iter_frames()) :           
            frame_number += 1
            frame = frame_org.copy()
            video_time = frame_number / frame_rate
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Initialize background model with the first frame
            if bg_model is None:
                bg_model = gray.copy().astype("float")
                continue
            # Update background model using running average
            cv2.accumulateWeighted(gray, bg_model, lr)
            # Compute absolute difference between current frame and background model
            diff = cv2.absdiff(gray, cv2.convertScaleAbs(bg_model))
            # Apply thresholding to obtain binary motion mask
            _, thresh = cv2.threshold(diff, trsh, 255, cv2.THRESH_BINARY)
            motion_frame = self.optimize_motion_mask(thresh,min_thresh=0)
            # Calculate centroid of motion mask
            areas, contours = self.calculate_contours(motion_frame)

            if contours :
                cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

            self.motion_detected =  areas > self.md_trsh 
            # Sending frame data to the JSON logger Class
            if json_gen:
                self.json_report.frame_info_gen(frame_num= frame_number, time_stmp= video_time, motion_amount= areas, motion_detected= True)
            
            if self.motion_detected:
                # Counting the number of frames that has movment
                motion_frames_count += 1
          
            if show:
                showmotions.show(frame, motion_frame, video_name, frame_number, areas, video_time)

        # Sending the video data to Json Logger
        if json_gen:
            self.json_report.alghorithm_info_gen(alg_name= "ABMM", lr= lr, md_trsh= md_trsh)
            self.json_report.video_info_gen(video_path = video_path, frames_with_motion = motion_frames_count)
            self.json_report.save_json_file()
            self.json_report.plot_json()
        # close all windows
        cv2.destroyAllWindows() 


if __name__ == "__main__":
    
    
    if len(sys.argv) < 2:
        print("The Videos directory not found! Check the Dockerfile."," The input arguments:",sys.argv)
        sys.exit(1)

    folder_path = sys.argv[1]
    choice2 = int(input("\n".join([
    "\n Choose the Operation: ",
    " 1- Json Report Generator and Plotting : Motion per Frame",
    " 2- Showing Frames : Motion Detected frames and Plot",
    " 3- Breath Detection\n"
    ])))

    if choice2 == 1:  
        lr = float(input("\n Learning Rate : "))
        trsh = int(input("\n Threshold : "))
        md_trsh = int(input("\n Threshold for motion detection: "))      
        file_list = glob.glob(folder_path + '/*')  
        for file_path in file_list :
            motion_detection = MotionDetectionABMM()
            print("New File Generating...")
            motion_detection.motion_detection(file_path, lr, trsh, md_trsh=md_trsh, show = False, json_gen= True)
            
    elif choice2 == 2:
        lr = float(input("\n Learning Rate : "))
        trsh = int(input("\n Threshold : "))
        md_trsh = int(input("\n Threshold for motion detection: "))      
        file_list = glob.glob(folder_path + '/*') 
        ylim = int(input("\n The high range for displaying detected movements in plotting: "))
        
        file_list = glob.glob(folder_path + '/*')  
        for file_path in file_list :
            motion_detection = MotionDetectionABMM()
            print("New File Generating...")
            motion_detection.motion_detection(file_path, lr, trsh, md_trsh=md_trsh, ylim= ylim, show = True, json_gen= False)

    elif choice2 == 3:
        print('\nchoosed!\n')
        
        file_list = glob.glob(folder_path + '/*')  
        print(file_list,folder_path)
        for file_path in file_list :
            bd = BreathDetection()
            print("New File Generating...")
            print(file_path)
            bd.motion_detection(file_path)

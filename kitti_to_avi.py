import cv2
import os

fps = 10
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
for i in range(14,21):
    file_list = os.listdir("/Volumes/SN550_2/data_tracking_image_2/training/image_02/" + str(i).zfill(4))
    video_writer = cv2.VideoWriter(filename='/Volumes/SN550_2/data_tracking_image_2/training/image_02/videos/' + str(i).zfill(4) + '.avi', fourcc=fourcc, fps=fps, frameSize=(1242, 375))
    for j in range(0,len(file_list)):
        if os.path.exists("/Volumes/SN550_2/data_tracking_image_2/training/image_02/" + str(i).zfill(4) + "/" +str(j).zfill(6) + '.png'):
            img = cv2.imread(filename="/Volumes/SN550_2/data_tracking_image_2/training/image_02/" + str(i).zfill(4) + "/" +str(j).zfill(6) + '.png')
            img = cv2.resize(img, (1242, 375))
            video_writer.write(img)
            print(str(i).zfill(4) + "/" +str(j).zfill(6) + '.png' + ' done!')
    video_writer.release()
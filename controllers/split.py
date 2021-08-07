import cv2
import subprocess


import moviepy.editor as mp


file_dir = './data/video_file/video.mp4'
audio_dir = './data/audio_data/audio.wav'

def image_split():
    vidcap = cv2.VideoCapture(file_dir)

    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            #cv2.imwrite("../frames/image"+str(count)+".jpg", image)
            name = './data/image_data/frame' + str(count) + '.jpg'   # save frame as JPG file
            print('Creating...' + name)
            cv2.imwrite(name, image)
        return hasFrames
    sec = 0
    frameRate = 1 #//it will capture image in each 0.5 second
    count = 1
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)


def audio_split():

    video = mp.VideoFileClip(file_dir)
    video.audio.write_audiofile('./data/audio_data/audio.wav')








import cv2
import os
def extract_frames(dataset_path):
    output_path = os.path.join(dataset_path, 'frames_all')
    dict_path=os.path.join(dataset_path,"videos_libx265")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for video_path in os.listdir(dict_path):
        video_path_detail=os.path.join(dict_path,video_path)
        angle_name=video_path.split("_")[0]
        vidcap = cv2.VideoCapture(video_path_detail)
        success, image = vidcap.read()
        count = 0
        while success:
            out_frame_path=os.path.join(output_path,f"frames_{count}")
            if not os.path.exists(out_frame_path):
                os.mkdir(out_frame_path)
            out_write_path=os.path.join(out_frame_path,f"{angle_name}.jpg")
            cv2.imwrite(out_write_path, image)     # save frame as JPEG file      
            success, image = vidcap.read()
            count += 1
if __name__=="__main__":
    extract_frames("/data/byj/learning_nerf/data/0008_01")
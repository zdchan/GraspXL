from moviepy.editor import ImageSequenceClip


def images2video(im_paths, video_name, fps):
    assert len(im_paths) > 0
    image_files = im_paths
    clip = ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(video_name)

cd _out/training/image_2/
ffmpeg -framerate 20 -pattern_type glob -i '*_0.png' -c:v libx264 -profile:v high -crf 17 -pix_fmt yuv420p output_0.mp4
ffmpeg -framerate 20 -pattern_type glob -i '*_1.png' -c:v libx264 -profile:v high -crf 17 -pix_fmt yuv420p output_1.mp4
ffmpeg -framerate 20 -pattern_type glob -i '*_2.png' -c:v libx264 -profile:v high -crf 17 -pix_fmt yuv420p output_2.mp4
ffmpeg -framerate 20 -pattern_type glob -i '*_3.png' -c:v libx264 -profile:v high -crf 17 -pix_fmt yuv420p output_3.mp4
ffmpeg -framerate 20 -pattern_type glob -i '*_4.png' -c:v libx264 -profile:v high -crf 17 -pix_fmt yuv420p output_4.mp4

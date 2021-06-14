ffmpeg -r 20 -i %6d.png -c:v libx264 -crf 17 -vf fps=20 -pix_fmt yuv420p out.mp4

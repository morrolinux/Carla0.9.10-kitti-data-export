# for f in *.png-1; do mv $f $(echo $f|cut -d. -f1)_1.png; done

ffmpeg -r 20 -i %6d.png -c:v libx264 -crf 17 -vf fps=20 -pix_fmt yuv420p out.mp4
# or
ffmpeg -framerate 20 -pattern_type glob -i '*_1.png' -c:v libx264 -profile:v high -crf 17 -pix_fmt yuv420p output_1.mp4

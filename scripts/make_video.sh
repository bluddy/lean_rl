#!sh
ffmpeg -framerate 1/24 -i '%03d.png' -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4

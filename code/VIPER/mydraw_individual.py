# SAMPLE USAGE:
# python3 mydraw_individual.py < add_chars.dat

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import sys

width,height = 24,24
font_size=22
font_color=(0,0,0)

for line in sys.stdin:
 
  #print(i) 
  unicode_text = line.strip()
  i = ord(unicode_text)
  print(i)
  if i%500==0: 
    sys.stderr.write("%d\n"%i)
    sys.stderr.flush()

  im  =  Image.new ( "RGB", (width,height) )
  draw  =  ImageDraw.Draw ( im )
  unicode_font = ImageFont.truetype("DejaVuSans.ttf", font_size)
  try:
    draw.text ( (2,0), unicode_text, font=unicode_font) #, fill=font_color )
  except SystemError:
    continue 

  im.save("images/text-other_%d.ppm"%i)

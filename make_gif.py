from PIL import Image, ImageDraw, ImageFont
import imageio
import numpy as np



def make_rl_gif(outfile: str, imagelist: list, labellist: list):
    fontpath = '/usr/share/fonts/truetype/freefont/FreeSans.ttf'
    font = ImageFont.truetype(font = fontpath, size = 25)
    labeled_im = []
    for im, lab in zip(imagelist, labellist):
        tmpim = Image.fromarray(np.uint8(im))
        draw  = ImageDraw.Draw(tmpim)
        txt = "episodes trained = {}".format(lab)
        draw.text((10,10), txt, font = font, fill = 'black')
        labeled_im.append(np.array(tmpim))
    
    imageio.mimsave(outfile, labeled_im)





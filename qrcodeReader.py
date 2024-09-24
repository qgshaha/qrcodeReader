from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFont
from cv2.typing import Point
from pyzbar.pyzbar import decode, ZBarSymbol
from tqdm import tqdm
import cv2
import numpy as np

import argparse
from sys import stderr
import multiprocessing as mp
from pathlib import Path
from time import time
import matplotlib.pyplot as plt

class KImage(object):
    def __init__(self, filename):
        self.image = Image.open(filename)
        self.filename = Path(filename).name


def write_on_image(image, text):
    image = image.copy()
    font = ImageFont.load_default()
    bottom_margin = 3   # bottom margin for text
    text_height = font.getsize(text)[1] + bottom_margin
    left, top = (5, image.size[1] - text_height)
    text_width = font.getsize(text)[0]
    locus = np.asarray(image.crop((left, top, left + text_width, top + text_height)))
    meancol = tuple(list(locus.mean(axis=(0,1)).astype(int)))
    opposite = (int(locus.mean()) + 96)
    if opposite > 255:
        opposite = (int(locus.mean()) - 96)
    oppositegrey = (opposite, ) * 3
    draw = ImageDraw.Draw(image)
    draw.rectangle((left-3, top-3, left + text_width + 3, top + text_height + 3),
                   fill=meancol)
    draw.text((left, top), text, fill=oppositegrey, font=font)
    return image

def write_on_image(image, text, left, top):
    cv2.putText(image, text, 
        (left, top - 5), 
        fontFace=cv2.FONT_HERSHEY_DUPLEX, 
        fontScale=1,
        color=(36,255,12),
        thickness=2,
        lineType=2)

    #Display the image
    cv2.imshow("img",image)

    #Save image
    cv2.imwrite("out.jpg", image)

    cv2.waitKey(0)
    return image

def scale_image(image, scalar=None, h=None):
    if scalar == 1:
        return image
    x, y = image.size
    if scalar is None:
        if h is None:
            raise ValueError("give either h or scalar")
        scalar = 1 if h > y else h/y
    return image.resize((int(round(x*scalar)), int(round(y*scalar))))


polygons_dic = {}
def qrdecode(image):
    codes = decode(image, [ZBarSymbol.QRCODE,])

    global polygons_dic
    for code in codes:
        s = code.data.decode('utf8').strip()
        if s not in polygons_dic.keys():
            polygons_dic[s] = [code.polygon, scalar_g]

    # Decoded(data=b'B:CtWGdgGRkofcIFnTuvJEbnuog2EF4v+HO9nWlYGFIWuUprVC2crI+EhWX0PaODQ+ooqvubt/6tkz2NabFsRU3ngSO6zjPLIW2TM0cbDaS1968nWlkxyUzPd315DhuJwylVeM3K2t+Zvx+2bWLCOgJhVLWe3nXXjHLp/b4yeng4Ph4xL2pzu+aYfWRS3p0xoMf+LPs6oe0yKS7uja0yXiqWig9oJHiQVPk6ObrsUQL6uXBB2W+rEPszFTE2xyhbegPLd6swRroofsekq3RzvJLo95F2yFqeeZsElfx6QZnUUH6LxB9fRL1WYGbesGsYi7P+5W39Clry2dvBj419E4xiS3CYglu3377oAVdHotjVF+ZSLOw1oPverV2xFWOBgveZtBi8YTsLjEuEGPL1vxmdbaJJ9OvnrMXjwP0DzJEHDhd6mFRH9EMdFu9xCiyPAn7meE3vrdDNq/E7yAMAhfy8lGNOxjRchE+o77XnfVcyCfzrDDUxuccQsuRTd8RJj/YFQCteNbhsnjgj0hv+iGePzG6nVUkG5kdN5CMKdBKYFljgI/aFeIbztBZYcAAoM+v9aROPSRdq9qsw6MvZwLZiMU9MwhIn6v/+8gEUNYZzeKu1gzsUVHo4pfk5YoYwa9J/a/kHRiN7UJRXdGPF2dxSe8Eg2M6hBSPSS7WnxhKl74fEOj8w1kfzWEX13gZDDAdOU8405lqCeQ8ZTF47Yd3yP8yDHB1/4O6xxhVR4gef4sqAybnXt+F0/++g48uoSq8fZODjFVhPKsA/LJlr1/uUdTRx4FVU1cOXjVlGbXLqLhvrP1VsYz6r2CgUW9tXse6mOm++TBqux94bMhZojeeuExRjmfYAkQs6fzoMjuz+zF8CPkRBNpU0tApMF9sW0IonMSN2v7ANFv3p0zzhaHehc1w8e45moxp+VKcqxiwAH46Z5El6Y903dDesJauM35lt3YHecWqcYf0xXNVIpK55N3VZn2kRrSKGSfZnceRu4aSBC/YHsopIOi3VJE5tVT4hVzkBBtn4GC6BHaFDOB+0PZ1hfPPhjRfckDlv0QeWb1IdDgQy57UKOUf5OW9Ftn', type='QRCODE', rect=Rect(left=1501, top=243, width=483, height=472), polygon=[Point(x=1501, y=715), Point(x=1973, y=715), Point(x=1984, y=243), Point(x=1507, y=245)], quality=1, orientation='UP')
    return list(sorted(d.data.decode('utf8').strip() for d in codes))


def normalise_CLAHE(image):
    cvim = np.array(ImageOps.grayscale(image))
    clahe = cv2.createCLAHE()
    clahe_im = clahe.apply(cvim)
    #cv2.imshow("clahe", clahe_im)
    return Image.fromarray(clahe_im)


def rotate(image, rot=30):
    return image.copy().rotate(rot, expand=1)


def autocontrast(image):
    return ImageOps.autocontrast(image)


def sharpen(image, amount=1):
    sharpener = ImageEnhance.Sharpness(image)
    return sharpener.enhance(amount)


scalar_g = 1
def do_one(image):
    image = KImage(image)
    l = time()
    def tick():
        nonlocal l
        n = time()
        t = n - l
        l = n
        return t

    union = set()
    first = []
    first_t = 0
    total_t = 0
    results = []
    for scalar in [0.5, 0.2, 0.1, 1]:
    # for scalar in [1]:
        global scalar_g
        scalar_g = scalar
        tick()
        if scalar != 1:
            image_scaled = scale_image(image.image, scalar=scalar)
        else:
            image_scaled = image.image
        st = tick()
        res = qrdecode(image_scaled)
        #print(str(len(res))+".."+f"scaled-{scalar}" if scalar != 1 else "original")
        union.update(res); total_t += st
        if res:
            first = res
            first_t = total_t
        results.append({"file": image.filename,
                        "what": f"scaled-{scalar}" if scalar != 1 else "original",
                        "result": res,
                        "time": st})

        for sharpness in [0.1, 0.5, 2]:
            tick()
            image_scaled_sharp = sharpen(image_scaled, sharpness)
            res = qrdecode(image_scaled_sharp)
            print(str(len(res))+".."+f"scaled-{scalar}_sharpen-{sharpness}")
            t = tick()
            union.update(res); total_t += st + t
            if res:
                first = res
                first_t = total_t
            results.append({"file": image.filename,
                            "what": f"scaled-{scalar}_sharpen-{sharpness}",
                            "result": res,
                            "time": t + st})

        tick()
        image_scaled_autocontrast = autocontrast(image_scaled)
        res = qrdecode(image_scaled_autocontrast)

        print(str(len(res))+".."+f"scaled-{scalar}_autocontrast")
        t = tick()
        union.update(res); total_t += st + t
        if res:
            first = res
            first_t = total_t
        results.append({"file": image.filename,
                        "what": f"scaled-{scalar}_autocontrast",
                        "result": list(res),
                        "time": t + st})

    mark_cut_qrcode(image_scaled_autocontrast, polygons_dic, image.filename)
    results.append({"file": image.filename,
                    "what": f"do-all-the-things",
                    "result": list(union),
                    "time": total_t})
    results.append({"file": image.filename,
                    "what": f"take-the-first-thing",
                    "result": first,
                    "time": first_t})
    return results


def mark_cut_qrcode(image:Image, polygons_dic:dict, filename=None):
    # print(list(polygons_dic.values()))
    sorted_polygons = polygon_sort(polygons_dic.values())
    #sorted_polygons = sorted(polygons_dic.values(), key=lambda x:x[0][0].x/x[0][0][1] + x[0][0].y/x[0][0][1])
    #sorted_polygons = list(polygons_dic.values())
    marked_image = image.copy()
    #print(list(sorted_polygons))
    for i in range(len(sorted_polygons)):
        polygon = sorted_polygons[i][0]
        (left, upper, right, lower) = (polygon[0][0], polygon[0][1], polygon[2][0], polygon[2][1])
        
        if lower < upper:
            print("lower")
            (upper, lower) = (lower, upper)
        if right < left:
            print("right")
            (left, right) = (right, left)
        print((left, upper, right, lower))
        cut_image =  image.crop((left-50, upper-50, right+50, lower+50))
        cut_image.save(f"{Path(filename).stem}_{i+1}{Path(filename).suffix}")

        font = ImageFont.load_default(60)
        draw = ImageDraw.Draw(marked_image)
        draw.polygon(polygon, width=5, outline="red")
        draw.text((left, upper - 65), str(i+1), fill="red", font=font)

    marked_image.save(Path(filename).stem + "_marked"+ Path(filename).suffix)


pis =[]
def polygon_sort(polygons:list):
    print(list(polygons))
    sorted_polygons = []
    sorted_first_point = []
    polygons_dic = {}
    new_polygons = []
    
    for polygon in polygons:
        new_polygon = []
        for p in polygon[0]:
            new_polygon.append((p.x/polygon[1], p.y/polygon[1]))
        key = sortd(new_polygon, polygon[1])

        swap = (key[0], key[1], key[3], key[2])
        polygons_dic[str(key)] = [swap, polygon[1]]
        sorted_first_point.append(key[0])
        new_polygons.append(new_polygon)

    #print("sorted_first_point")
    #print(sorted_first_point)
    sorted_first_point = sortd(sorted_first_point)
    #print(sorted_first_point)
    global pis
    pis = sorted_first_point
    i = 0
    for point in sorted_first_point:
        item = list(filter(lambda x: polygons_dic[x][0][0] == point, polygons_dic))
        sorted_polygons.append(polygons_dic[item[0]])
        i=i+1
    print("01")
    print(list(sorted_polygons))
    return sorted_polygons


def sortd(Array, scalar = 1):
    y = Array[0][1]
    line1 = list(filter(lambda x: abs(x[1] - y) < 50/scalar, Array))
    line2 = list(filter(lambda x: abs(x[1] - y) > 50/scalar, Array))

    a = sorted(line1, key=lambda x:(x[0]),reverse=False)
    b = sorted(line2, key=lambda x:(x[0]),reverse=False)
    if(a[0][1] > b[0][1]):
        (a, b) = (b, a)
    #b = sorted(b, key=lambda x:(-x[0]),reverse=False)
    
    print("line1")
    print(a)
    print("line2")
    print(b)
    return a + b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--threads", type=int, default=None,
            help="Number of CPUs to use for image decoding/scanning")
    ap.add_argument("images", nargs="+", help="List of images")
    args = ap.parse_args()

    pool = mp.Pool(args.threads)

    print("Scanner", "Image", "Result", "Time", sep="\t")
    for image_results in tqdm(pool.imap(do_one, args.images), unit="images", total=len(args.images)):
        for result in image_results:
            barcodes = ";".join(sorted(result["result"]))
            print(result["what"], result["file"], barcodes, result["time"], sep="\t")


if __name__ == "__main__":
    main()

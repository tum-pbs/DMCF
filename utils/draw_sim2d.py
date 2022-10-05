#!/usr/bin/env python3
import sys
import h5py
import numpy as np
import skia  # python -m pip install skia-python
import argparse
from pathlib import Path
from collections import OrderedDict


def draw_frame(bnd, particles, width, height, particle_color, boundary_color,
               particle_radius, boundary_radius, **kwargs):
    """Draws boundary and fluid particels and returns the image as numpy array
    
    Args:
        bnd: boundary particles in pixel coordinates
        particles: fluid particles with shape [N,2] in pixel coordinates
        width: image width in px
        height: image height in px
        particle_color: ARGB color, e.g., 0xffff0000
        boundary_color: ARGB color, e.g., 0xffff0000
        particle_radius: circle radius in px
        boundary_radius: circle radius in px for boundary particles
    """
    surface = skia.Surface(width, height)
    with surface as canvas:
        canvas.drawColor(skia.ColorWHITE)

    # particles
    paint = skia.Paint(AntiAlias=True,
                       StrokeWidth=0,
                       Style=skia.Paint.kStrokeAndFill_Style,
                       Color=particle_color)
    for p in particles:
        canvas.drawCircle(*p, particle_radius, paint)

    # boundary
    paint = skia.Paint(AntiAlias=True,
                       StrokeWidth=0,
                       Style=skia.Paint.kStrokeAndFill_Style,
                       Color=boundary_color)
    for p in bnd:
        canvas.drawCircle(*p, boundary_radius, paint)

    return surface.toarray()


def draw_labels(labels, height, font_size=36, rot90=True):
    """Draws text labels and returns the images as numpy arrays"""
    font = skia.Font(None, font_size)

    if rot90:
        width = max([
            int(np.ceil(skia.TextBlob(l, font).bounds().height()))
            for l in labels
        ])
    else:
        width = max([
            int(np.ceil(skia.TextBlob(l, font).bounds().width()))
            for l in labels
        ])

    result = []
    for label in labels:
        surface = skia.Surface(width, height)
        with surface as canvas:
            canvas.drawColor(skia.ColorWHITE)
            blob = skia.TextBlob(label, skia.Font(None, font_size))
            rect = blob.bounds()
            if rot90:
                canvas.rotate(-90)
                canvas.translate(0, 0)
                canvas.translate(-(rect.centerX() + 0.5 * height),
                                 0.5 * width - rect.centerY())
            else:
                canvas.translate(0.5 * width - rect.centerX(),
                                 0.5 * height - rect.centerY())
            paint = skia.Paint(AntiAlias=True, Color=0xff000000)
            canvas.drawTextBlob(blob, 0, 0, paint)
            # canvas.drawRect(blob.bounds(), skia.Paint(AntiAlias=True, Color=skia.ColorRED, Style=skia.Paint.kStroke_Style))
        result.append(surface.toarray())
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Renders a simulation sequence from an hdf5 file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("path", type=str, help="The path to the h5 file")
    parser.add_argument("output",
                        type=str,
                        help="The path to the png output file")
    parser.add_argument(
        "--out_pattern",
        type=str,
        help=
        "Pattern like 'out_dir/{pointset}_{frame:04d}.png'. If defined individual frames will be written with this pattern"
    )
    parser.add_argument("--height",
                        type=int,
                        default=360,
                        help="The target height for a single frame")
    parser.add_argument(
        "--width",
        type=int,
        help="The target width for a single frame. Overrides 'height'")
    parser.add_argument(
        "--pr",
        dest="particle_radius",
        type=float,
        default=0.005,
        help="The radius for the particles in simulation units")
    parser.add_argument(
        "--br",
        dest="boundary_radius",
        type=float,
        help=
        "The radius for the boundary particles in simulation units. If not set uses the particle radius"
    )
    parser.add_argument("--margin",
                        type=float,
                        default=0.1,
                        help="The margin in percent")
    parser.add_argument(
        "--pointsets",
        type=str,
        nargs='+',
        default=['gt,GT', 'pred,Ours'],
        help=
        "List of pairs (pointsets,labels) to render, e.g., '--pointsets gt,GT pred,Ours"
    )
    parser.add_argument("--font_size",
                        type=float,
                        default=36.0,
                        help="Font size for the label")
    parser.add_argument("--num_frames",
                        type=int,
                        default=5,
                        help="Number of frames to render")
    parser.add_argument(
        "--frames",
        type=int,
        nargs='+',
        help="List of frames to render. This overrides 'num_frames'")
    parser.add_argument("--pc",
                        type=str,
                        default="0xff0071c5",
                        help="Particle color.")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 1
    args = vars(parser.parse_args())

    # colors in ARGB
    args['particle_color'] = int(args['pc'], base=16)
    # 0xff0071c5 # ours
    # 0xff919191 # gt
    # 0xffd64317 # gns
    # 0xffebc310 # cconv
    # 0xff82db36 # pointnet
    args['boundary_color'] = 0xff000000

    print(args)

    #
    # preprocess data
    #
    with h5py.File(args['path'], 'r') as h:
        key = 'SymNet'
        if not key in h and len(h) == 1:
            key = next(iter(h))
        data = {k: v[:] for k, v in h[key].items()}

    # discard z and mirror y
    coords = [0, 1]  # select x and y
    for k in data:
        v = data[k]
        v = v[..., coords]
        v[..., -1] *= -1  # mirror y
        data[k] = v

    # compute canvas size and transform to pixels
    bnd = data['bnd']
    if len(bnd) > 0:
        bb_min = bnd.min(axis=0)
        bb_max = bnd.max(axis=0)
    else:
        bb_min = np.full(2, -0.5)
        bb_max = np.full(2, 0.5)
    margin_scale = 1 + 2 * args['margin']
    bb_size = margin_scale * (bb_max - bb_min)
    bb_center = 0.5 * (bb_min + bb_max)
    bb_min = bb_center - 0.5 * bb_size
    bb_max = bb_center + 0.5 * bb_size
    shift = -bb_min

    if args['width'] is not None:
        scale = args['width'] / bb_size[0]
        args['height'] = int(np.round(bb_size[1] * scale))
    else:
        scale = args['height'] / bb_size[1]
        args['width'] = int(np.round(bb_size[0] * scale))

    for k in data:
        v = data[k]
        v = scale * (v + shift)
        data[k] = v

    args['particle_radius'] *= scale
    if args['boundary_radius'] is None:
        args['boundary_radius'] = args['particle_radius']
    else:
        args['boundary_radius'] *= scale

    # determine which frames to use
    if args['frames'] is None:
        # get the number of frames of the shortest sequence
        lengths = []
        for k, arr in data.items():
            if arr.ndim == 3:  # identify arrays with particle sequences
                lengths.append(arr.shape[0])
                print(k, 'has length', lengths[-1])
        frames = np.arange(min(*lengths))
        frames = np.array_split(frames, args['num_frames'])
        # get evenly spaced frame numbers
        frames = [x[0] for x in frames]
    else:
        frames = args['frames']
    print('selecting frames', frames)

    sequences = OrderedDict()
    pointset_images = OrderedDict()
    pointsets = [x.split(',')[0] for x in args['pointsets']]
    labels = [x.split(',')[-1] for x in args['pointsets']]
    label_images = draw_labels(labels, args['height'], args['font_size'])
    label_images = OrderedDict(zip(pointsets, label_images))
    for pointset in pointsets:
        images = []
        for frame in frames:
            images.append(
                draw_frame(data['bnd'], data[pointset][frame], **args))

            if args['out_pattern']:
                im_path = Path(args['out_pattern'].format(pointset=pointset,
                                                          frame=frame))
                if not im_path.parent.exists():
                    im_path.parent.mkdir(parents=True)
                skia.Image.fromarray(images[-1]).save(str(im_path), skia.kPNG)
        pointset_images[pointset] = images
        sequences[pointset] = np.concatenate([label_images[pointset]] + images,
                                             axis=1)

    result = np.concatenate(list(sequences.values()), axis=0)
    im = skia.Image.fromarray(result)
    im.save(args['output'], skia.kPNG)


if __name__ == '__main__':
    sys.exit(main())

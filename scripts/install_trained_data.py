#!/usr/bin/env python

import argparse
import multiprocessing

import jsk_data


def download_data(*args, **kwargs):
    p = multiprocessing.Process(
            target=jsk_data.download_data,
            args=args,
            kwargs=kwargs)
    p.start()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', dest='quiet', action='store_false')
    args = parser.parse_args()
    quiet = args.quiet

    PKG = 'rs_addons'

    # faster rcnn vgg16 voc07
    download_data(
        pkg_name=PKG,
        path='trained_data/faster_rcnn_vgg16_voc07_trained.npz',
        url='https://chainercv-models.preferred.jp/'
        'faster_rcnn_vgg16_voc07_trained_2018_06_01.npz',
        md5='c4411beaaff934b5c17e8ce8b74b0aa2',
        quiet=quiet,
    )

    # faster rcnn vgg16 voc0712
    download_data(
        pkg_name=PKG,
        path='trained_data/faster_rcnn_vgg16_voc0712_trained.npz',
        url='https://chainercv-models.preferred.jp/'
        'faster_rcnn_vgg16_voc0712_trained_2017_07_21.npz',
        md5='b2268f7612817ac71011807a901ef506',
        quiet=quiet,
    )

    # ssd300 voc0712
    download_data(
        pkg_name=PKG,
        path='trained_data/ssd300_voc0712_converted.npz',
        url='https://chainercv-models.preferred.jp/'
        'ssd300_voc0712_converted_2017_06_06.npz',
        md5='d420da9ec06c3d33820658394cfc305a',
        quiet=quiet,
    )

    # ssd512 voc0712
    download_data(
        pkg_name=PKG,
        path='trained_data/ssd512_voc0712_converted.npz',
        url='https://chainercv-models.preferred.jp/'
        'ssd512_voc0712_converted_2017_06_06.npz',
        md5='45de03b7bdaabb83cf5a25119662cd9e',
        quiet=quiet,
    )

    # fcis resnet101 sbd
    download_data(
        pkg_name=PKG,
        path='trained_data/fcis_resnet101_sbd_trained.npz',
        url='https://chainercv-models.preferred.jp/'
        'fcis_resnet101_sbd_trained_2018_06_22.npz',
        md5='a379573749ba64062d60d453ef9cd13d',
        quiet=quiet,
    )

    # maskrcnn resnet50 coco
    download_data(
        pkg_name=PKG,
        path='trained_data/mask_rcnn_resnet50_coco_trained.npz',
        url='https://drive.google.com/uc?id=19sciU40y_a3tN18QyLiQcWAuGc2hZw9p',
        md5='8e06483c0726acdb007ecbf503316a2d',
        quiet=quiet,
    )


if __name__ == '__main__':
    main()

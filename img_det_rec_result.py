import argparse
import logging as log
import sys
import time
from datetime import datetime
import cv2
import numpy as np
from prepeocess import DetResize, OcrPreProcess
from postprocess import DBPostProcess, OCRPostProcess, CLSPostProcess
from ov_operator_async import TextRecognizer, TextDetector, TextClassfier
from openvino.runtime import Core
import os


np.set_printoptions(threshold=np.inf)

def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    # fmt: off
    args.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    args.add_argument('-d', '--det', required=True, type=str,
                      help='Required. Path to an .xml or .onnx file with a model.')
    args.add_argument('-o', '--ocr', required=True, type=str,
                      help='Required. Path to an .xml or .onnx file with a model.')
    args.add_argument('-c', '--cls', default=None, type=str, help='Required. Path to an .xml or .onnx file with a model.')
    args.add_argument('-i', '--input', required=True, type=str, help='Required. Path to an image file dir.')
    args.add_argument('-l','--labels', default=None, type=str, help='Optional. Path to a labels mapping file.')
    args.add_argument('-n','--count', default=1, type=int, help='Optional. loop count.')
    args.add_argument('--det_streams', default=1, type=int, help='Optional. detection stream num.')
    args.add_argument('--ocr_streams', default=4, type=int, help='Optional. ocr stream num.')
    args.add_argument('-b','--obz', default=10, type=int, help='Optional. ocr batchsize.')
    args.add_argument('--drop_score', default=0.5, type=float, help='Optional. threshold of recognization.')
    args.add_argument('--cls_thresh', default=0.9, type=float, help='Optional. threshold of recognization.')
    args.add_argument('--output', type=str, help='Optional. Path to output dir.')
    args.add_argument('--det_db_box_thresh', default=0.5, type=float, help='Optional. det_db_box_thresh.')
    args.add_argument('--det_db_thresh', default=0.3, type=float, help='Optional. det_db_thresh.')
    args.add_argument('--det_db_unclip_ratio', default=1.6, type=float, help='Optional. det_db_unclip_ratio.')
    args.add_argument('--det_limit_side_len', default=960, type=float, help='Optional. det_limit_side_len.')
    args.add_argument('--det_limit_type', default='max', type=str, help='Optional. det_limit_type.')


    # fmt: on
    return parser.parse_args()

def run_inference(input, img_preprocess, detector, det_postprocess, 
                          ocr_preprocess, classfier, cls_postprocess, 
                          recognizer, ocr_postprocess, ocr_batch_num) :
    iImg = input
    img = cv2.imread(input)
    image = img_preprocess(img)
    # shape_list=[[src_h, src_w, ratio_h, ratio_w]]
    image = np.expand_dims(image, axis=0)
    res = detector(image)
    dt_boxes = det_postprocess(res, img.shape[0], img.shape[1])
    batch_list, indices = ocr_preprocess(img, dt_boxes, ocr_batch_num)
    # if classfier is not None and cls_postprocess is not None:
    #     cls_res = classfier(batch_list)
    #     batch_list, indices = cls_postprocess(cls_res, indices, dt_boxes)
    txt_res = recognizer(batch_list)
    txt_res = ocr_postprocess(txt_res, indices, dt_boxes)
    return txt_res
        
def run_inference_time(input, img_preprocess, detector, det_postprocess, 
                          ocr_preprocess, classfier, cls_postprocess, 
                          recognizer, ocr_postprocess, ocr_batch_num, tf) : ######
    step0 = datetime.utcnow()
    iImg = input
    img = cv2.imread(input)
    step1 = datetime.utcnow()
    print("read img {}, shape={}".format((step1-step0).total_seconds(), img.shape))
    step1 = datetime.utcnow()
    image = img_preprocess(img)
    # shape_list=[[src_h, src_w, ratio_h, ratio_w]]
    image = np.expand_dims(image, axis=0)
    step2 = datetime.utcnow()
    print("img_preprocess img {}, shape={}".format((step2-step1).total_seconds(), image.shape))
    step1 = datetime.utcnow()
    res = detector(image)
    step2 = datetime.utcnow()
    print("detetion {}".format((step2-step1).total_seconds()))
    step1 = datetime.utcnow()
    dt_boxes = det_postprocess(res, img.shape[0], img.shape[1])
    step2 = datetime.utcnow()
    print("det_postprocess img {}, got {}".format((step2-step1).total_seconds(), len(dt_boxes)))
    step1 = datetime.utcnow()
    batch_list, indices = ocr_preprocess(img, dt_boxes, ocr_batch_num)
    step2 = datetime.utcnow()
    print("ocr_preprocess {}".format((step2-step1).total_seconds()))
    # for bb in batch_list :
    #     print("batch size = {}".format(bb.shape))
    # if classfier is not None and cls_postprocess is not None:
    #     step1 = datetime.utcnow()
    #     cls_res = classfier(batch_list)
    #     step2 = datetime.utcnow()
    #     print("classfication {}".format((step2-step1).total_seconds()))
    #     step1 = datetime.utcnow()
    #     batch_list, indices = cls_postprocess(cls_res, indices, dt_boxes)
    #     step2 = datetime.utcnow()
    #     print("cls_postprocess {}".format((step2-step1).total_seconds()))
    step1 = datetime.utcnow()
    txt_res = recognizer(batch_list)
    step2 = datetime.utcnow()
    print("recognition {}".format((step2-step1).total_seconds()))
    step1 = datetime.utcnow()
    txt_res = ocr_postprocess(txt_res, indices, dt_boxes)
    step2 = datetime.utcnow()
    print("ocr_postprocess {}, got {}".format((step2-step1).total_seconds(), len(txt_res)))
    print("total e2e using: {}".format((step2-step0).total_seconds()))
    tf.write("total e2e using: {}\n".format((step2-step0).total_seconds())) ######
    return txt_res
    
def print_res(results):
    num = len(results)
    for i in range(num)  :
        print("{},{},{},{},{},{},{},{} {} {}".format(
                results[i][1][0][0], results[i][1][0][1],
                results[i][1][1][0], results[i][1][1][1],
                results[i][1][2][0], results[i][1][2][1],
                results[i][1][3][0], results[i][1][3][1],
                results[i][0][0], results[i][0][1]))

def fileout_res(filename, results, output):
    image_num = filename.split('e')[-1].split('.')[0]
    filepath='{}/{}.txt'.format(output,image_num)
    if not os.path.exists(output):
        os.makedirs(output)
    print("writing result to {}".format(filepath))
    with open(filepath, mode='w', encoding='utf-8') as file:
        num = len(results)
        for i in range(num)  :
            item = "{},{},{},{},{},{},{},{} {}\n".format(
                results[i][1][0][0], results[i][1][0][1],
                results[i][1][1][0], results[i][1][1][1],
                results[i][1][2][0], results[i][1][2][1],
                results[i][1][3][0], results[i][1][3][1],
                results[i][0][0])
            file.write(item)

def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    start = datetime.utcnow()
    args = parse_args()

    core = Core()

    #init Detecter
    detector = TextDetector(args.det, core)
    detector.setup_model(args.det_streams)
    #init det preprocess (reize)
    img_preprocess = DetResize(args.det_limit_side_len, args.det_limit_type)
    #init Detecter postpreocess
    det_postprocess = DBPostProcess(thresh=args.det_db_thresh, box_thresh=args.det_db_box_thresh,
                 max_candidates=1000, unclip_ratio=args.det_db_unclip_ratio, use_dilation=True)

    #init OCR
    recognizer = TextRecognizer(args.ocr, core)
    recognizer.setup_model(args.ocr_streams)
    #init OCR prepreocess
    ocr_preprocess = OcrPreProcess()
    #init ocr postpreocess
    ocr_postprocess = OCRPostProcess(args.labels, args.drop_score, use_space_char = True)

   #init CLS & postpreocess
    classfier = None
    cls_postprocess = None
    if args.cls is not None:
        classfier = TextClassfier(args.cls, core)
        classfier.setup_model(args.cls_streams)
        cls_postprocess = CLSPostProcess(args.labels, args.orc_threshold)
 

    #begin inference
    num_of_input = len(args.input)
    ocr_batch_num = args.obz

    filenames=[]
    for filename in os.listdir(args.input):
        #filename = args.input[0]+filename

        filenames.append(filename)
    print("start inference testing ... ")
    time_file = open('{}/time_log.txt'.format(args.output), mode='w', encoding='utf-8') ######
    first_start = datetime.utcnow()
    first_end = datetime.utcnow()
    for inputz in filenames:
        res = run_inference_time(args.input+inputz, img_preprocess, detector, det_postprocess, 
                          ocr_preprocess, classfier, cls_postprocess, 
                          recognizer, ocr_postprocess, ocr_batch_num, time_file) ######
        if args.output is not None :
             fileout_res(inputz, res, args.output)
        else:
             print_res(res)
    end = datetime.utcnow()
    time_file.close()
    files=len(filenames) * args.count
    first_ms = f"{(first_end - first_start).total_seconds() :.3f}"
    ave_ms = f"{(end - first_end).total_seconds() / files :.3f}"
    print("first inference using {}, average using {}".format(first_ms, ave_ms))
    return 0

if __name__ == '__main__':
    sys.exit(main())

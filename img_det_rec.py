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
import gc

gc.disable()


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
    args.add_argument('-i', '--input', required=True, type=str, nargs='+', help='Required. Path to an image file(s).')
    args.add_argument('-l','--labels', default=None, type=str, help='Optional. Path to a labels mapping file.')
    args.add_argument('-n','--count', default=1, type=int, help='Optional. loop count.')
    args.add_argument('--det_streams', default=1, type=int, help='Optional. detection stream num.')
    args.add_argument('--ocr_streams', default=4, type=int, help='Optional. ocr stream num.')
    args.add_argument('-b','--obz', default=10, type=int, help='Optional. ocr batchsize.')
    args.add_argument('-O','--orc_threshold', default=0.7, type=float, help='Optional. threshold of recognization.')
    args.add_argument('-C','--cls_threshold', default=0.7, type=float, help='Optional. threshold of recognization.')

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
                          recognizer, ocr_postprocess, ocr_batch_num) :
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
    print("ocr_postprocess {}".format((step2-step1).total_seconds()))
    print("total e2e using: {}".format((step2-step0).total_seconds()))
    return txt_res

def print_res(results):
    num = len(results)
    for i in range(num)  :
        print("[[{}, {}, {}, {}], {}]".format( 
            results[i][1][0], results[i][1][1], results[i][1][2], results[i][1][3], results[i][0]))
        # print("{}: {} at ({},{}), ({},{})".format(i, results[i][0], 
        # results[i][1][0], results[i][1][1], results[i][1][2], results[i][1][3]))

def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    start = datetime.utcnow()
    args = parse_args()

    core = Core()

    #init Detecter
    detector = TextDetector(args.det, core)
    detector.setup_model(args.det_streams)
    
    #init OCR
    recognizer = TextRecognizer(args.ocr, core)
    recognizer.setup_model(args.ocr_streams)

    #init CLS
    classfier = None
    cls_postprocess = None
    # if args.cls is not None:
    #     classfier = TextClassfier(args.cls, core)
    #     classfier.setup_model(args.cls_streams)
  

    #init det preprocess (reize)
    img_preprocess = DetResize()

    #init Detecter postpreocess
    det_postprocess = DBPostProcess()

    #init OCR prepreocess
    ocr_preprocess = OcrPreProcess()

    #init cls postpreocess
    cls_postprocess = CLSPostProcess(args.labels, args.orc_threshold)
 
    #init ocr postpreocess
    ocr_postprocess = OCRPostProcess(args.labels, args.orc_threshold)


    #begin inference
    num_of_input = len(args.input)
    ocr_batch_num = args.obz

    args.input.sort()
    results =[]
    print("start inference testing ... ")
    first_start = datetime.utcnow()
    run_inference_time(args.input[0], img_preprocess, detector, det_postprocess, 
                          ocr_preprocess, classfier, cls_postprocess, 
                          recognizer, ocr_postprocess, ocr_batch_num)
    # results.append(len(res))
    first_end = datetime.utcnow()
    for loop in range(args.count) :
        for input in args.input:
            run_inference(input, img_preprocess, detector, det_postprocess, 
                          ocr_preprocess, classfier, cls_postprocess, 
                          recognizer, ocr_postprocess, ocr_batch_num)
            # results.append(len(res))
    end = datetime.utcnow()
#    res = run_inference(args.input[0], img_preprocess, detector, det_postprocess, 
#                          ocr_preprocess, classfier, cls_postprocess, 
#                          recognizer, ocr_postprocess, ocr_batch_num)
#    results.append(len(res))
    first_ms = f"{(first_end - first_start).total_seconds() :.3f}"
    ave_ms = f"{(end - first_end).total_seconds() / args.count :.3f}"
    print("first inference using {}, average using {}".format(first_ms, ave_ms))
    return 0

if __name__ == '__main__':
    sys.exit(main())

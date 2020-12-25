from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmdet.core import get_classes
import mmcv
import numpy as np
import cv2
import torch
from os import listdir

def main():
    parser = ArgumentParser()
    parser.add_argument('imgfolder', help='Image file folder')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    cococlasses = get_classes('coco')

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    flist = listdir(args.imgfolder) # list all img names in folder
    for im in flist:
        img = args.imgfolder + im
        print('inferencing: '+img)
        # test a single image
        result = inference_detector(model, img)
        #print(result)
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            # both is array of size 80
        else:
            bbox_result, segm_result = result, None

        # gather all bboxs
        bboxes = np.vstack(bbox_result)

        # create labels from bbox_result
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        # only make label for not-none segs
        if segm_result is None or len(labels)==0:
            print('No result found.')
            continue

        # gather all segmentations
        segms = mmcv.concat_list(segm_result)

        # cilter out useful detection indexs
        filtered_inds = np.where(bboxes[:, -1] > args.score_thr)[0]
        print('  -indexs above threshold: '+str(filtered_inds))

        carortruck_inds = []
        for i in filtered_inds:
            if cococlasses[labels[i]]=='car' or cococlasses[labels[i]]=='truck':
                carortruck_inds.append(i)
        filtered_inds = np.array(carortruck_inds)

        if len(filtered_inds)==0:
            print('No result found is above threshold.')
            continue

        # Now, there is result found
        # begin create labelme-format json content
        content = '{\n"version": "4.5.6", \n"flags": {}, \n"shapes": [\n'
        #--------------------------------------------------


        # gather filtered items
        bboxes = bboxes[filtered_inds]
        segms = [segms[ind] for ind in filtered_inds]
        labels = labels[filtered_inds]

        # imread original image, for visualization purpose
        curvs = cv2.imread(img)

        for ind in range(len(segms)):

            if cococlasses[labels[ind]]!='car' and cococlasses[labels[ind]]!='truck':
                continue
            content += '{\n"label":"' + cococlasses[labels[ind]] + '",\n'
            #--------------------------------------------------

            one_seg = segms[ind]
            if isinstance(one_seg, torch.Tensor):#if seg is type Tensor, convert to nparray
                one_seg = one_seg.detach().cpu().numpy() # ms rcnn

            # get the binary mask of current seg
            mask = one_seg.astype(np.uint8) * 255
            #cv2.imwrite('output'+im[:-4]+'-mask-'+str(ind)+'.jpg', mask)

            # get contour(s) of current seg
            contours, hie = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
            print('  -contours total in current seg: '+str(len(contours)))

            #if len(contours)!=0:
            content += '"points": [\n'

            # for each contour part, use cv2.approxPolyDP to reduce number of points in contour
            for j,cont in enumerate(contours):
                print('    -contour '+str(j)+' point set size: '+str(len(cont)))
                epsilon = 0.001 * cv2.arcLength(cont, True)
                approx = cv2.approxPolyDP(cont, epsilon, True)
                #print(approx)
                print('     after approxPolyDP: '+str(len(approx)))
                # draw on image to visualize
                curvs = cv2.drawContours(curvs, [approx], -1, (0,255), 2)

                #--------------------------------------------------
                for k,pt in enumerate(approx):
                    pt = pt[0]
                    print(pt)
                    content += '['+str(pt[0])+','+str(pt[1])+']'
                    if j==len(contours)-1 and k == len(approx)-1:
                        content += '\n'
                    else:
                        content += ',\n'
                    #--------------------------------------------------
                    curvs = cv2.circle(curvs, tuple(pt), 0, (255,0,0), 5)

            content += '],\n'
            content += '"group_id": null,\n"shape_type": "polygon",\n"flags": {}\n}'
            if ind != len(segms)-1:
                content += ',\n'
            else:
                content += '\n],'
        content += '"imagePath": "'+im+'",\n"imageData": null,\n'
        content += '"imageHeight": '+str(curvs.shape[0])+',\n'
        content += '"imageWidth": '+str(curvs.shape[1])+'\n}'
        #--------------------------------------------------


        #cv2.imwrite('output/'+im[:-4]+'-poly.jpg', curvs)

        with open('output/'+im[:-3]+'json', 'w') as fjson:
            fjson.writelines(content)


        # show the results
        #show_result_pyplot(model, img, result, score_thr=args.score_thr)
        #model.show_result(img, result, score_thr=args.score_thr, show=False, out_file='output/out_'+im)


if __name__ == '__main__':
    main()

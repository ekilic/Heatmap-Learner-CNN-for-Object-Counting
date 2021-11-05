import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import localizerVgg
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plot
import visdom
import math
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from CARPK import CARPK

def vis_MAP(MAP, vis, epoch, batch_idx, mapId, upsampler):
    M1 = MAP.data.cpu().contiguous().numpy().copy()
    M1_norm = (M1[0,] - M1[0,].min()) / (M1[0,].max() - M1[0,].min())
    b = upsampler(torch.Tensor(M1_norm))
    b = np.uint8(cm_jet(np.array(b)) * 255)
    vis.image(np.transpose(b, (2, 0, 1)), opts=dict(
        title=str(epoch) + '_' + str(batch_idx) + '_' + str(mapId) + '_heatmap'))

def detect_peaks(image):
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    background = (image == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks

if __name__ == '__main__':
    downsampling_ratio = 8 # Downsampling ratio
    test_dataset = CARPK('', 'test', train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    cm_jet = mpl.cm.get_cmap('jet')
    model = localizerVgg.localizervgg16(pretrained=True, dsr=downsampling_ratio)
    model.load_state_dict(torch.load('trained_model_CARPK_x8_2_12.pt'))

    model.eval()
    model.cuda()

    vis = visdom.Visdom(server='http://localhost', port='8097')

    gi = 0
    gRi = 0
    ind = 0
    with torch.no_grad():
        for batch_idx, (im, GAM, numCar) in enumerate(test_loader):
            id_= batch_idx
            image = im.cuda()
            MAP = model(image)
            cMap = MAP[0,0,].data.cpu().numpy()
            cMap = (cMap - cMap.min()) / (cMap.max() - cMap.min())

            img_vis = im[0].cpu()
            img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())

            upsampler = transforms.Compose([transforms.ToPILImage(), transforms.Resize((im.shape[2], im.shape[3]))])
            vis_MAP(MAP, vis, 0, batch_idx, 1, upsampler)


            cMap[cMap < 0.05] = 0
            peakMAP = detect_peaks(cMap)


            arrX = np.where(peakMAP)[0]
            arrY = np.where(peakMAP)[1]
            for i in range(0, arrX.shape[0]):
                for k in range(-2, 2):
                    for j in range(-2, 2):
                        img_vis[0, arrX[i]*downsampling_ratio+k, arrY[i]*downsampling_ratio+j] = 1
                        img_vis[1, arrX[i]*downsampling_ratio+k, arrY[i]*downsampling_ratio+j] = 0
                        img_vis[2, arrX[i]*downsampling_ratio+k, arrY[i]*downsampling_ratio+j] = 0

            vis.image(img_vis, opts=dict(title=str(batch_idx) + '_image'))

            fark = np.sum(peakMAP) - int(numCar[0])
            gi = gi + abs(fark)
            gRi = gRi + fark*fark
            ind = ind + 1

            print(id_,'\t', np.sum(peakMAP), int(numCar[0]), '\tAE: ', abs(fark))

            upsampler = transforms.Compose([transforms.ToPILImage(), transforms.Resize((im.shape[2] , im.shape[3]))])

            img = im.numpy()[0,]
            img = np.array(img)
            img = (img - img.min()) / (img.max() - img.min())
            plot.imshow(img.transpose((1, 2, 0)))

            M1 = MAP.data.cpu().contiguous().numpy().copy()
            M1_norm = (M1[0, ] - M1[0, ].min()) / (M1[0, ].max() - M1[0, ].min())
            a = upsampler(torch.Tensor(M1_norm))
            a = np.uint8(cm_jet(np.array(a)) * 255)
            if batch_idx > 0:
                from PIL import Image
                ima = Image.fromarray(a)
                peakMAP = np.uint8(np.array(peakMAP) * 255)
                peakI = Image.fromarray(peakMAP).convert("RGB")
                peakI = peakI.resize((1280,720))
                ima.save("res1/heatmap-" + str(batch_idx) + ".bmp")
                peakI.save("res1/peakmap-" + str(batch_idx) + ".bmp")
                # print(peakI.size)
                # print(ima.size)
                # plot.imshow(a)
                # plot.show()

                # plot.imshow(peakMAP)
                # plot.show()

        print('MAE:', gi / ind)
        print('RMSE:', math.sqrt(gRi/ind))




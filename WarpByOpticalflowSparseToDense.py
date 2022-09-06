import cv2
import numpy as np
import torch

def WarpByOpticalflowSparseToDense(from_image, to_image, device):
    ''' from_image : torch.Tensor ,to_image : torch.Tensor '''

    #@torch.Tensor to ndarray
    nd_from_image = from_image.detach().to('cpu')
    nd_to_image   = to_image.detach().to('cpu')

    sq_from_image = nd_from_image.squeeze().permute(1,2,0).numpy()
    sq_to_image   = nd_to_image.squeeze().permute(1,2,0).numpy()

    gray_from_image = cv2.cvtColor(sq_from_image, cv2.COLOR_BGR2GRAY)
    gray_to_image   = cv2.cvtColor(sq_to_image,   cv2.COLOR_BGR2GRAY)

    from_image = np.uint8(gray_from_image * 255)
    to_image = np.uint8(gray_to_image *255)
    
    flow = cv2.optflow.calcOpticalFlowSparseToDense(from_image, to_image, flow=None, grid_step=3, sigma=0.5)
    h, w, c = flow.shape

    flow_u = flow[:,:,0]
    flow_v = flow[:,:,1]
    index_u = np.arange(w)
    index_v = np.arange(h)

    to_image_hat = sq_to_image.copy()

    sub_index_v = np.ones(w, int)
    for v in index_v:
        # 以下，画像1行分の処理．
        sub_u = index_u + flow_u[v]
        # sub_v = index_v + flow_v[v]
        sub_v = np.full(w, v) + flow_v[v] 

        sub_u = np.where( (w <= sub_u) | (sub_u <= 0), 0, sub_u).astype(int).tolist()
        sub_v = np.where( (h <= sub_v) | (sub_v <= 0), 0, sub_v).astype(int).tolist()

        sub_u = np.nan_to_num(sub_u)
        sub_v = np.nan_to_num(sub_v)

        sub_index_v = np.full(w, v)
        to_image_hat[sub_v, sub_u, :] = sq_from_image[sub_index_v, index_u]
        
    to_image_hat = np.expand_dims(to_image_hat, 0)
    torch_to_image_hat = torch.from_numpy(to_image_hat.astype(np.float32)).clone().permute(0,3,1,2).to(device)

    return torch_to_image_hat

def main():
    pass

if __name__ == '__main__':
    main()
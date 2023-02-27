import torch
import torch.nn.functional as F
import numpy as np


def numpy_to_torch(a: np.ndarray):
    return torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0)


def torch_to_numpy(a: torch.Tensor):
    return a.squeeze(0).permute(1,2,0).numpy()


def sample_patch_transformed(im, pos, scale, image_sz, transforms, is_mask=False):
    """Extract transformed image samples.
    args:
        im: Image.
        pos: Center position for extraction.
        scale: Image scale to extract features from.
        image_sz: Size to resize the image samples to before extraction.
        transforms: A set of image transforms to apply.
    """

    # Get image patche  得到采样的图像块和对应的坐标
    im_patch, _ = sample_patch(im, pos, scale*image_sz, image_sz, is_mask=is_mask)  #[1，3，576，576]

    # Apply transforms 应用transforms
    im_patches = torch.cat([T(im_patch, is_mask=is_mask) for T in transforms])  #[1,3,288,288]

    return im_patches

#在某一中心点处，采样一定大小和尺度的图像块
def sample_patch_multiscale(im, pos, scales, image_sz, mode: str='replicate', max_scale_change=None):
    """Extract image patches at multiple scales. 以多个比例提取图像patches
    args:
        im: Image.      [1,3,1080,1920]
        pos: Center position for extraction.  新样本的中心位置
        scales: Image scales to extract image patches from.   缩放比例 1.97
        image_sz: Size to resize the image samples to .  将图片resize到 [288，288]
        mode: how to treat image borders: 'replicate' (default), 'inside' or 'inside_major'   和train一样为inside_major
        max_scale_change: maximum allowed scale change when using 'inside' and 'inside_major' mode .使用“inside”和“inside_major”模式时允许的最大刻度变化
    """
    if isinstance(scales, (int, float)):
        scales = [scales]
    # Get image patches
    patch_iter, coord_iter = zip(*(sample_patch(im, pos, s*image_sz, image_sz, mode=mode,
                                                max_scale_change=max_scale_change) for s in scales))
    im_patches = torch.cat(list(patch_iter))       #patches块[1,3,288,288]
    patch_coords = torch.cat(list(coord_iter))     #patches块坐标

    return  im_patches, patch_coords

#得到采样的图像块和对应的坐标
def sample_patch(im: torch.Tensor, pos: torch.Tensor, sample_sz: torch.Tensor, output_sz: torch.Tensor = None,
                 mode: str = 'replicate', max_scale_change=None, is_mask=False):
    """Sample an image patch.

    args:
        im: Image
        pos: center position of crop  裁剪的中心点坐标
        sample_sz: size to crop
        output_sz: size to resize to
        mode: how to treat image borders: 'replicate' (default), 'inside' or 'inside_major'
        max_scale_change: maximum allowed scale change when using 'inside' and 'inside_major' mode
    """

    # if mode not in ['replicate', 'inside']:
    #     raise ValueError('Unknown border mode \'{}\'.'.format(mode))

    # copy and convert
    posl = pos.long().clone()

    pad_mode = mode   #mode='replicate'

    # Get new sample size if forced inside the image  不进入
    if mode == 'inside' or mode == 'inside_major':
        pad_mode = 'replicate'
        im_sz = torch.Tensor([im.shape[2], im.shape[3]])
        shrink_factor = (sample_sz.float() / im_sz)
        if mode == 'inside':
            shrink_factor = shrink_factor.max()
        elif mode == 'inside_major':
            shrink_factor = shrink_factor.min()
        shrink_factor.clamp_(min=1, max=max_scale_change)
        sample_sz = (sample_sz.float() / shrink_factor).long()

    # Compute pre-downsampling factor
    if output_sz is not None:
        #采样大小相对于输出大小的倍数
        resize_factor = torch.min(sample_sz.float() / output_sz.float()).item()
        df = int(max(int(resize_factor - 0.1), 1))
    else:
        df = int(1)

    sz = sample_sz.float() / df     # new size 新的output size

    # Do downsampling
    if df > 1:
        os = posl % df              # offset
        posl = (posl - os) / df     # new position
        im2 = im[..., os[0].item()::df, os[1].item()::df]   # downsample
    else:
        im2 = im

    # compute size to crop 四舍五入取整
    szl = torch.max(sz.round(), torch.Tensor([2])).long()

    # Extract top and bottom coordinates 根据中心点和尺寸求取左上角和右下角坐标
    tl = posl - (szl - 1) / 2
    br = posl + szl/2 + 1

    # Shift the crop to inside 不进入
    if mode == 'inside' or mode == 'inside_major':
        im2_sz = torch.LongTensor([im2.shape[2], im2.shape[3]])
        shift = (-tl).clamp(0) - (br - im2_sz).clamp(0)
        tl += shift
        br += shift

        outside = ((-tl).clamp(0) + (br - im2_sz).clamp(0)) // 2
        shift = (-tl - outside) * (outside > 0).long()
        tl += shift
        br += shift

        # Get image patch
        # im_patch = im2[...,tl[0].item():br[0].item(),tl[1].item():br[1].item()]

    # Get image patch  填充
    pad = (-tl[1].int().item(), br[1].int().item() - im2.shape[3],
           -tl[0].int().item(), br[0].int().item() - im2.shape[2])

    if not is_mask:
        im_patch = F.pad(im2, pad, pad_mode)
    else:
        im_patch = F.pad(im2, pad)

    # Get image coordinates 获得image patch对应的坐标
    patch_coord = df * torch.cat((tl, br)).view(1,4)

    if output_sz is None or (im_patch.shape[-2] == output_sz[0] and im_patch.shape[-1] == output_sz[1]):
        return im_patch.clone(), patch_coord

    # Resample 将image patch插值成output_sz大小
    if not is_mask:
        im_patch = F.interpolate(im_patch, output_sz.long().tolist(), mode='bilinear')
    else:
        im_patch = F.interpolate(im_patch, output_sz.long().tolist(), mode='nearest')

    return im_patch, patch_coord  #返回采样的图像块和对应的左上角和右下角坐标

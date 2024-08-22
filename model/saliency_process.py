import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def get_image():
    image = torch.randn(1, 3, 224, 224)
    return image


def get_salience_map():
    salience_map = torch.randn(1, 1, 224, 224)
    return salience_map


def get_index(image_size=224, patch_size=16):
    num_patches = (image_size // patch_size) ** 2
    return torch.arange(num_patches)


def batch_index_select(salience_map, idx):
    image_size = 224
    num_blocks = 14

    block_labels = np.arange(1, num_blocks ** 2 + 1).reshape((num_blocks, num_blocks))
    saliency_map = np.array(salience_map)

    saliency_map = saliency_map.astype(float) / 255.0

    attention_map = np.zeros_like(saliency_map)
    attention_map[saliency_map > 0.5] = 1.0  # 设定一个阈值，大于该阈值的像素设为1，表示高注意力区域



    block_saliency_values = []
    for i in range(num_blocks):
        for j in range(num_blocks):
            block_saliency_values.append((block_labels[i, j], saliency_map[0, 0, i, j]))

    block_saliency_values.sort(key=lambda x: x[1], reverse=True)

    top_blocks = [block[0] for block in block_saliency_values[:98]]

    top_blocks.sort()

    return top_blocks


def patch_flat(patch):

    sub_patch1 = patch[:, :, :8, :8]
    sub_patch2 = patch[:, :, :8, 8:]
    sub_patch3 = patch[:, :, 8:, :8]
    sub_patch4 = patch[:, :, 8:, 8:]


    flattened_sub_patch1 = sub_patch1.reshape(1, -1)
    flattened_sub_patch2 = sub_patch2.reshape(1, -1)
    flattened_sub_patch3 = sub_patch3.reshape(1, -1)
    flattened_sub_patch4 = sub_patch4.reshape(1, -1)

    return flattened_sub_patch1,flattened_sub_patch2,flattened_sub_patch3,flattened_sub_patch4


def patch_down(patch):

    downsampled_patch = torch.nn.functional.interpolate(patch, scale_factor=0.5, mode='bilinear', align_corners=False)

    downsampled_patch_flat = downsampled_patch.reshape(-1)
    return downsampled_patch_flat




def patch_process(image, salience_map):
    image_size = image.size(2)
    patch_size = 16
    idx = get_index(image_size, patch_size)
    idx_select = batch_index_select(salience_map, idx)
    # print("idx_select：",idx_select)
    patches = []
    for i in range(len(idx)):

        patch_idx = idx[i].item()
        row = patch_idx // (image_size // patch_size)
        col = patch_idx % (image_size // patch_size)
        patch = image[:, :, row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size]

        if i+1 in idx_select:
            processed_patch = patch_flat(patch)

            patches.append(processed_patch[0])
            patches.append(processed_patch[1])
            patches.append(processed_patch[2])
            patches.append(processed_patch[3])
        else:
            processed_patch = patch_down(patch)
            processed_patch = processed_patch.unsqueeze(0)

            patches.append(processed_patch)




    concatenated_patches = torch.cat(patches, dim=0).unsqueeze(0)  # torch.Size([1, 490, 192])
    # concatenated_patches = torch.cat(patches, dim=0)  #torch.Size([490, 192])
    return concatenated_patches

def sal_process(image,salience_map):
    B = image.size(0)
    out = []
    for i in range(B):
        current_image = image[i].unsqueeze(0)
        current_sal = salience_map[i].unsqueeze(0)
        processed_patches = patch_process(current_image, current_sal)
        out.append(processed_patches)
        # print("Processed patches shape:", i, ":", processed_patches.shape)
    out = torch.cat(out, dim=0)
    # print("out", out.shape)
    return out


import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import numpy as np

mask_generator = None

def initiate_sam(sam_checkpoint, model_type):
    global mask_generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)


def get_common_mask(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    mm = sorted_anns[0]['segmentation']
    final_msk = np.zeros((mm.shape[0], mm.shape[1]))
    for j, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        final_msk += np.where(m == True, j+1 ,0)
    return final_msk.astype(int)


def SAM_segmentation_fn(image):
    image = image.astype('uint8')
    masks = mask_generator.generate(image)
    common_mask = get_common_mask(masks)
    print(image.shape)
    print(common_mask.shape)
    print(np.unique(common_mask))
    return common_mask


# Load image ====================================================================================================
# import keras.utils as image
#
# img_path = "/home/jaydeep/dev/Reg/SAM-LIME/cat_mouse.jpg"
# img = image.load_img(img_path, target_size=(299, 299))
# # img = image.load_img(img_path)
# img = image.img_to_array(img)
#
# segmentation_fn_img = img.copy()
# print(segmentation_fn_img.shape)
#
# common_mask = SAM_segmentation_fn(segmentation_fn_img)
#
# from skimage.segmentation import mark_boundaries
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10,10))
# plt.imshow(mark_boundaries(pic, common_mask))
# plt.axis('off')
# plt.show()

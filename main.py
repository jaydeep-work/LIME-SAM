import lime_image
import keras.utils as image
from keras.applications import inception_v3 as inc_net
import numpy as np
import matplotlib.pyplot as plt
import sam

# Initiate classification model ==================================================================================
inet_model = inc_net.InceptionV3()


# Load image ====================================================================================================
img_path = "/home/jaydeep/dev/Reg/SAM-LIME/cat_mouse.jpg"
img = image.load_img(img_path, target_size=(299, 299))
img = image.img_to_array(img)

segmentation_fn_img = img.copy()
print(segmentation_fn_img.shape)

img = np.expand_dims(img, axis=0)
img = inc_net.preprocess_input(img)
print(img.shape)

# segmentation_fn_img = cv2.imread('/content/cat_mouse.jpg')
# segmentation_fn_img = cv2.cvtColor(segmentation_fn_img, cv2.COLOR_BGR2RGB)

# initiate LIME =================================================================================================
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(img[0].astype('double'), inet_model.predict, top_labels=5, hide_color=0, num_samples=500,
                                         segmentation_fn=sam.SAM_segmentation_fn,
                                         segmentation_fn_img=segmentation_fn_img)


# post explanation work =======================================================================================
from skimage.segmentation import mark_boundaries

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=1000, hide_rest=False, min_weight=0.1)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()

#Select the same class explained on the figures above.
ind =  explanation.top_labels[0]
#Map each explanation weight to the corresponding superpixel
dict_heatmap = dict(explanation.local_exp[ind])
heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
#Plot. The visualization makes more sense if a symmetrical colorbar is used.
plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
plt.colorbar()
plt.show()
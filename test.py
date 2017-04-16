# import the necessary packages
from keras.preprocessing import image as image_utils
from imagenet_utils import decode_predictions
from imagenet_utils import preprocess_input
from vgg16 import VGG16
from resnet50 import ResNet50
import numpy as np
import argparse
import cv2
from keras.models import Model



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())
 
# load the original image via OpenCV so we can draw on it and display
# it to our screen later
orig = cv2.imread(args["image"])

# load the input image using the Keras helper utility while ensuring
# that the image is resized to 224x224 pxiels, the required input
# dimensions for the network -- then convert the PIL image to a
# NumPy array
print("[INFO] loading and preprocessing image...")
image = image_utils.load_img(args["image"], target_size=(224, 224))
image = image_utils.img_to_array(image)

# our image is now represented by a NumPy array of shape (3, 224, 224),
# but we need to expand the dimensions to be (1, 3, 224, 224) so we can
# pass it through the network -- we'll also preprocess the image by
# subtracting the mean RGB pixel intensity from the ImageNet dataset
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)


# load the ResNet50 network
print("[INFO] loading network...")
model = ResNet50(weights="imagenet")

# classify the image
print("[INFO] classifying image...")
# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_outputs = [layer.output for layer in model.layers]
viz_model = Model(input=model.input, output=layer_outputs)
preds = model.predict(image)

features = viz_model.predict(image)
pool=features[174]


P = decode_predictions(preds,top=10)

(imagenetID, label1, prob)=P[0][0];

#writing embeddings to TSV file
print(pool.shape)
with open("globel_pool.tsv","w") as fp:
	fp.write("#\tFeatures (Channels)\n")
	for (i, val) in enumerate(pool[0][0][0]):
		fp.write(str(i+1))
		fp.write("\t")
		fp.write(str(val))
		fp.write("\n")
	fp.close()


# loop over the predictions and display the rank-5 predictions +
# probabilities to our terminal
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
	print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

#print('Predicted:', decode_predictions(preds, top=5)[0])
'''
cv2.putText(orig, "Label: {}".format(label1), (10, 30),
	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)
'''

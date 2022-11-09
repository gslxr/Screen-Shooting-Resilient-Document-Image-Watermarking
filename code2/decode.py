import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants


def main():
    model = "./model/cnki/cnki"  # load the model
    save_dir = "./recovered_image/cnki/cnki_factor_1.0"  # load the recovered image set, recovered images are from the photographs.

    sess = tf.InteractiveSession(graph=tf.Graph())
    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], model)

    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_encoded_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs[
        'decoded'].name
    output_secret = tf.get_default_graph().get_tensor_by_name(output_secret_name)

    width = height = 400
    count_total = 0
    count_single = 0
    # 100
    secret_binary = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                     1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                     0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1]
    size = (width, height)

    image_encoded_list = glob.glob(save_dir + '/*')
    for image_idx in range(len(image_encoded_list)):
        # encoded_image
        encode_image = cv2.imread(filename=image_encoded_list[image_idx], flags=cv2.IMREAD_UNCHANGED)
        encode_image = cv2.cvtColor(encode_image, cv2.COLOR_BGR2RGB)
        encoded_image = np.array(cv2.resize(encode_image, size), dtype=np.float32) / 255.

        hidden_image = (encoded_image * 255.).astype(np.uint8)  # hidden_image2
        encoded_image_2 = np.array(cv2.resize(hidden_image, size), dtype=np.float32) / 255.

        feed_dict = {input_encoded_image: [encoded_image_2]}
        secret_extract = sess.run([output_secret], feed_dict=feed_dict)[0][0]

        for i in range(len(secret_extract)):
            if secret_binary[i] == int(secret_extract[i]):
                count_single += 1
                count_total += 1

        accuracy_single = (count_single / 100.0) * 100
        count_single = 0
        print("Document image{0} accuracy is:{1}%".format(image_encoded_list[image_idx], accuracy_single))

    accuracy_total = count_total / len(image_encoded_list)  # len(image_encoded_list)
    print("===================================")
    print("Total document image accuracy is:{0}%".format(accuracy_total))


if __name__ == "__main__":
    main()

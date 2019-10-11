import os

BASE_PATH = '/'.join(os.getcwd().split('/')[:-1]) # Using ubuntu machine may require removing this -1
face_describer_input_tensor_names = ['img_inputs:0']
face_describer_output_tensor_names = ['resnet_v1_50/E_BN2/Identity:0']
face_describer_device = '/gpu:0'
face_describer_model_fp = '{}/tf-insightface/pretrained/insightface_without_dropout_resnet.pb'.format(BASE_PATH)
face_describer_tensor_shape = (112, 112)
face_describer_drop_out_rate = 0.1
test_img_fp = '{}/tf-insightface/tests/test.jpg'.format(BASE_PATH)

face_similarity_threshold = 100#800
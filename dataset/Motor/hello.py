from flask import Flask, request, redirect, url_for, jsonify
import os
import numpy as np
import tensorflow as tf
import base64
from PIL import Image

app = Flask(__name__)

# imagePath = '/tmp/bicycle2.jpg'                                      # 추론을 진행할 이미지 경로
modelFullPath = '/tmp/output_graph.pb'  # 읽어들일 graph 파일 경로
labelsFullPath = '/tmp/output_labels.txt'  # 읽어들일 labels 파일 경로


def create_graph():
    """저장된(saved) GraphDef 파일로부터 graph를 생성하고 saver를 반환한다."""
    # 저장된(saved) graph_def.pb로부터 graph를 생성한다.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')





@app.route('/upload', methods=['POST'])
def upload_file():
    # f = request.files['the_file']
    # f.save('/tmp' + secure_filename(f.filename))
    # Todo 이미지파일이 들어오면 해당 파일을 RUN_INFERENCE 함수에 인자로 전달
    # 2. 해당 함수에서 분석한뒤 결과로 나오는 라벨 데이터를 반환 (Run inference)
    # 3. 반환된 값을 jsonify로 클라이언트에 리턴
    # logo = Image('tmp/a.jpg')
    # logo.save('tmp/1.jpg')
    # image = open('/tmp/car.gif', 'rb')  # open binary file in read mode
    # image_read = image.read()
    image_read = request.form['image']
    print(image_read)
    #  image_64_encode = base64.b64encode(image_read)
    #  print('convert s')
    #
    #  image_64_decode = base64.b64decode(image_64_encode)
    #  image_result = open('deer_decode.gif', 'wb')  # create a writable image and write the decoding result
    #  image_result.write(image_64_decode)
    #  print('decode s')
    # # image_pass = open('deer_decode.gif')
    # # print('binary s')
    #  print(image_result)
    #  run_inference_on_image('deer_decode.gif')
    list = [
        {
            "rec_create_date": "12 Jun 2016",
            "rec_dietary_info": "nothing",
            "rec_dob": "01 Apr 1988",
            "rec_first_name": "New",
            "rec_last_name": "Guy",
        },
        {
            "rec_create_date": "1 Apr 2016",
            "rec_dietary_info": "Nut allergy",
            "rec_dob": "01 Feb 1988",
            "rec_first_name": "Old",
            "rec_last_name": "Guy",
        },
    ]
    return jsonify(list)


# @app.route('/')
#
# def run_inference_on_image(image_data):
#     # answer = None
#
#     # if not tf.gfile.Exists(imagePath):
#     #   tf.logging.fatal('File does not exist %s', imagePath)
#     #  return answer
#
#     # image_data = tf.gfile.FastGFile(imagePath, 'rb').read()
#
#     # 저장된(saved) GraphDef 파일로부터 graph를 생성한다.
#     create_graph()
#
#     with tf.Session() as sess:
#         softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
#         predictions = sess.run(softmax_tensor,
#                                {'DecodeJpeg/contents:0': image_data})
#         predictions = np.squeeze(predictions)
#
#         top_k = predictions.argsort()[-5:][::-1]  # 가장 높은 확률을 가진 5개(top 5)의 예측값(predictions)을 얻는다.
#         f = open(labelsFullPath, 'rb')
#         lines = f.readlines()
#         labels = [str(w).replace("\n", "") for w in lines]
#         for node_id in top_k:
#             human_string = labels[node_id]
#             score = predictions[node_id]
#             print('%s (score = %.5f)' % (human_string, score))
#
#         answer = labels[top_k[0]]
#         print('aa')
#         return ''.join(answer)


@app.route('/')
def recieveImage():
    list = [
        {
            "rec_create_date": "12 Jun 2016",
            "rec_dietary_info": "nothing",
            "rec_dob": "01 Apr 1988",
            "rec_first_name": "New",
            "rec_last_name": "Guy",
        },
        {
            "rec_create_date": "1 Apr 2016",
            "rec_dietary_info": "Nut allergy",
            "rec_dob": "01 Feb 1988",
            "rec_first_name": "Old",
            "rec_last_name": "Guy",
        },
    ]
    return jsonify(list)


if __name__ == '__main__':
    #    run_inference_on_image()
    #recieveImage()
    app.run(host='0.0.0.0')

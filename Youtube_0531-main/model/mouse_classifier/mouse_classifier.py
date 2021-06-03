#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class MouseClassifier(object):
    def __init__(
            self,
            model_path='model/mouse_classifier/mouse_classifier_final1.tflite',
            num_threads=1,
            score_th=0.5,
            invalid_value=2,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.score_th = score_th
        self.invalid_value = invalid_value

    def __call__(
            self,
            landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        _index = 0
        _id = ['One', 'Scissor', 'None','six']
        for _ in result[:][0]:
            # if _ > 0.1:
            #     print(f'{_id[_index]},{_:.2f}')
            _index += 1

        result_index = np.argmax(np.squeeze(result))
        # print(f'\n Maxprob_id: {result_index}')

        if np.squeeze(result)[result_index] < self.score_th:
            result_index = self.invalid_value
        # print(f'output {result_index}')

        return result_index

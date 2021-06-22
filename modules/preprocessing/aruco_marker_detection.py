from .preprocessing_blueprint import PreprocessingAlgorithm
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
import os
import cv2.aruco as aruco
import numpy as np
np.set_printoptions(precision=2)
import math


class ArUcoMarkerDetection(PreprocessingAlgorithm):
    Name = "ArUcoMarkerDetection"

    def __init__(self, model_path=""):
        self.model_rvec = load_model(os.path.join(model_path, "model_RVEC.h5"))
        self.model_tvec = load_model(os.path.join(model_path, "model_TVEC.h5"))
        self.mtx, self.dist = self.load_coefficients(os.path.join(model_path, 'calibration_charuco.yml'))

        self.predictions_tvec = np.zeros(3)
        self.predictions_rvec = np.zeros(4)

    def get_output_shapes(self, environment_configuration):
        output_shapes = []
        for obs_shape in environment_configuration["ObservationShapes"]:
            if len(obs_shape) == 3:
                output_shapes.append(self.model_rvec.output_shape[1:])
                output_shapes.append(self.model_tvec.output_shape[1:])
            else:
                output_shapes.append(obs_shape)
        output_shapes = [sum([o[0] for o in output_shapes])]
        output_shapes = [13]
        return output_shapes

    def preprocess_observations(self, decision_steps, terminal_steps):
        if len(decision_steps):
            vector_predictions = None
            for idx, o in enumerate(decision_steps.obs):
                if len(o.shape) == 4:
                    suc, all_tvecs_mean, take_rvec, ids = self.detect_aruco_marker(np.array(o[0]*255, dtype=np.uint8))
                    if suc:
                        self.predictions_tvec = self.model_tvec.predict(np.array([self.rotate(all_tvecs_mean)]))
                        self.predictions_rvec = self.model_rvec.predict(np.array([take_rvec[0]]))
                if len(o.shape) == 2:
                    vector_predictions = o
            predictions = [vector_predictions[0, :3],
                           self.predictions_tvec[0],
                           vector_predictions[0, 6:9],
                           [self.predictions_rvec[0, 3]],
                           self.predictions_rvec[0, :3]]
            new_prediction = np.expand_dims(np.concatenate(predictions, axis=0), axis=0)
            decision_steps.obs = new_prediction
        if len(terminal_steps):
            vector_predictions = None
            for idx, o in enumerate(terminal_steps.obs):
                if len(o.shape) == 4:
                    suc, all_tvecs_mean, take_rvec, ids = self.detect_aruco_marker(np.array(o[0]*255, dtype=np.uint8))
                    if suc:
                        self.predictions_tvec = self.model_tvec.predict(np.array([self.rotate(all_tvecs_mean)]))
                        self.predictions_rvec = self.model_rvec.predict(np.array([take_rvec[0]]))
                if len(o.shape) == 2:
                    vector_predictions = o
            predictions = [vector_predictions[0, :3],
                           self.predictions_tvec[0],
                           vector_predictions[0, 6:9],
                           [self.predictions_rvec[0, 3]],
                           self.predictions_rvec[0, :3]]
            terminal_steps.obs = np.expand_dims(np.concatenate(predictions, axis=0), axis=0)
        return decision_steps, terminal_steps

    def detect_aruco_marker(self, frame):
        # Calculate Camera Matrix
        h,  w = frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
        # Undistort Image
        undistorted_image = cv2.undistort(frame, self.mtx, self.dist, None, new_camera_matrix)
        # Crop Image
        x, y, w, h = roi
        frame = undistorted_image[y:y+h, x:x+w]
        rgb_frame = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Get ArUco Dictionary and Parameters
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
        aruco_parameters = aruco.DetectorParameters_create()
        corners, ids, rejected_img_points = aruco.detectMarkers(frame, aruco_dict, parameters=aruco_parameters)

        if ids is not None:
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, self.mtx, self.dist)
            (rvec-tvec).any()

            max_x = 0
            take_id = None
            for tvec_sing_id in range(tvec.shape[0]):
                if max_x < np.abs(tvec[tvec_sing_id][0][0]):

                    max_x = np.abs(tvec[tvec_sing_id][0][0])
                    take_id = tvec_sing_id

            take_rvec = np.array([[0.00, 0.00, 0.0]])

            all_tvecs = np.zeros(shape=(rvec.shape[0],3))
            for i in range(rvec.shape[0]):
                rvecmid = rvec[i, :, :].copy()
                tvecmid = tvec[i, :, :].copy()

                rot_mat,_ = cv2.Rodrigues(rvec[i][0].copy())
                t0_tvec = np.zeros((4, 4))
                t0_tvec[:3, :3] = rot_mat
                t0_tvec[:4, 3] = [0, 0, 0, 1]
                t0_tvec[:3, 3] = np.transpose(tvec[i][0].copy())

                tvecmid = np.append(tvecmid, 1)
                tvec_num_t0 = np.dot(t0_tvec, tvecmid)

                if ids[i] == 5:
                    tvec_num_t0[0] = tvec_num_t0[0] + 0.031 #- 0.027 für links

                else:
                    tvec_num_t0[2] = tvec_num_t0[2] - 0.031 #- 0.027 für links

                T0_tvec_inv = np.linalg.inv(t0_tvec)

                tvecmid_new = np.dot(T0_tvec_inv, tvec_num_t0)

                all_tvecs[i] = tvecmid_new[0:3]

                if i == take_id:
                    take_rvec = rvecmid

            all_tvecs_mean = np.mean(all_tvecs, axis=0)

            aruco.drawAxis(rgb_frame, self.mtx, self.dist, take_rvec, all_tvecs_mean, 0.1) #Draw axis
            aruco.drawDetectedMarkers(rgb_frame, corners) #Draw a square around the marker

            cv2.imshow("bla", rgb_frame)
            cv2.waitKey(1)

            return [True, all_tvecs_mean, take_rvec, ids]
        else:
            return False, None, None, None

    def load_coefficients(self, path):
        """Loads camera matrix and distortion coefficients."""
        # FILE_STORAGE_READ
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        camera_matrix = cv_file.getNode('K').mat()
        dist_matrix = cv_file.getNode('D').mat()

        cv_file.release()
        return [camera_matrix, dist_matrix]

    def rotate(self, array):
        newarray = array.copy()
        newarray[0] = array[2]
        newarray[2] = array[0]
        return newarray
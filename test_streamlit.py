import streamlit as st
import cv2
import numpy as np
from camera_input_live import camera_input_live

pro_sim_matrix = np.array([[0, 0.90822864, 0.008192], [0, 1, 0], [0, 0, 1]], dtype=np.float16)
deu_sim_matrix =  np.array([[1, 0, 0], [1.10104433,  0, -0.00901975], [0, 0, 1]], dtype=np.float16)
tri_sim_matrix = np.array([[1, 0, 0], [0, 1, 0], [-0.15773032,  1.19465634, 0]], dtype=np.float16)

rgb_matrix = np.array(
        [[ 2.85831110e+00, -1.62870796e+00, -2.48186967e-02],
        [-2.10434776e-01,  1.15841493e+00,  3.20463334e-04],
        [-4.18895045e-02, -1.18154333e-01,  1.06888657e+00]]
        )

lms_matrix = np.array(
        [[0.3904725 , 0.54990437, 0.00890159],
        [0.07092586, 0.96310739, 0.00135809],
        [0.02314268, 0.12801221, 0.93605194]]
        )

def project():
    st.title("color blind project")

    img_file_buffer = st.camera_input("Take a photo")

    if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
    
        lms_img = np.tensordot(frame,  lms_matrix, axes=([2],[1]))
        transformed = np.tensordot(lms_img, pro_sim_matrix, axes=([2],[1]))
        pro_img = np.tensordot(transformed, rgb_matrix, axes=([2],[1])).astype(np.uint8)

        transformed = np.tensordot(lms_img, deu_sim_matrix, axes=([2],[1]))
        deu_img = np.tensordot(transformed, rgb_matrix, axes=([2],[1])).astype(np.uint8)

        transformed = np.tensordot(lms_img, tri_sim_matrix, axes=([2],[1]))
        tri_img = np.tensordot(transformed, rgb_matrix, axes=([2],[1])).astype(np.uint8)

        final_frame = np.concatenate((frame, pro_img), axis=1)
        deu_tri = np.concatenate((tri_img,deu_img), axis=1)
        final_frame = np.concatenate((final_frame, deu_tri), axis=0)

        pos_x, pos_y = 20, 60
        cv2.putText(final_frame, "Original", (pos_x, pos_y), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255))
        cv2.putText(final_frame, "Protagonia", (pos_x+frame.shape[1], pos_y), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255))
        cv2.putText(final_frame, "Tritanopia", (pos_x, pos_y+frame.shape[0]), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255))
        cv2.putText(final_frame, "Deuteranopia",(pos_x+frame.shape[1], pos_y+frame.shape[0]), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255))
        
        st.image(final_frame)

    st.write("Introduction ")
    st.write("hi ")

if __name__ == '__main__':
    project()

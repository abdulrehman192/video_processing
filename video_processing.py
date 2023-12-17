# app/video_processing
import cv2
import mediapipe as mp
import numpy as np
import skimage as skimage
from skimage import color, feature, filters
import warnings
from moviepy.editor import VideoFileClip

target_width = 224
target_height = 224


def extract_facial_features(img):
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        result = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Initialize an empty array to store facial embeddings
        facial_embeddings_list = []
        gabor_list = []
        blur_list = []
        lbp_list = []

        if result.detections:
            for detection in result.detections:
                # Extract features for each detected face
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)

                # Extract face with extra region
                extra_margin = 40
                x, y, width, height = bbox
                x -= extra_margin
                y -= extra_margin
                width += 2 * extra_margin
                height += 2 * extra_margin

                # Ensure coordinates are within image boundaries
                x = max(0, x)
                y = max(0, y)
                width = min(img.shape[1] - x, width)
                height = min(img.shape[0] - y, height)

                # Extract face with extra region
                face_img = img[y:y + height, x:x + width]

                face_img_resized = cv2.resize(face_img, (224, 24))
                face_img = cv2.resize(face_img, (100, 100))

                # Normalize the face image
                face_img_normalized = face_img_resized / 255.0
                face_img = face_img / 255.0

                face_img_normalized = (face_img_normalized * 255).astype(np.uint8)
                face_img = (face_img * 255).astype(np.uint8)

                facial_embeddings = extract_glcm_feature(face_img_normalized)
                gabor = extract_gabor_features(face_img_normalized)
                blur = calculate_blur_score(face_img_normalized)
                lbp = extract_lbp_feature_from_frame(face_img_normalized)
                # print(lbp.shape)

                # Append the facial embeddings to the list
                facial_embeddings_list.append(facial_embeddings)
                gabor_list.append(gabor)
                # print(blurriness)
                blur_list.append(blur)
                lbp_list.append(lbp)

            faces_list = convert_to_same(facial_embeddings_list, 5)
            gabors = convert_to_same(gabor_list, 4)
            b = convert_to_same(blur_list, 1)
            l = convert_to_same(lbp_list, 10)
            return np.concatenate((np.array(faces_list), np.array(gabors), np.array(l), np.array(b)), axis=1), face_img

        else:
            return None, None


def convert_to_same(input_array, num, faces=3):
    length = len(input_array)
    zero_array = np.zeros(num)

    if length >= faces:
        return input_array[:faces]
    else:
        return input_array + [zero_array] * (faces - length)


def extract_glcm_feature(frame):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the GLCM (Gray Level Co-occurrence Matrix)
    distances = [1]  # Distance between pixel pairs
    angles = [0]  # Angle in radians (0 degrees)

    glcm = skimage.feature.graycomatrix(gray_frame, distances=distances, angles=angles, symmetric=True, normed=True)

    # Calculate GLCM properties (contrast, dissimilarity, homogeneity, energy, correlation)
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    glcm_features = np.zeros((len(distances), len(angles), len(properties)))

    for d in range(len(distances)):
        for a in range(len(angles)):
            for p in range(len(properties)):
                glcm_features[d, a, p] = skimage.feature.graycoprops(glcm, properties[p])[0, 0]

    # Flatten the GLCM feature matrix into a one-dimensional vector
    feature_vector = glcm_features.flatten()

    return feature_vector


def extract_lbp_feature_from_frame(frame):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate LBP feature
    lbp_radius = 1
    lbp_points = 8 * lbp_radius
    lbp_histogram = feature.local_binary_pattern(gray_frame, lbp_points, lbp_radius, method="uniform")

    # Calculate the histogram of LBP feature
    lbp_hist, _ = np.histogram(lbp_histogram.flat, bins=np.arange(0, lbp_points + 3), range=(0, lbp_points + 2))

    # Normalize the histogram (optional)
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-8)

    return lbp_hist


def calculate_blur_score(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the Laplacian for edge detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Calculate the variance of the Laplacian to identify blur
    blur_score = np.var(laplacian)

    return np.array([blur_score])


def extract_gabor_features(image, frequency=0.1, theta=0.0):
    gray_image = color.rgb2gray(image)

    # Define Gabor filter parameters
    sigma = 1.0
    lambd = 2.0  # Wavelength of the sinusoidal factor
    psi = 0.0  # Phase offset
    gamma = 0.5  # Spatial aspect ratio

    # Create Gabor filter
    gabor_filter_real, gabor_filter_imag = filters.gabor(gray_image, frequency, theta=theta, sigma_x=sigma,
                                                         sigma_y=sigma, n_stds=3)

    # Calculate magnitude and phase of Gabor response
    gabor_magnitude = np.sqrt(gabor_filter_real ** 2 + gabor_filter_imag ** 2)
    gabor_phase = np.arctan2(gabor_filter_imag, gabor_filter_real)

    # Extract statistical features from magnitude and phase
    mean_magnitude = np.mean(gabor_magnitude)
    std_magnitude = np.std(gabor_magnitude)
    mean_phase = np.mean(gabor_phase)
    std_phase = np.std(gabor_phase)

    # Combine features into a single vector
    feature_vector = np.array([mean_magnitude, std_magnitude, mean_phase, std_phase])

    return feature_vector


def resize_frame(input_frame):
    # Ensure the target dimensions maintain the original aspect ratio
    aspect_ratio = input_frame.shape[1] / input_frame.shape[0]
    if target_width / aspect_ratio < target_height:
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(new_height * aspect_ratio)

    # Resize the frame
    resized_frame = cv2.resize(input_frame, (new_width, new_height))

    return resized_frame


def is_valid_frame(frame):
    return frame.size != 0 and frame.nbytes != 0


def process_video(video_path, max_frames=120, frames_per_set=60):
    facial_features = []
    lbp_features = []
    glcm_features = []
    gabor_filters = []
    faces = []
    i = 0

    try:
        # Open the video file using moviepy
        video_clip = VideoFileClip(video_path)
        # Get the number of frames in the video
        total_frames = int(video_clip.fps * video_clip.duration)

        # Calculate the number of frames to extract (minimum of max_frames and total_frames)
        num_frames_to_extract = min(max_frames, total_frames)

        # Calculate the number of sets based on frames_per_set
        num_sets = num_frames_to_extract // frames_per_set

        if total_frames < frames_per_set:
            num_sets = 1  # Extract all frames in a single set

        for set_index in range(num_sets):
            # Calculate frame indices for the current set
            start_frame_index = set_index * frames_per_set
            end_frame_index = start_frame_index + frames_per_set

            # Extract frames for the current set
            for frame_index in range(start_frame_index, end_frame_index):

                # Read the frame at the specified index
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    frame = video_clip.get_frame(frame_index / video_clip.fps)

                # Check if the frame is valid
                if not is_valid_frame(frame):
                    continue

                facial_feature, face = extract_facial_features(frame)

                if facial_feature is not None:
                    resized_frame = cv2.resize(frame, (target_width, target_height))
                    resized_frame = resized_frame / 255.0
                    resized_frame = (resized_frame * 255).astype(np.uint8)

                    faces.append(face)

                    facial_feature = facial_feature.flatten()
                    facial_features.append(facial_feature)

                    lbp = extract_lbp_feature_from_frame(resized_frame)
                    lbp_features.append(lbp)

                    glcm = extract_glcm_feature(resized_frame)
                    glcm_features.append(glcm)

                    gabor = extract_gabor_features(resized_frame)
                    gabor_filters.append(gabor)

                # Close the video clip
        video_clip.close()
    except Exception as e:
        print(f"Error processing {video_path}: {e}")

    return np.concatenate((
        np.array(facial_features),
        np.array(lbp_features),
        np.array(glcm_features),
        np.array(gabor_filters)
    ), axis=1), np.array(faces)

from setuptools import setup, find_packages

setup(
    name='video_processing',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'cv2',
        'mediapipe',
        'numpy',
        'skimage',
        'moviepy',
    ],
)

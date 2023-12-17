from setuptools import setup, find_packages

setup(
    name='video_processing',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'opencv-contrib-python==4.8.1.78',
        'mediapipe==0.10.9',
        'numpy==1.26.2',
        'scikit-image==0.18.3',
        'moviepy==1.0.3',
    ],
)

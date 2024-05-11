from setuptools import setup, find_packages

setup(
    name="transnetv2",
    version="1.0.0",
    install_requires=[
        "ffmpeg-python",
        "opencv-contrib-python",
        "tqdm",
        "numpy"
    ],
    entry_points={
        "console_scripts": [
            "TRANSNET = transnetv2.cli.run:main",
        ]
    },
    packages=find_packages(),
    package_dir={"transnetv2": "./transnetv2"},
    package_data={"transnetv2": ["./checkpoints/*"]},
    zip_safe=True
)

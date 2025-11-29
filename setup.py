from setuptools import find_packages, setup


setup(
    name="image-deblurring",
    version="0.1.0",
    description="Image deblurring training pipeline with Hydra configs",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy==2.1.2",
        "albucore==0.0.21",
        "albumentations==1.4.23",
        "opencv-python==4.10.0.84",
        "scikit-image==0.24.0",
        "scipy==1.14.1",
        "tqdm==4.66.5",
        "pandas==2.2.3",
        "pytorch-msssim==1.0.0",
        "hydra-core==1.3.2",
        "omegaconf==2.3.0",
        "mlflow==2.16.0",
    ],
)

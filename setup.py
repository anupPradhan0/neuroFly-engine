from setuptools import setup, find_packages

setup(
    name="biofly-arcade",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=1.0.0",
        "stable-baselines3>=2.0.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pygame>=2.5.0",
        "opencv-python>=4.8.0",
        "matplotlib>=3.8.0",
        "mujoco>=3.1.0",
    ],
    author="BioFly Team",
    description="BioFly-Arcade Engine: Closed-loop RL simulation with connectome-constrained Drosophila agent",
)

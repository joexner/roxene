FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Install Jupyter
RUN pip install jupyter

ADD roxene /workspace/notebooks/roxene
ADD notebooks /workspace/notebooks/roxene-notebooks

WORKDIR /workspace/notebooks

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

# copy files
ADD scripts /workspace/
ADD weights /weights/
RUN chmod +x /workspace/*.sh
RUN mkdir /mnt/data
RUN mkdir /mnt/pred
RUN pip install nibabel tqdm torch-summary scikit-image==0.19.3

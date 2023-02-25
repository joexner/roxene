FROM tensorflow/tensorflow:devel-gpu AS builder

WORKDIR /tensorflow_src
RUN ["git", "fetch", "origin", "r2.10:r2.10"]
RUN ["git", "checkout", "--depth=1", "r2.10"]
RUN ["bazel", "build", "--config=cuda", "//tensorflow/tools/pip_package:build_pip_package"]
RUN ["./bazel-bin/tensorflow/tools/pip_package/build_pip_package", "/mnt" ]


FROM tensorflow/tensorflow:2.10.1-gpu-jupyter as jupyter

RUN ["pip", "uninstall", "tensorflow"]
RUN --mount=type=bind,target=/mnt,source=/mnt/from=builder ["pip", "install",  "/mnt/tensorflow-*.whl"]

ADD ../../roxene /tf/notebooks/roxene
ADD ../../notebooks /tf/notebooks/roxene-notebooks

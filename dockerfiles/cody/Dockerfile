FROM tensorflow/tensorflow:devel-gpu AS builder

WORKDIR /tensorflow_src
RUN ["git", "fetch", "--depth=1", "origin", "r2.12:t"]
RUN ["git", "checkout", "t"]
RUN ["bazel", "build", "--verbose_failures", "--config=cuda", "//tensorflow/tools/pip_package:build_pip_package"]
RUN ["./bazel-bin/tensorflow/tools/pip_package/build_pip_package", "/mnt" ]


FROM tensorflow/tensorflow:2.12.0rc0-gpu-jupyter

RUN ["pip", "uninstall", "tensorflow"]
RUN --mount=type=bind,target=/mnt,source=/mnt/from=builder ["pip", "install",  "/mnt/tensorflow-*.whl"]

ADD ../../roxene /tf/notebooks/roxene
ADD ../../notebooks /tf/notebooks/roxene-notebooks

# Overview

Instructions for development.

## Development

`docker build --target runtime -t evabyte:runtime .` to build runtime image.

`docker build --target dev -t evabyte:dev .` to build dev image.

`docker run -ti --gpus all -v"$(pwd)/src:/job/src" --entrypoint bash evabyte:dev` to bash into dev image and mount local src directory into image (in read/write mode -- writes inside container are persisted to host).

Or, open project root directory in VSCode and load devcontainer.

## Notes

Various notes and thoughts that occurred that we may want to keep in mind.

1. According to Ed in [this discussion](https://edstem.org/us/courses/77432/discussion/6630668), timing starts when "docker build -t" _starts_, rather than when it ends. This means optimizing build time and keeping on the same base image will be helpful for speed. We should try to find the right balance between setup time (docker build time, model download and load time) and inference time.
2. The base image for the inference runtime doesn't have gcc or clang, which are required by triton.
3. I've asked [here](https://edstem.org/us/courses/77432/discussion/6658204) what machine we'll be running on. GPU type will be especially important.

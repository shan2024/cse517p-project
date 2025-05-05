# Overview

Instructions for development.

## Development

`docker build -t evabyte .` to build image.

`docker run -ti --gpus all -v"$(pwd)/src:/job/src" --entrypoint bash evabyte` to bash into image and mount local src directory into image (in read/write mode -- writes inside container are persisted to host).

## Notes

Various notes and thoughts that occurred that we may want to keep in mind.

1. According to Ed in [this discussion](https://edstem.org/us/courses/77432/discussion/6630668), timing starts when "docker build -t" _starts_, rather than when it ends. This means optimizing build time and keeping on the same base image will be helpful for speed.
2. I've asked [here](https://edstem.org/us/courses/77432/discussion/6658204) what machine we'll be running on. GPU type will be especially important.

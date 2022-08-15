FROM pytorch/pytorch

RUN apt-get update
RUN apt-get install -y --no-install-recommends locales
RUN apt-get install -y python3 python3-pip
RUN echo "ja_JP UTF-8" > /etc/locale.gen
RUN locale-gen

# setup directory
RUN mkdir /car-rotation
ADD . /car-rotation
WORKDIR /car-rotation

# install python packages
RUN pip3 --no-cache-dir install -r requirements.txt

# cleanup
RUN apt-get purge -y --auto-remove cmake
RUN apt-get autoremove -y
RUN rm -rf /root/.cache/
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /car-rotation/src
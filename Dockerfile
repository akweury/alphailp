# Select the base image
#FROM nvcr.io/nvidia/pytorch:21.06-py3
FROM nvcr.io/nvidia/pytorch:23.02-py3
# Select the working directory
WORKDIR  /aILP

# Install Python requirements
COPY ./requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Add cuda

# Add fonts for serif rendering in MPL plots
RUN apt-get update
RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections
RUN apt-get install --yes ttf-mscorefonts-installer
RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime
RUN apt-get install dvipng cm-super fonts-cmu --yes
RUN apt-get install fonts-dejavu-core --yes
RUN pip install opencv-python==4.5.5.64
RUN git clone https://github.com/akweury/alphailp.git
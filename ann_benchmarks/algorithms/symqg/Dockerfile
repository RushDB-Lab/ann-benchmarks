FROM ann-benchmarks

RUN apt-get install -y python-setuptools python-pip
RUN git clone -b baseline https://github.com/RushDB-Lab/LiteQG.git
RUN cd LiteQG/python; pip3 install -r requirements.txt; bash build.sh
RUN python3 -c 'import symphonyqg'

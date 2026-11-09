FROM 565393045996.dkr.ecr.us-west-2.amazonaws.com/cibobase:baseubuntu_2604_numba AS common

RUN apt-get update

ENV PATH=${SW_VOLUME}/bin:${SW_VOLUME}/local/bin:${PATH}
ENV PYTHONPATH=/usr/local/lib/python3.14/dist-packages:/usr/lib/python3/dist-packages
ENV PYTHONPATH=${SW_VOLUME}/local/lib/python${PYTHON_VERSION}/dist-packages:${SW_VOLUME}/lib/python${PYTHON_VERSION}/site-packages:${PYTHONPATH}

RUN --mount=type=bind,target=/tmp/drcm,rw pip install --prefix=${SW_VOLUME} /tmp/drcm

RUN python3 -c 'import drcm'
RUN seasonalDRCMIbra -h

ENTRYPOINT ["seasonalDRCMIbra"]

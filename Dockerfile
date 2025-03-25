FROM python:3.10-slim as base
LABEL maintainer="jnation@lco.global"

# use bash
SHELL ["/bin/bash", "-c"]

# install any security updates
RUN apt-get update && apt-get -y upgrade && apt-get install libgl1 libglib2.0-0 -y

# upgrade pip and install poetry
RUN pip install --upgrade pip && pip install "poetry>=1.4.2"

WORKDIR /datalab

# copy bare minimum needed to install python dependecies with poetry
COPY ./README.md ./pyproject.toml ./poetry.lock ./

RUN pip install gunicorn[gevent]==21.2.0
RUN pip install -r <(poetry export)

# copy everything else
COPY ./ ./

# install our app
RUN pip install .

# collect all static assets into one place: /static
RUN mkdir -p static && python manage.py collectstatic --noinput

ENV PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1


# add a multi-stage build target which also has dev (test) dependencies
# usefull for running tests in docker container
# this won't be included in the final image
# e.g. docker build --target dev .
FROM base as dev

RUN pip install -r <(poetry export --dev)

ENTRYPOINT ["bash"]


# final image
FROM base as prod

# add a non-root user to run the app
RUN useradd appuser

# switch to non-root user
USER appuser

CMD ["gunicorn", "datalab.wsgi", "--bind=0.0.0.0:8080", "--worker-class=gevent", "--workers=4", "--timeout=300"]

EXPOSE 8080/tcp

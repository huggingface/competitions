COMP_IMAGE ?= ghcr.io/stresearch/competitions:latest

CFLAGS += -std=c99 -Wall -O2
LDFLAGS += -lseccomp
.PHONY: quality style test

quality:
	python -m black --check --line-length 119 --target-version py38 .
	python -m isort --check-only .
	python -m flake8 --max-line-length 119

style:
	python -m black --line-length 119 --target-version py38 .
	python -m isort .

docker:
	docker build --build-arg COMP_IMAGE=$(COMP_IMAGE) -t $(COMP_IMAGE) .
	# docker tag competitions:latest $(COMP_IMAGE)
	docker push $(COMP_IMAGE)

test:
	pytest -sv .

sandbox: sandbox.c
	gcc $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm *.so sandbox

pip:
	rm -rf build/
	rm -rf dist/
	make style && make quality
	python setup.py sdist bdist_wheel
	twine upload dist/* --verbose

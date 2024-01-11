CFLAGS += -std=c99 -Wall
.PHONY: quality style test

quality:
	python -m black --check --line-length 119 --target-version py38 .
	python -m isort --check-only .
	python -m flake8 --max-line-length 119

style:
	python -m black --line-length 119 --target-version py38 .
	python -m isort .

docker:
	docker build -t competitions:latest .
	docker tag competitions:latest huggingface/competitions:latest
	docker push huggingface/competitions:latest

test:
	pytest -sv .

socket-kit.so: socket-kit.c
	gcc $(CFLAGS) -shared -fPIC $^ -o $@ -ldl

clean:
	rm *.so

pip:
	rm -rf build/
	rm -rf dist/
	make style && make quality
	python setup.py sdist bdist_wheel
	twine upload dist/* --verbose
.PHONY: clean virtualenv test docker dist dist-upload

clean:
	find . -name '*.py[co]' -delete
	rm -rf *.lock *.dirlock worker-* build *.egg-info .pytest_* coverage-report .coverage*

# 	make -C docs clean


# clean-outputs:
#     rm -rf outputs/ logs/


virtualenv:
	rm -rf venv/
	virtualenv --python python3 --prompt '|> wavenet <| ' venv
	venv/bin/pip install -r requirements.txt
	venv/bin/python setup.py develop
	@echo
	@echo "VirtualENV Setup Complete. Now run: source venv/bin/activate"
	@echo

virtualenv-dev:
	rm -rf venv/
	virtualenv --python python3 --prompt '|> wavenet <| ' venv
	venv/bin/pip install -r requirements-dev.txt
	venv/bin/python setup.py develop
	@echo
	@echo "VirtualENV Setup Complete. Now run: source venv/bin/activate"
	@echo

test:
	@echo "-------------"
	@echo "Running tests"
	@echo "-------------"
	@echo
	python -m pytest \
		-v \
		--cov=temporalnn \
		--cov-report=term \
		--cov-report=html:coverage-report \
		tests/
zipfile:
	find . -not \
		-name env -o \
		-name "*.zip" -o \
		-name tests -o

docker: clean
	docker build -t daish:latest .

install-pre-commit-hook:
	ln -svf ../../util/git-hooks/pre-commit .git/hooks/pre-commit

setup:
	pip install -r requirements-dev.txt
	python setup.py develop

lint:
	@echo "---------------------------"
	@echo "Linting project source code"
	@echo "---------------------------"
	@echo
	flake8 --extend-ignore=E501 --exclude=env/
	@echo

html-docs:
	@echo "---------------------------"
	@echo "Building html documentation"
	@echo "---------------------------"
	@echo
	make -C docs html
	@echo

docstyle:
	pydocstyle

prepare-dbvis:
	sudo apt install git -y
	-git init
	-git add remote origin ssh://git@gitlab.dbvis.de:2222/student-projects/duy-lam.git
	git pull
	git checkout feature/generate-train
	-unzip data/climate.zip /data
	-tar -xf data/data.tar.xz

wheel:
	python setup.py bdist_wheel -d whl

# ! git add remote origin https://github.com/dungthuapps/mts-cnn-xai.git

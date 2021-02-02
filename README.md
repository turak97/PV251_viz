## PV251\_viz

Vizualization project to PV251 seminar. Vizualization of OLS, GLS regression and Cochranne-Orcutt method.

## How to install:
Supposing you have some bash and Python3 in it

Go to a directory, where you whish this a new home

```bash
git clone https://github.com/turak97/PV251_viz.git

cd PV251_viz/

python3 -m venv env

source env/bin/activate

# ensure latest versions for installing packages
python -m pip install --upgrade pip setuptools

python -m pip install -r requirements.txt
```

Then try run the code.
Runs also on Ubuntu subsystem for Windows.

## How to run:
Make sure you are in folder PV251\_viz
```bash
source env/bin/activate

python main.py

# Press Ctrl+C to exit webserver.
```

You can also explore your own datasets. Check
```bash
python main.py --help
```

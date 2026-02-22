import json

with open("G25AIT1078.ipynb", "r") as f:
    nb = json.load(f)

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "!pip install gdown\n",
        "!gdown \"187dSnEGn1g2t1UjSJwqevbW9vjh_sWMT\"\n"
    ]
}

nb["cells"].insert(1, new_cell)

with open("G25AIT1078.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated.")

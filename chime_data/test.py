import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    "savefig.bbox": "tight",
    "savefig.format": "pdf",
})

plt.figure(figsize=(4,3))
plt.text(0.5, 0.5, r"$E = mc^2$", fontsize=24, ha="center", va="center")
plt.axis("off")
plt.savefig("test.pdf")
print('saved test.pdf')

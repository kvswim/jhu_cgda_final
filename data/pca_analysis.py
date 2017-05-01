# Student: Nawaf Al-Dhelaan
# Course: 600.438
# Date: April 7th, 2017

# =========================== Dependencies ===========================

from __future__ import division    # In case you run with python2

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import sys, math

# =========================== Auxilary Functions ===========================

# Standarizes the given data matrix
def standarize(data):
	return (data - np.mean(data)) / np.std(data, ddof=1)
	
# Makes a scatter plot for two PCs. Used in part 3
# I put it in a function to make things neater
def loadings_plot(PC1, PC2, title):
	n = 11
	colors = 2 * np.pi * np.random.rand(n)
	plt.scatter(PC1[asw_indices], PC2[asw_indices], color='black', label="African ancestry in SW USA")
	plt.scatter(PC1[ceu_indices], PC2[ceu_indices], color='blue', label="Utah residents, NW European ancestry")
	plt.scatter(PC1[chb_indices], PC2[chb_indices], color='red', label="Han Chinese in Beijing, China")
	plt.scatter(PC1[chd_indices], PC2[chd_indices], color='pink', label="Chinese in Metropolitan Denver, Colorado")
	plt.scatter(PC1[gih_indices], PC2[gih_indices], color='yellow', label="Gujarati Indians in Houston, Texas")
	plt.scatter(PC1[jpt_indices], PC2[jpt_indices], color='purple', label="Japanese in Tokyo, Japan")
	plt.scatter(PC1[lwk_indices], PC2[lwk_indices], color='orange', label="Luhya in Webuye, Kenya")
	plt.scatter(PC1[mex_indices], PC2[mex_indices], color='beige', label="Mexican ancestry in Los Angeles, California")
	plt.scatter(PC1[mkk_indices], PC2[mkk_indices], color='grey', label="Maasai in Kinyawa, Kenya")
	plt.scatter(PC1[tsi_indices], PC2[tsi_indices], color='cyan', label="Toscani in Italia")
	plt.scatter(PC1[yri_indices], PC2[yri_indices], color='green', label="Yoruba in Ibadan, Nigeria")

	plt.title(title)
	plt.legend()
	plt.show()

# =========================== MAIN ===========================
print("Reading data..")
data = np.genfromtxt(sys.argv[1], delimiter=",", skip_header=1)[:, 1:]
population_data = np.genfromtxt(sys.argv[2], delimiter=",", skip_header=1, usecols=[-1])

# =========================== PART 1 ===========================
print("Standarizing data..")
data = standarize(data)

# =========================== PART 2 ===========================
print("Computing PCAs..")

# Dimensionality reduction over features
pca = PCA()
pca.n_components = 20
X_proj = pca.fit_transform(data)
pca_loadings = X_proj.transpose()	# transpose for easier indexing

# Indices - useful for fast computation
asw_indices = np.where(population_data == 1)
ceu_indices = np.where(population_data == 2)
chb_indices = np.where(population_data == 3)
chd_indices = np.where(population_data == 4)
gih_indices = np.where(population_data == 5)
jpt_indices = np.where(population_data == 6)
lwk_indices = np.where(population_data == 7)
mex_indices = np.where(population_data == 8)
mkk_indices = np.where(population_data == 9)
tsi_indices = np.where(population_data == 10)
yri_indices = np.where(population_data == 11)

loadings_plot(pca_loadings[0], pca_loadings[1], "PC1 loadings vs. PC2 loadings")
loadings_plot(pca_loadings[1], pca_loadings[2], "PC2 loadings vs. PC3 loadings")
loadings_plot(pca_loadings[2], pca_loadings[3], "PC3 loadings vs. PC4 loadings")
loadings_plot(pca_loadings[-2], pca_loadings[-1], "PC8 loadings vs. PC9 loadings")
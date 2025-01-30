import matplotlib.pyplot as plt
from matplotlib import font_manager


def set_plot_defaults():
	plt.rcParams['text.usetex'] = False
	plt.rcParams['font.size'] = '14'
	plt.rcParams["font.family"] = "serif"
	plt.rcParams["font.serif"] = "Times New Roman"
	plt.rcParams['mathtext.fontset'] = 'cm'

	for fontpath in font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
	    if 'Times'.lower() in fontpath.lower():
	        font_manager.fontManager.addfont(fontpath)
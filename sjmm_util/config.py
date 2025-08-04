import matplotlib.pyplot as plt
from matplotlib import font_manager


def set_plot_defaults():
	plt.style.use('sjmm_util.sjmm_util.sjmdefaults')

	for fontpath in font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
	    if 'Times'.lower() in fontpath.lower():
	        font_manager.fontManager.addfont(fontpath)
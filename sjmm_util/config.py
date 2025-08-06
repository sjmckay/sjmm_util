import matplotlib.pyplot as plt
from matplotlib import font_manager
import importlib.resources as resources
import sjmm_util

def set_plot_defaults():
	with resources.path(sjmm_util, "sjmdefaults.mplstyle") as style_path:
		plt.style.use(str(style_path))

	for fontpath in font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
	    if 'Times'.lower() in fontpath.lower():
	        font_manager.fontManager.addfont(fontpath)
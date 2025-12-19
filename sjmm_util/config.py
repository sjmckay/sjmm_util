import matplotlib.pyplot as plt
from matplotlib import font_manager
import importlib.resources as resources
import sjmm_util

def set_plot_defaults():
	plt.style.use('default')
	with resources.path(sjmm_util, "mydefaults.mplstyle") as style_path:
		plt.style.use(str(style_path))
	add_font('Times')


def set_plot_defaults_paper():
	plt.style.use('default')
	with resources.path(sjmm_util, "paper.mplstyle") as style_path:
		plt.style.use(str(style_path))
	add_font('Times')

def set_plot_defaults_talk():
	plt.style.use('ggplot')
	plt.style.use('dark_background')
	with resources.path(sjmm_util, "talk.mplstyle") as style_path:
		plt.style.use(str(style_path))
	add_font('Times')
	
def add_font(font):
	for fontpath in font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
	    if font.lower() in fontpath.lower():
	        font_manager.fontManager.addfont(fontpath)
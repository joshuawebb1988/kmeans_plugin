#!/usr/bin/env python
"""
	DataExplore Application plugin example.
	Created Jun 2017
	Copyright (C) Joshua Webb

	This program is free software; you can redistribute it and/or
	modify it under the terms of the GNU General Public License
	as published by the Free Software Foundation; either version 3
	of the License, or (at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program; if not, write to the Free Software
	Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
"""

from __future__ import absolute_import, division, print_function
from pandastable.plugin import Plugin
from pandastable import plotting, dialogs, Table, TableModel
try:
	from tkinter import *
	from tkinter.ttk import *
except:
	from Tkinter import *
	from ttk import *
from collections import OrderedDict, defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import random
from math import sqrt



class KMeansPlugin(Plugin):
	"""K-Means Plugin for DataExplore"""

	capabilities = ['gui','uses_sidepane'] #['gui','uses_sidepane']
	requires = ['']
	menuentry = 'K-Means Plugin'
	gui_methods = {}
	version = '0.1'
	pluginrow = 6 #row to add plugin frame beneath table
	
	def main(self, parent):
		if parent==None:
			return
		self.parent = parent
		self.parentframe = None
		self._doFrame()
		return

	def _doFrame(self):
		self.table = self.parent.getCurrentTable()
		self.table.showIndex()
		if 'uses_sidepane' in self.capabilities:
			self.mainwin = Frame(self.table.parentframe)
			self.mainwin.grid(row=self.pluginrow,column=0,columnspan=2,sticky='news')
		else:
			self.mainwin=Toplevel()
			self.mainwin.title('K-Means')
			self.mainwin.geometry('1024x500+200+100')
			
		#Create Plot Window
		sheet = self.parent.getCurrentSheet()
		pw = self.parent.sheetframes[sheet]
		self.parent.hidePlot()
		self.pf = Frame(pw)
		pw.add(self.pf, weight=3)
		self.fig, self.canvas = plotting.addFigure(self.pf)
		self.ax = self.fig.add_subplot(111)
		self.c10=['#1f77b4',
				'#ff7f0e',
				'#2ca02c',
				'#d62728',
				'#9467bd',
				'#8c564b',
				'#e377c2',
				'#7f7f7f',
				'#bcbd22',
				'#17becf']

		self.ID='Basic Plugin'
		#self._createMenuBar()

		l=Label(self.mainwin, text='In parent table above select columns  with numeric features (to apply K-Means) ')
		l.pack(side=TOP,fill=BOTH)
		
		grps = {'plot':['x','y','hue'],
				'K-Means':['K','init_type','n_init'],}
		inits = ['forgy','random_partition','K-Means++']
		self.groups = grps = OrderedDict(grps)
		datacols = []
		self.scores = [0., 0.]
		self.selectcols=list(self.table.model.df.columns)
		self.k_iteration = 0
		
		self.opts = {'K': {'type':'scale','default':2,'range':(1,30),'interval':1,'label':'K clusters'},
					 'init_type': {'type':'combobox','default':'forgy','items':inits},
					 'n_init': {'type':'entry','default':10},
					 'x': {'type':'combobox','default':'','items':datacols},
					 'y': {'type':'combobox','default':'','items':datacols},
					 'hue': {'type':'combobox','default':'','items':datacols},
					 }
		settingsFrame = Frame(self.mainwin)
		settingsFrame.pack(side=LEFT)
		b=Button(settingsFrame, text='Update with Selected Columns', command=self.updateclustertable)
		b.pack(side=TOP,fill=X,pady=2)
		
		controlsFrame = self._settingWidgets(settingsFrame)
		controlsFrame.pack(side=TOP)
		
		
		buttonsFrame = Frame(settingsFrame, padding=2)
		buttonsFrame.pack(side=BOTTOM)
		buttonsFrameLeft = Frame(buttonsFrame,padding=2)
		buttonsFrameLeft.pack(side=LEFT)
		buttonsFrameRight = Frame(buttonsFrame,padding=2)
		buttonsFrameRight.pack(side=RIGHT)
		b=Button(buttonsFrameLeft, text='Apply Settings', command=self.applySettings)
		b.pack(side=TOP,fill=X,pady=2)
		b=Button(buttonsFrameLeft, text='Start K-Means', command=self.startKMeans)
		b.pack(side=BOTTOM,fill=X,pady=2)
		b=Button(buttonsFrameRight, text='Plot Results', command=self._plot)
		b.pack(side=TOP,fill=X,pady=2)
		b=Button(buttonsFrameRight, text='Plot Scores', command=self._plot_scores)
		b.pack(fill=X,pady=2)
		b=Button(buttonsFrameRight, text='Close', command=self.quit)
		b.pack(side=BOTTOM,fill=X,pady=2)
		
		self.clustertableFrame = Frame(self.mainwin)
		self.clustertableFrame.pack(side=RIGHT,fill=BOTH)
		
		self.newwin = Toplevel()
		self.newwin.title('K-Means')
		self.newwin.geometry('200x300+200+100')
		
		tablesFrame = Frame(self.newwin)
		tablesFrame.pack(side=RIGHT,fill=BOTH)
		
		self.n_init_tableFrame = Frame(tablesFrame)
		self.n_init_tableFrame.pack(side=RIGHT,fill=BOTH)
		
		self.update()

		self.mainwin.bind("<Destroy>", self.quit)
		return
	
	def _plot_scores(self):
		#Create Plot Window
		self.applyOptions()
		
		hue= self.kwds['hue']
		hue = str(hue)
		if (hue == '') or (hue == "_KCluster"):
			df = self.table.model.df[["_KCluster","_KScore"]]
			df = df.sort_values(by=["_KCluster","_KScore"],ascending=[False,True])
			df.reset_index(drop=True, inplace=True)
			groups = df.groupby("_KCluster")
		else:
			df = self.table.model.df[["_KCluster","_KScore",hue]]
			df = df.sort_values(by=["_KCluster","_KScore"],ascending=[False,True])
			df.reset_index(drop=True, inplace=True)
			groups = df.groupby(hue)
		
		try:
			f=Figure()
			ax=f.add_subplot(111)
			ax.set_title("K-Means, iterations:{0}, avg score:{1}".format(self.k_iteration, np.mean(self.scores)))
			ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
			i=0
			for name, group in groups:
				ax.barh(group.index.values, group["_KScore"].values, label="{0}".format(name), color=self.c10[i%10], edgecolor='none', height=1)
				i+=1
			ax.set_xlabel("Silhouette Score")
			ax.legend(numpoints=1, loc='upper right')
		except Exception as e:
			self.showWarning(e)
			return
			
		for child in self.pf.winfo_children():
			child.destroy()
		self.fig, self.canvas = plotting.addFigure(self.pf, f)

		self.canvas.show()
		return
	
	
	def _plot(self):
		"""Do plot"""
		
		self.applyOptions()
		
		x= self.kwds['x']
		y= self.kwds['y']
		hue= self.kwds['hue']
		x=str(x)
		y=str(y)
		hue=str(hue)
		
		df=self.table.model.df
		df.columns = df.columns.astype(str)
		if hue == '':
			df = df.loc[:,[x,y,"_KCluster"]]
			groups = df.groupby("_KCluster")
		else:
			df = df.loc[:,[x,y,hue]]
			groups = df.groupby(hue)
			
		
		
		df_clustertable = self.clustertable.model.df
		df_clustertable.columns = df_clustertable.columns.astype(str)
		df_clustertable = df_clustertable.loc[:,[x,y]]
		clusters = df_clustertable.groupby(level="_KCluster")
		
		try:
			f=Figure()
			ax=f.add_subplot(111)
			ax.set_title("K-Means, iterations:{0}, avg score:{1}".format(self.k_iteration, np.mean(self.scores)))
			ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
			i=0
			for name, group in groups:
				ax.plot(group.loc[:,[x]], group.loc[:,[y]], marker='o', linestyle='', color=self.c10[i%10] , label="{0}".format(name), markersize=8)
				i+=1
			i=0
			for name, group in clusters:
				ax.plot(group.loc[:,[x]], group.loc[:,[y]], marker='D', linestyle='', color=self.c10[i%10], label="{0} center".format(name), markersize=14)
				i+=1
			ax.set_xlabel(x)
			ax.set_ylabel(y)
			ax.legend(numpoints=1, loc='upper right')
		except Exception as e:
			self.showWarning(e)
			return
			
		for child in self.pf.winfo_children():
			child.destroy()
		self.fig, self.canvas = plotting.addFigure(self.pf, f)

		self.canvas.show()
		return
		
	def _settingWidgets(self, parent, callback=None):
		"""Auto create tk vars, widgets for corresponding options and
		   and return the frame"""
		dialog, self.tkvars, self.widgets = plotting.dialogFromOptions(parent, self.opts, self.groups)
		#self.applyOptions()
		return dialog
		
	def update(self):
		"""Update data widget(s)"""
		df = self.table.model.df
		if '_KCluster' not in df.columns:
			df.insert(0,'_KCluster',np.uint32(0))
			self.table.showIndex()
			self.table.redraw()
		if '_KScore' not in df.columns:
			df.insert(1,'_KScore',0.)
			self.table.showIndex()
			self.table.redraw()
		cols = list(df.columns)
		cols += ''
		self.widgets['x']['values'] = cols
		self.widgets['y']['values'] = cols
		self.widgets['hue']['values'] = cols
		
		self.updateclustertable()
		self.update_n_init_table()

		return
		
	def update_n_init_table(self):
		self.applyOptions()
		n_init = self.kwds['n_init']
		df_n_init_table = pd.DataFrame(data=np.zeros(n_init), columns=["Avg Silhouette Score"])
		df_n_init_table.index.name = "n"
		self.n_init_table = Table(self.n_init_tableFrame, dataframe=df_n_init_table)
		self.n_init_table.show()
		self.n_init_table.showIndex()
		self.n_init_table.redraw()
		
		return
		
	def updateclustertable(self):
		"""Update the cluster table with selected columns"""
		self.applyOptions()
		df = self.table.model.df
		if '_KCluster' not in df.columns:
			df.insert(0,'_KCluster',np.uint32(0))
			self.table.showIndex()
			self.table.redraw()
		if '_KScore' not in df.columns:
			df.insert(1,'_KScore',0.)
			self.table.showIndex()
			self.table.redraw()
		cols = df.columns[self.table.multiplecollist].values.tolist()
		if '_KCluster' in cols:
			cols.remove('_KCluster')
		if '_KScore' in cols:
			cols.remove('_KScore')
		self.selectcols = cols
		K=int(self.kwds['K'])
		self.showWarning("K={0} cols={1}".format(K, cols))
		df_clusters = pd.DataFrame(data=np.zeros((K,len(cols))), columns=cols, index=np.arange(0,K, dtype=np.uint32))
		df_clusters.index.name = "_KCluster"
		self.clustertable = Table(self.clustertableFrame, dataframe=df_clusters)
		self.clustertable.show()
		self.clustertable.showIndex()
		self.clustertable.redraw()
		#self.clustertable.show()
		
		return

	def applySettings(self):
		"""Apply K-Means settings"""
		self.applyOptions()
		df = self.table.model.df
		if '_KCluster' not in df.columns:
			df.insert(0,'_KCluster',np.uint32(0))
			self.table.showIndex()
			self.table.redraw()
		if '_KScore' not in df.columns:
			df.insert(1,'_KScore',0.)
			self.table.showIndex()
			self.table.redraw()
		cols = self.selectcols
		if '_KCluster' in cols:
			cols.remove('_KCluster')
		if '_KScore' in cols:
			cols.remove('_KScore')
		K=int(self.kwds['K'])
		self.showWarning("K={0} cols={1}".format(K, cols))
		df_clusters = pd.DataFrame(data=np.zeros((K,len(cols))), columns=cols, index=np.arange(0,K, dtype=np.uint32))
		df_clusters.index.name = "_KCluster"
		self.clustertable = Table(self.clustertableFrame, dataframe=df_clusters)
		self.clustertable.show()
		self.clustertable.showIndex()
		self.clustertable.redraw()
		
		return
		
	def applyOptions(self):
		"""Set the options"""
		kwds = {}
		for i in self.opts:
			if self.opts[i]['type'] == 'listbox':
				items = self.widgets[i].curselection()
				kwds[i] = [self.widgets[i].get(j) for j in items]
				print (items, kwds[i])
			else:
				kwds[i] = self.tkvars[i].get()
		self.kwds = kwds
		return
		
	def startKMeans(self):
		self.applySettings()
		
		df=self.table.model.df
		
		#selectcols = df.columns[self.table.multiplecollist].values.tolist()
		selectcols = self.selectcols
		
		is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
		if not(is_number(df[selectcols].dtypes).all()):
			self.showWarning("Selected data contains non-numeric data")
			return
		
		pnts_data=df[selectcols].values.tolist()
		
		K= int(self.kwds['K'])
		init_type= self.kwds['init_type']
		n_init= self.kwds['n_init']
		
		self.showWarning("prior to algo")#
		self.kmeansalgo = KMeansAlgorithm(self, pnts_data, K, init_type, n_init)
		self.showWarning("prior to algo run")#
		self.kmeansalgo.run()
		self.showWarning("post algo run")#
		assignments = self.kmeansalgo.assignments
		distances = self.kmeansalgo.distances
		scores = self.kmeansalgo.scores
		final_scores = self.kmeansalgo.final_scores
		centers = self.kmeansalgo.centers
		self.showWarning("post results")#
		self.assignments = assignments
		self.centers = centers
		self.distances = distances
		self.scores = scores
		self.final_scores = final_scores
		
		df['_KCluster'] = self.assignments
		df['_KCluster'] = df['_KCluster'].astype(int)
		df['_KScore'] = self.scores
		self.table.show()
		self.table.showIndex()
		self.table.redraw()
		
		df_clusters = pd.DataFrame(data=centers, columns=selectcols)
		df_clusters.index.name = "_KCluster"
		self.clustertable = Table(self.clustertableFrame, dataframe=df_clusters)
		self.clustertable.show()
		self.clustertable.showIndex()
		self.clustertable.redraw()
		
		df_n_init_table = pd.DataFrame(data=final_scores, columns=["Avg Silhouette Score"])
		df_n_init_table = df_n_init_table.sort_values(by=["Avg Silhouette Score"],ascending=[False])
		df_n_init_table.reset_index(drop=True, inplace=True)
		df_n_init_table.index.name = "n"
		self.n_init_table = Table(self.n_init_tableFrame, dataframe=df_n_init_table)
		self.n_init_table.show()
		self.n_init_table.showIndex()
		self.n_init_table.redraw()

		return
		
	def showWarning(self, s='plot error'):
		self.fig.clear()
		ax=self.fig.add_subplot(111)
		ax.text(.5, .5, s,transform=ax.transAxes,
					   horizontalalignment='center', color='blue', fontsize=16)
		self.canvas.draw()
		return
		
	def _createMenuBar(self):
		"""Create the menu bar for the application. """
		self.menu=Menu(self.mainwin)
		self.file_menu={ '01Quit':{'cmd':self.quit}}
		self.file_menu=self.create_pulldown(self.menu,self.file_menu)
		self.menu.add_cascade(label='File',menu=self.file_menu['var'])
		self.mainwin.config(menu=self.menu)
		return

	def quit(self, evt=None):
		"""Override this to handle pane closing"""
		self.mainwin.destroy()
		return

	def about(self):
		"""About this plugin"""
		txt = "This plugin implements ...\n"+\
			   "version: %s" %self.version
		return txt



class KMeansAlgorithm:
	"""
	Generates 'k' clusters and assigns data points to nearest cluster.
	Returns list of id's of the cluster assigned for each point, 
	and returns list of points representing cluster centres.
	"""
	def __init__(self, parent, pnts_data, k, init_type='forgy', n_init=10):
		"""Init KMeans Algorithm handler"""
		self.parent = parent
		self.pnts_data = pnts_data
		self.k = int(k)
		self.init_type = init_type
		self.n_init = int(n_init)
		self.assignments = []
		self.distances= []
		self.centers = []
		self.k_iteration = 0
		return

	def run(self, pnts_data=None, k=None, init_type=None, n_init=None):
		"""
		Returns points assigned to clusters, and the center point of each cluster.
		"""
		ITERATION_MAX = 300
		
		if pnts_data == None:
			pnts_data = self.pnts_data
		if k == None:
			k = self.k
		if init_type == None:
			init_type = self.init_type
		if n_init == None:
			n_init = self.n_init
		k=int(k)
		
		self.parent.showWarning("Starting K-Means algorithm")#
		self.final_score = 0.
		self.final_scores = []
		for i in range(n_init):
			centers = self.init_k(pnts_data, k, init_type)
			assignments, distances = self.assign_points(pnts_data, centers)
			old_assignments = None
			k_iteration = 0
			while (assignments != old_assignments) and (k_iteration < ITERATION_MAX):
				self.parent.showWarning("Iteration: {0}".format(k_iteration))#
				centers = self.update_centers(pnts_data, assignments)
				old_assignments = assignments
				assignments, distances = self.assign_points(pnts_data, centers)
				k_iteration +=1
			scores = self.silhouette(pnts_data, assignments, centers, distances)
			n_score = np.mean(scores)
			self.final_scores.append(n_score)
			if n_score > self.final_score:
				self.final_score = n_score
				self.scores = scores
				self.assignments = assignments
				self.distances = distances
				self.centers = centers
				self.parent.k_iteration = k_iteration
		return self.assignments, self.centers

	def init_k(self, pnts_data=None, k=None, init_type=None):
		"""
		Return initial center points of 'k' clusters.
		"""
		self.parent.showWarning("Initialising clusters init_type: {0}".format(init_type))#
		if pnts_data == None:
			pnts_data = self.pnts_data
		if k == None:
			k = self.k
		if init_type == None:
			init_type = self.init_type
		k=int(k)
		centers = []
		num_pnts = len(pnts_data)
		dims = len(pnts_data[0])
		
		if (init_type == 'forgy'): #forgy
			self.parent.showWarning("forgy init_type: {0}".format(init_type))#
			picks = random.sample(range(num_pnts),k)
			#self.parent.showWarning("picks: {0}".format(picks))#
			for i in picks:
				centers.append(pnts_data[i])
			return centers
		elif (init_type == 'random_partition'): #random partition
			self.parent.showWarning("random_partition init_type: {0}".format(init_type))#
			assignments = []
			for i in range(num_pnts):
				assign = random.randrange(k)
				assignments.append(assign)
			centers = self.update_centers(pnts_data, assignments)
			return centers
		else: #K-Means++
			self.parent.showWarning("K-Means++ init_type: {0}".format(init_type))#
			rando = random.randrange(num_pnts)
			#self.parent.showWarning("rando: {0}".format(rando))#
			center = pnts_data[rando]
			centers.append(center)
			while len(centers) < k:
				assignments, distances = self.assign_points(pnts_data, centers)
				weights = distances
				#self.parent.showWarning("weights: {0}".format(weights))#
				norm = [float(i)/sum(weights) for i in weights]
				rando2 = np.random.choice(a=num_pnts,p=norm)
				#self.parent.showWarning("rando2: {0}".format(rando2))#
				center2=pnts_data[rando2]
				centers.append(center2)
				#self.parent.showWarning("centers: {0}".format(centers))#
		return centers

		
		
	def silhouette(self, pnts_data=None, assignments=None, centers=None, distances=None):
		"""
		Returns list of silhouette scores for each point
		"""
		if pnts_data == None:
			pnts_data = self.pnts_data
		if assignments == None:
			assignments = self.assignments
		if centers == None:
			centers = self.centers
		if distances == None:
			distances = self.distances
		
		s_list=[]
		num_pnts = len(pnts_data)
		#dims = len(pnts_data[0])
		k = len(centers)
		for i in range(num_pnts):
			a = pnts_data[i]
			a_i = distances[i]
			#a_i= self.distance(a,centers[i])
			b_i = float("inf")
			for j in range(k):
				if assignments[i] == j:
					continue
				b=centers[j]
				value=self.distance(a,b)
				if value < b_i:
					b_i = value
			s_i = b_i - a_i
			s_i /= max(a_i, b_i)
			s_list.append(s_i)
		return s_list
		
	def pnts_mean(self, pnts):
		"""
		Returns a mean (center) point for a set of points.
		i.e. one cluster
		"""
		#self.parent.showWarning("Calculating mean of points")#
		dims = len(pnts[0])
		center = []
		for dim in range(dims):
			dim_sum = 0
			for pnt in pnts:
				dim_sum += pnt[dim]
			center.append(dim_sum / float(len(pnts)))
		return center

	def update_centers(self, pnts_data=None, assignments=None):
		"""
		Return a set of mean (center) points for a set of points and cluster assignments.
		i.e. 'K' mean (center) points, corresponding to each unique cluster.
		"""
		#self.parent.showWarning("Updating cluster centers")#
		if pnts_data == None:
			pnts_data = self.pnts_data
		if assignments == None:
			assignments = self.assignments
		cluster_pnts = defaultdict(list)
		centers = []
		#self.parent.showWarning("begin cluster_pnts {0}".format(cluster_pnts))#
		for assign, pnt in zip(assignments, pnts_data):
			cluster_pnts[assign].append(pnt)
		#self.parent.showWarning("end cluster_pnts {0}".format(cluster_pnts))#
		for pnts in iter(cluster_pnts.values()):
			#self.parent.showWarning("pnts {0}".format(pnts))#
			mean_pnt = self.pnts_mean(pnts)
			centers.append(mean_pnt)
		return centers

	def assign_points(self, pnts_data=None, centers=None):
		"""
		Return closest clusters for each point.
		"""
		#self.parent.showWarning("Assigning points to clusters")#
		if pnts_data == None:
			pnts_data = self.pnts_data
		if centers == None:
			centers = self.centers
		assignments = []
		distances = []
		for pnt in pnts_data:
			k_dist = float("inf")
			k_index = 0
			#self.parent.showWarning("k_index:{0}".format(k_index))#
			for i in range(len(centers)):
				value = self.distance(pnt, centers[i])
				#self.parent.showWarning("value {0} ".format(value))#
				if value < k_dist:
					k_dist = value
					k_index = i
			assignments.append(k_index)
			distances.append(k_dist)
		return assignments, distances

	def distance(self, a, b):
		"""
		Calculate Euclidean distance between two points
		"""
		#self.parent.showWarning("Calculating distance between points")#
		dims = len(a)
		
		distance_sq = 0
		for dim in range(dims):
			dim_dist_sq = (a[dim] - b[dim]) ** 2
			distance_sq += dim_dist_sq
		return sqrt(distance_sq)




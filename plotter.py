#!/usr/bin/env python

import sys
import msgpack
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from matplotlib.widgets import Slider
import yaml



class Plotter:
	def __init__(self, file_name, config_file_name):
		print("Initializing...")

		# Read config file
		with open(config_file_name) as file:
			self.cfg = yaml.load(file, Loader=yaml.FullLoader)

		# Read msg pack
		with open(file_name, "rb") as msg_pack_file:
		    msg_pack_byte_data = msg_pack_file.read()

		self.data = msgpack.unpackb(msg_pack_byte_data)
		
		# Settings for plotting
		plt.figure(1, figsize=[20,10])

		self.ax_3d = plt.axes([0,0.2,0.5,0.8],projection='3d')
		self.ax_3d.set_xlim(self.cfg['x_min'], self.cfg['x_max'])
		self.ax_3d.set_ylim(self.cfg['y_min'], self.cfg['y_max'])
		self.ax_3d.set_zlim(self.cfg['z_min'], self.cfg['z_max'])
		self.ax_3d.set_xlabel('X axis')
		self.ax_3d.set_ylabel('Y axis')
		self.ax_3d.set_zlabel('Z axis')

		self.ax_occ_grid = plt.axes([0.5,0.2,0.45,0.75])
		self.ax_occ_grid.set_xlim(self.cfg['x_min'], self.cfg['x_max'])
		self.ax_occ_grid.set_ylim(self.cfg['y_min'], self.cfg['y_max'])
		self.ax_occ_grid.set_xlabel('X axis')
		self.ax_occ_grid.set_ylabel('Y axis')

		self.ax_roll	= 	plt.axes([0.2, 0.1, 0.6, 0.02])
		self.ax_pitch	= 	plt.axes([0.2, 0.075, 0.6, 0.02])
		self.ax_yaw	= 	plt.axes([0.2, 0.05, 0.6, 0.02])

		self.roll = Slider(self.ax_roll,label='Roll',valmin=-90,valmax=90,valstep=0.1)
		self.pitch = Slider(self.ax_pitch,label='Pitch',valmin=-90,valmax=90)
		self.yaw = Slider(self.ax_yaw,label='Yaw',valmin=-90,valmax=90)

		self.roll.on_changed(self.roll_fn)

	def roll_fn(self, val):
		print(val)
		self.cfg['roll'] = val

		self.ax_3d.clear()

		self.ax_3d.set_xlim(self.cfg['x_min'], self.cfg['x_max'])
		self.ax_3d.set_ylim(self.cfg['y_min'], self.cfg['y_max'])
		self.ax_3d.set_zlim(self.cfg['z_min'], self.cfg['z_max'])
		self.ax_3d.set_xlabel('X axis')
		self.ax_3d.set_ylabel('Y axis')
		self.ax_3d.set_zlabel('Z axis')

		self.load_setup()
		
	def rotation_matrix(self, axis, theta):
	    """
	    Return the rotation matrix associated with counterclockwise rotation about
	    the given axis by theta radians.
	    """
	    axis = np.asarray(axis)
	    axis = axis / np.sqrt(np.dot(axis, axis))
	    a = np.cos(theta / 2.0)
	    b, c, d = -axis * np.sin(theta / 2.0)
	    aa, bb, cc, dd = a * a, b * b, c * c, d * d
	    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
	    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
	                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
	                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

	def rotate(self, axis, angle, points):
		"""
		Rotate 3d coordinates
		"""
		rad_angle = (angle/180)*np.pi
		points = np.transpose(points)
		points = np.dot(self.rotation_matrix(axis, rad_angle), points)
		points = np.transpose(points)
		return points

	def extract_keyframe_poses(self):
		"""
		Extract keyframe poses
		"""
		print('Extracting keyframes ...')

		kfs_trans = []
		kfs_rot = []
		kfs_pos = []
		for kf in self.data['keyframes']:
			ktrans = self.data['keyframes'][kf]['trans_cw']
			kf_trans = [ktrans[0],ktrans[1],ktrans[2]]
			
			krot = self.data['keyframes'][kf]['rot_cw']
			kf_rot = [krot[0],krot[1],krot[2],krot[3]]
			
			kf_rot_mat = R.from_quat(kf_rot)
			kf_pos = np.matmul(-np.transpose(kf_rot_mat.as_matrix()), np.transpose(kf_trans))
			# print(kf_trans, kf_pos)

			kfs_trans.append(kf_trans)
			kfs_rot.append(kf_rot)
			kfs_pos.append(kf_pos)



		self.keyframes_trans = np.array(kfs_trans)
		self.keyframes_rot = np.array(kfs_rot)
		self.keyframes_pos = np.array(kfs_pos)

	def extract_landmark_poses(self):
		"""
		Extract landmark poses
		"""
		print('Extracting landmarks ...')
		lms_trans = []
		for lm in self.data['landmarks']:
			lmtrans = self.data['landmarks'][lm]['pos_w']
			lm_trans = [lmtrans[0],lmtrans[1],lmtrans[2]]
			lms_trans.append(lm_trans)

		self.landmarks_trans = np.array(lms_trans)

	def rotate_keypoints_and_landmarks(self, x_angle, y_angle, z_angle):

		print('Applying rotation ...')

		self.keyframes_pos = self.rotate([1,0,0],x_angle,self.keyframes_pos)
		self.landmarks_trans = self.rotate([1,0,0],x_angle,self.landmarks_trans)

		self.keyframes_pos = self.rotate([0,1,0],y_angle,self.keyframes_pos)
		self.landmarks_trans = self.rotate([0,1,0],y_angle,self.landmarks_trans)

		self.keyframes_pos = self.rotate([0,0,1],z_angle,self.keyframes_pos)
		self.landmarks_trans = self.rotate([0,0,1],z_angle,self.landmarks_trans)

	

	def height_thresholder(self, points, vmin, vmax):
		new_points = []
		for point in points:
			if point[2] > vmin and point[2] < vmax:
				new_points.append(point)

		return np.array(new_points)

	def apply_height_threshold(self, lower_threshold, upper_threshold):

		print("Applying height threshod ...")

		v_min = np.amin(self.landmarks_trans, axis=0)[2]
		v_max = np.amax(self.landmarks_trans, axis=0)[2]
		range_ = v_max-v_min

		print('Min Z = ', v_min)
		print('Max Z = ', v_max)

		upper_val = v_min + ((100-upper_threshold)/100)*range_
		lower_val = v_min + (lower_threshold/100)*range_

		self.landmarks_trans = self.height_thresholder(self.landmarks_trans,lower_val,upper_val)



	def get_path_mask(self, coordinate, radius):
	
		minx = self.landmarks_trans[:,0] < (coordinate[0]-radius)
		maxx = self.landmarks_trans[:,0] > (coordinate[0]+radius)
		miny = self.landmarks_trans[:,1] < (coordinate[1]-radius)
		maxy = self.landmarks_trans[:,1] > (coordinate[1]+radius)

		x_filter = np.logical_or(minx,maxx)
		y_filter = np.logical_or(miny,maxy)
		all_filter = np.logical_or(x_filter,y_filter)

		return all_filter

	def apply_path_threshold(self, radius):

		print("Applying path threshold... ")

		all_mask = self.get_path_mask(self.keyframes_pos[0],radius)

		for kp in self.keyframes_pos:
			mask = self.get_path_mask(kp,radius)
			all_mask = np.logical_and(mask,all_mask)

		self.landmarks_trans = self.landmarks_trans[all_mask]

	def plot_keyframes(self):

		self.ax_3d.scatter3D(self.keyframes_pos[:,0],self.keyframes_pos[:,1],self.keyframes_pos[:,2],color="blue", s=4);
		self.ax_occ_grid.scatter(self.keyframes_pos[:,0],self.keyframes_pos[:,1], color="blue", s=2);

	def plot_landmarks(self):

		self.ax_3d.scatter3D(self.landmarks_trans[:,0],self.landmarks_trans[:,1],self.landmarks_trans[:,2],color="black");
		self.ax_occ_grid.scatter(self.landmarks_trans[:,0],self.landmarks_trans[:,1], color="black", s=4);

	def animate_keyframes(self, delay):
		i=0
		for point in self.keyframes_pos:
			i += 1
			self.ax_occ_grid.scatter(point[0], point[1], color="red");
			self.ax_occ_grid.text(point[0], point[1], i)

			self.ax_3d.scatter3D(point[0], point[1], point[2], color="red");
			self.ax_3d.text(point[0], point[1], point[2], i)
			plt.pause(delay)
			# if i>50:
			# 	break

	def load_setup(self):

		print("Loading setup ...")
		self.rotate_keypoints_and_landmarks(self.cfg['roll'],self.cfg['pitch'],self.cfg['yaw'])
		self.apply_height_threshold(self.cfg['bottom_threshold'],self.cfg['top_threshold'])
		self.apply_path_threshold(self.cfg['path_threshold'])

		self.plot_keyframes()
		self.plot_landmarks()



if __name__ == "__main__": 

	if len(sys.argv) != 3:
		print('Implementation:   python3 plotter.py /path/to/msgpack/file.msg /path/to/msgpack/config_file.yaml')
		sys.exit()
    
	p = Plotter(sys.argv[1],sys.argv[2])

	p.extract_keyframe_poses()
	p.extract_landmark_poses()

	p.load_setup()

	# p.animate_keyframes(0.1);




	plt.show()






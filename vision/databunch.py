from fastai.vision import *
from core.databunch import DashDatabunch
from fastai.data_block import *
from path import Path



# Only tested with Mnist and coco
class DashVisionDatabunch:

	def create_vision_databunch(response):
		path = Path('./')

		src = DashVisionDatabunch.get_itemlist(response["vision"])
		src = DashDatabunch.split_databunch(response, src)
		src = DashDatabunch.label_databunch(response, src)

		# Add test


		tra = DashVisionDatabunch.create_transform(response['vision']['transform'])
		src = src.transform([tra,tra], tfm_y=True)

		# manually putting extra args like collate_fn, if we pass stuff from dictionary, it will be taken as a string
		return DashDatabunch.create_databunch(response, src, collate_fn=bb_pad_collate)

	@staticmethod
	def get_itemlist(response):
		# path = Path('data/mnist_tiny')
		# if response["subtask"] == "classification-single-label":

		# might be a better way to do this
		if response["subtask"] == "object-detection":
			return ObjectItemList.from_folder(path = response["input"]["from_folder"]["path"])
		if response['subtask'] == 'gan':
			return GANItemList.from_folder(
				path=response['input']['from_folder']['path'],
				noise_sz=response['subtask']['gan']['noise_sz'])

		if response["input"]["method"] == "from_folder":
			return ImageList.from_folder(response["input"]["from_folder"])
		if response["input"]["method"] == "from_csv":
			return ImageList.from_csv(response["input"]["from_csv"])

	@staticmethod
	def create_transform(response):
		if(response['chosen_data_aug']=='basic_transforms'):
			if(response['basic_transforms']['do_flip']):
				do_flip=bool(response['basic_transforms']['do_flip'])
			#if(response['transforms']['flip_vert']):
			flip_vert=bool(response['basic_transforms']['flip_vert'])
			if(response['basic_transforms']['max_rotate']):
				max_rotate=response['basic_transforms']['max_rotate']
			if(response['basic_transforms']['max_zoom']):
				max_zoom=response['basic_transforms']['max_zoom']
			if(response['basic_transforms']['max_lighting']):
				max_lighting=response['basic_transforms']['max_lighting']
			if(response['basic_transforms']['max_warp']):
				max_warp=response['basic_transforms']['max_warp']
			if(response['basic_transforms']['p_affine']):
				p_affine=response['basic_transforms']['p_affine']
			if(response['basic_transforms']['p_lighting']):
				p_lighting=response['basic_transforms']['p_lighting']
			tfms = get_transforms(do_flip,flip_vert,max_rotate,max_zoom,max_lighting,max_warp,p_affine,p_lighting)
		if(response['chosen_data_aug']=='zoom_crop'):
			sc=tuple(response['zoom_crop']['scale'])
			tfms = zoom_crop(scale=sc, do_rand=response['zoom_crop']['do_rand'],p=response['zoom_crop']['p'])   
		if(response['chosen_data_aug']=='manual'):
			tras=list()
			p=response['manual']
			if(p['brightness']):
				tras.append(brightness(change=p['brightness']['change']))
			if(p['contrast']):
				tras.append(contrast(scale=p['contrast']['scale']))
			if(p['crop']):
				tras.append(crop(size=p['crop']['size'],row_pct=p['crop']['row_pct'],col_pct=p['crop']['col_pct']))
			if(p['crop_pad']):
				tras.append(crop_pad(size=p['crop_pad']['size'],padding_mode=p['crop_pad']['padding_mode'],row_pct=p['crop']['row_pct'],col_pct=p['crop']['col_pct']))
			if(p['dihedral']):
				tras.append(dihedral(k=p['dihedral']['k']))
			if(p['dihedral_affine']):
				tras.append(dihedral_affine(k=p['dihedral']['k']))
			if(p['flip_lr']):
				tras.append(flip_lr())
			if(p['flip_affine']):
				tras.append(flip_affine())
			if(p['jitter']):
				tras.append(jitter(magnitude=p['jitter']['magnitude']))
			if(p['pad']):
				tras.append(pad(padding=p['pad']['padding'],mode=p['pad']['mode']))
			if(p['rotate']):
				tras.append(rotate(degrees=p['rotate']['degrees']))
			if(p['rgb_randomize']):
				tras.append(rgb_randomize(channel=p['rgb_randomize']['chosen_channels'],thresh=p['rgb_randomize']['chosen_thresh']))
			if(p['skew']):
				tras.append(skew(direction=p['skew']['direction'],invert=p['skew']['invert'],magnitude=p['skew']['magnitude']))
			if(p['squish']):
				tras.append(squish(scale=p['squish']['scale'],row_pct=p['squish']['row_pct'],col_pct=p['squish']['col_pct']))
			if(p['symmetric_wrap']):
				mag=tuple(p['symmetric_wrap']['magnitude'])
				tras.append(symmetric_warp(magnitude=mag))
			if(p['tilt']):
				tras.append(tilt(magnitude=p['tilt']['magnitude'],direction=p['tilt']['direction']))
			if(p['zoom']):
				tras.append(zoom(scale=p['zoom']['scale'],row_pct=p['zoom']['row_pct'],col_pct=p['zoom']['col_pct']))
			#if(p['cutout']):
			#  tras.append(cutout(length=p['cutout']['length'],n_holes=['cutout']['n_holes']))

			tfms= tras
		return tfms
			
	

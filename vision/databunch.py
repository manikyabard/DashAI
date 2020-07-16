from fastai.vision import *
from core.databunch import DashDatabunch
from fastai.data_block import *
from path import Path



# Only tested with Mnist and coco
class DashVisionDatabunch:

	def create_vision_databunch(response):
		path = Path('./')

		src = DashVisionDatabunch.get_itemlist(response)
		#src = DashDatabunch.split_databunch(response, src)
		src=src.split_by_rand_pct()
		src = DashDatabunch.label_databunch(response, src)

		# Add test
		tra = DashVisionDatabunch.create_transform(response['vision']['transform'])
		src = src.transform(tra, tfm_y=True,size=response['vision']['transform']['size'])
		#src = src.transform(tra)
		#print(tra[0])
		#print(tra[1])
		# manually putting extra args like collate_fn, if we pass stuff from dictionary, it will be taken as a string
		return DashDatabunch.create_databunch(response, src, collate_fn=bb_pad_collate)

	@staticmethod
	def get_itemlist(response):
		# path = Path('data/mnist_tiny')
		# if response["subtask"] == "classification-single-label":

		# might be a better way to do this
		if response["vision"]["subtask"] == "object-detection":
			return ObjectItemList.from_folder(path = response["vision"]["input"]["from_folder"]["path"])
		if response["vision"]['subtask'] == 'gan':
			return GANItemList.from_folder(
				path=response["vision"]['input']['from_folder']['path'],
				noise_sz=response["vision"]['subtask']['gan']['noise_sz'])
		if response["vision"]["subtask"] == "segmentation":
			return SegmentationItemList.from_folder(path=response["vision"]["input"]["from_folder"]["path"])
		if response["vision"]["input"]["method"] == "from_folder":
			return ImageList.from_folder(response["vision"]["input"]["from_folder"])
		if response["vision"]["input"]["method"] == "from_csv":
			return ImageList.from_csv(response["vision"]["input"]["from_csv"])

	@staticmethod
	def create_transform(response):
		if(response['train']):
			response_tr=response['train']
			if(response['chosen_aug_train']=='basic_transforms'):
				if(response_tr['basic_transforms']['do_flip']):
					do_flip=bool(response_tr['basic_transforms']['do_flip'])
				#if(response['transforms']['flip_vert']):
				flip_vert=bool(response_tr['basic_transforms']['flip_vert'])
				if(response_tr['basic_transforms']['max_rotate']):
					max_rotate=response_tr['basic_transforms']['max_rotate']
				if(response_tr['basic_transforms']['max_zoom']):
					max_zoom=response_tr['basic_transforms']['max_zoom']
				if(response_tr['basic_transforms']['max_lighting']):
					max_lighting=response_tr['basic_transforms']['max_lighting']
				if(response_tr['basic_transforms']['max_warp']):
					max_warp=response_tr['basic_transforms']['max_warp']
				if(response_tr['basic_transforms']['p_affine']):
					p_affine=response_tr['basic_transforms']['p_affine']
				if(response_tr['basic_transforms']['p_lighting']):
					p_lighting=response_tr['basic_transforms']['p_lighting']
				tfms_1 = get_transforms(do_flip,flip_vert,max_rotate,max_zoom,max_lighting,max_warp,p_affine,p_lighting)
				return tfms_1
			if(response['chosen_aug_train']=='zoom_crop'):
				tfms_1 = zoom_crop(scale=response_tr['zoom_crop']['scale'], do_rand=response_tr['zoom_crop']['do_rand'],p=response_tr['zoom_crop']['p'])   
				return tfms_1
			if(response['chosen_aug_train']=='manual'):
				tras=list()
				p=response_tr['manual']
				if('brightness' in response['manual_train']):
					tras.append(brightness(change=p['brightness']['change']))
				#Passed high and low as the same value in a tuple
				#Accept high and low seperately
				if('contrast' in response['manual_train']):
					tras.append(contrast(scale=(p['contrast']['l_scale'],p['contrast']['h_scale'])))
				
				if('crop' in response['manual_train']):
					tras.append(crop(size=p['crop']['size'],row_pct=p['crop']['row_pct'],col_pct=p['crop']['col_pct']))
				
				if('crop_pad' in response['manual_train']):
					tras.append(crop_pad(size=p['crop_pad']['size'],padding_mode=p['crop_pad']['padding_mode'],row_pct=p['crop']['row_pct'],col_pct=p['crop']['col_pct']))
				'''
				if('dihedral' in response['manual_train']):
					tras.append(dihedral(k=p['dihedral']['k']))
				
				if('dihedral_affine' in response['manual_train]):
					tras.append(dihedral_affine(k=p['dihedral']['k']))
				'''
				if('flip_lr' in response['manual_train']):
					tras.append(flip_lr())
				
				if('flip_affine' in response['manual_train']):
					tras.append(flip_affine())
				
				if('jitter' in response['manual_train']):
					tras.append(jitter(magnitude=p['jitter']['magnitude']))
				#padding should be less than input less
				if('pad' in response['manual_train']):
					tras.append(pad(padding=p['pad']['padding'],mode=p['pad']['mode']))
				

				if('rotate' in response['manual_train']):
					tras.append(rotate(degrees=p['rotate']['degrees']))
					'''
				
			# It's not possible to apply those transforms to your dataset:invalid literal for int() with base 10: 'Red'
				if(p['rgb_randomize']):
					tras.append(rgb_randomize(channel=p['rgb_randomize']['chosen_channels'],thresh=p['rgb_randomize']['chosen_thresh']))
					'''
				
				#Passed high and low as the same value in a tuple
				if('skew' in response['manual_train']):
					tras.append(skew(direction=(p['skew']['l_direction'],p['skew']['h_direction']),invert=p['skew']['invert'],magnitude=p['skew']['magnitude']))
				
				if('squish' in response['manual_train']):
					tras.append(squish(scale=p['squish']['scale'],row_pct=p['squish']['row_pct'],col_pct=p['squish']['col_pct']))
				
				if('symmetric_wrap' in response['manual_train']):
					tras.append(symmetric_warp(magnitude=p['symmetric_wrap']['magnitude']))
				#Passed high and low as the same value in a tuple
				if('tilt' in response['manual_train']):
					tras.append(tilt(magnitude=p['tilt']['magnitude'],direction=(p['tilt']['l_direction'],p['tilt']['h_direction'])))
				
				if('zoom' in response['manual_train']):
					tras.append(zoom(scale=p['zoom']['scale'],row_pct=p['zoom']['row_pct'],col_pct=p['zoom']['col_pct']))
				#Passed high and low as the same value in a tuple
				if('cutout' in response['manual_train']):
					tras.append(cutout(length=(p['cutout']['l_length'],p['cutout']['h_length']),n_holes=(p['cutout']['l_n_holes'],p['cutout']['h_n_holes'])))
				tfms_1= tras
			

		if(response['valid']):
			response_va=response['valid']
			if(response['chosen_aug_valid']=='basic_transforms'):
				if(response_va['basic_transforms']['do_flip']):
					do_flip=bool(response_va['basic_transforms']['do_flip'])
				#if(response['transforms']['flip_vert']):
				flip_vert=bool(response_va['basic_transforms']['flip_vert'])
				if(response_va['basic_transforms']['max_rotate']):
					max_rotate=response_va['basic_transforms']['max_rotate']
				if(response_va['basic_transforms']['max_zoom']):
					max_zoom=response_va['basic_transforms']['max_zoom']
				if(response_va['basic_transforms']['max_lighting']):
					max_lighting=response_va['basic_transforms']['max_lighting']
				if(response_va['basic_transforms']['max_warp']):
					max_warp=response_va['basic_transforms']['max_warp']
				if(response_va['basic_transforms']['p_affine']):
					p_affine=response_va['basic_transforms']['p_affine']
				if(response_va['basic_transforms']['p_lighting']):
					p_lighting=response_va['basic_transforms']['p_lighting']
				tfms_2 = get_transforms(do_flip,flip_vert,max_rotate,max_zoom,max_lighting,max_warp,p_affine,p_lighting)
				return tfms_2
			if(response['chosen_aug_valid']=='zoom_crop'):
				tfms_2 = zoom_crop(scale=response_va['zoom_crop']['scale'], do_rand=response_va['zoom_crop']['do_rand'],p=response_va['zoom_crop']['p'])   
				return tfms_2
			if(response['chosen_aug_valid']=='manual'):
				tras=list()
				p=response_tr['manual']
				if('brightness' in response['manual_valid']):
					tras.append(brightness(change=p['brightness']['change']))
				#Passed high and low as the same value in a tuple
				#Accept high and low seperately
				if('contrast' in response['manual_valid']):
					tras.append(contrast(scale=(p['contrast']['l_scale'],p['contrast']['h_scale'])))
				
				if('crop' in response['manual_valid']):
					tras.append(crop(size=p['crop']['size'],row_pct=p['crop']['row_pct'],col_pct=p['crop']['col_pct']))
				
				if('crop_pad' in response['manual_valid']):
					tras.append(crop_pad(size=p['crop_pad']['size'],padding_mode=p['crop_pad']['padding_mode'],row_pct=p['crop']['row_pct'],col_pct=p['crop']['col_pct']))
				'''
				if('dihedral' in response['manual_valid']):
					tras.append(dihedral(k=p['dihedral']['k']))
				
				if('dihedral_affine' in response['manual_valid']):
					tras.append(dihedral_affine(k=p['dihedral']['k']))
				'''
				if('flip_lr' in response['manual_valid']):
					tras.append(flip_lr())
				
				if('flip_affine' in response['manual_valid']):
					tras.append(flip_affine())
				
				if('jitter' in response['manual_valid']):
					tras.append(jitter(magnitude=p['jitter']['magnitude']))
				#padding should be less than input less
				if('pad' in response['manual_valid']):
					tras.append(pad(padding=p['pad']['padding'],mode=p['pad']['mode']))
				

				if('rotate' in response['manual_valid']):
					tras.append(rotate(degrees=p['rotate']['degrees']))
				
				'''
				# It's not possible to apply those transforms to your dataset:invalid literal for int() with base 10: 'Red'
				if(p['rgb_randomize']):
					tras.append(rgb_randomize(channel=p['rgb_randomize']['chosen_channels'],thresh=p['rgb_randomize']['chosen_thresh']))
				'''
				
				#Passed high and low as the same value in a tuple
				if('skew' in response['manual_valid']):
					tras.append(skew(direction=(p['skew']['l_direction'],p['skew']['h_direction']),invert=p['skew']['invert'],magnitude=p['skew']['magnitude']))
				
				if('squish' in response['manual_valid']):
					tras.append(squish(scale=p['squish']['scale'],row_pct=p['squish']['row_pct'],col_pct=p['squish']['col_pct']))
				
				if('symmetric_wrap' in response['manual_valid']):
					tras.append(symmetric_warp(magnitude=p['symmetric_wrap']['magnitude']))
				#Passed high and low as the same value in a tuple
				if('tilt' in response['manual_valid']):
					tras.append(tilt(magnitude=p['tilt']['magnitude'],direction=(p['tilt']['l_direction'],p['tilt']['h_direction'])))
				
				if('zoom' in response['manual_valid']):
					tras.append(zoom(scale=p['zoom']['scale'],row_pct=p['zoom']['row_pct'],col_pct=p['zoom']['col_pct']))
				#Passed high and low as the same value in a tuple
				if('cutout' in response['manual_valid']):
					tras.append(cutout(length=(p['cutout']['l_length'],p['cutout']['h_length']),n_holes=(p['cutout']['l_n_holes'],p['cutout']['h_n_holes'])))
				tfms_2= tras
		return (tfms_1,tfms_2)


			
	

import numpy as np
import skimage.morphology as morph
import skimage.segmentation as segm
import skimage.feature as feature
import skimage.measure as measure
from scipy import ndimage



### calculation of (maximum) reconstruction volume size
### along z axis
def calc_zsize(x, y):
    return x+y


### calculation of number of voxels in the 
### reconstruction volume
def calc_voxels(x, y):
    return x*y*calc_zsize(x,y)


### definition of regional maxima
def find_reg_max(im, mask):
    dilated = ndimage.grey_dilation(im, 
                                    footprint=morph.square(3), 
                                    mode='nearest')
    reg_max = ((dilated-im <= 0.4) & (im>0) & mask)
    return reg_max


### definition of regional maxima for edge regions
def find_reg_max_lone(im, mask):
    dilated = ndimage.grey_dilation(im, 
                                    footprint=morph.disk(2), 
                                    mode='nearest') 
    reg_max = ((dilated-im ==0.) & (im>0) & mask)
    return reg_max


### definition of substructures
def label_image_ws(im, include_edge_regions=True):
    labels, num_labels = ndimage.measurements.label(im)
    dist = ndimage.distance_transform_edt(im)
    dist_taxicab = \
            ndimage.distance_transform_cdt(im, 
                                           metric='taxicab')
    
    im_new = np.copy(im)
    
    ### removal of structures at the image edge
    if not include_edge_regions:
        reg_max = find_reg_max_lone(dist, mask=im>0)
        labelled_array_max, num_max = \
            ndimage.measurements.label(reg_max, 
                                       structure=morph.square(3))
        labels_lone_ws = \
            segm.watershed(-dist, 
                            markers=labelled_array_max, 
                            mask=im>0, connectivity=2, 
                            compactness=0.)
        list_edge = np.unique(np.concatenate( \
                        (labels_lone_ws[:1,:].flatten(), 
                        labels_lone_ws[-2:,:].flatten(), 
                        labels_lone_ws[:,:1].flatten(), 
                        labels_lone_ws[:,-2:].flatten())))
        
        for l in list_edge:
            if l>0:
                im_new[labels_lone_ws==l]=0
                dist[labels_lone_ws==l]=0
                dist_taxicab[labels_lone_ws==l]=0
    
    ### regional maxima are found and used as input for 
    ### watershed segmentation
    reg_max = find_reg_max(dist, mask=im_new>0)
    labelled_array_max, num_max = \
        ndimage.measurements.label(reg_max, 
                                   structure=[[0,0,0], 
                                              [0,1,0], 
                                              [0,0,0]])
    labels_ws = segm.watershed(-dist, 
                                markers=labelled_array_max, 
                                mask=im_new>0, connectivity=2, 
                                compactness=0.)
    
    return (labels_ws, dist, dist_taxicab, reg_max, num_max)


### calculation of density reconstruction for one substructure
def calc_Abel_shell(fl, dist, reg_max, R1, max_zsize):
    (c_xsize, c_ysize) = np.shape(fl)
    
    c_zsize = 2*int(R1+0.5)+2
    if max_zsize is not None:
        c_zsize = min((max_zsize, c_zsize))
    
    z_centre_pix = int(c_zsize/2.)
    z_centre = c_zsize/2.
    
    fl_fromthres = np.zeros((c_xsize, c_ysize, c_zsize), 
                            dtype='float32')
    
    ### calculation of distances along z axis
    z_line=np.abs(np.arange(0., c_zsize+0.)-z_centre)**2
    z_plane=np.stack([z_line]*c_ysize, axis=0)
    z_dist2=np.stack([z_plane]*c_xsize, axis=0)  
    
    fl_comp = np.copy(fl)
    mask_thres = fl_comp > 0.
      
    ### calculation of 3D distance to substructure centre
    R1_map = np.zeros(np.shape(dist))
    R1_map[mask_thres] = R1
    R1_cube = np.dstack([R1_map]*c_zsize)
    rho_map = R1_map-dist+0.5
    r_cube = np.sqrt((rho_map[:,:,np.newaxis])**2 + z_dist2)  
        
    ### setting lower and upper integration limits
    r_upplim = r_cube + 0.5
    r_lowlim = r_cube - 0.5
    r_lowlim[r_lowlim < 0.] = 0.
    r_lowlim[:,:,z_centre_pix][reg_max] = 0.
    r_upplim[r_upplim > R1_cube] = R1_cube[r_upplim > R1_cube]
    
    ### calculation of reconstructed density
    g = fl_comp[:,:,np.newaxis] / np.pi * \
            (np.arcsin(r_upplim/R1_cube) - \
             np.arcsin(r_lowlim/R1_cube))

    ### calculation of reconstructed density at 
    ### substructure centres
    g1_centres = fl_comp[reg_max] / np.pi /R1_map[reg_max]
    g[:,:,z_centre_pix][reg_max] = g1_centres
        
    mask_reg1 = (np.isfinite(g))
    fl_fromthres[mask_reg1] += g[mask_reg1]

    return fl_fromthres


### calculation of AVIATOR reconstruction
def estimate_3D(im, thres_list=None, max_zsize=None, 
                include_edge_regions=True, max_volume=None):
    ### calculation of threshold levels, if none given
    if thres_list is None:
        thres_list = np.sort(np.unique(im.flatten()))
        if thres_list[0] != 0:
            thres_list = np.concatenate([[0.], thres_list])
    
    ### calculation of z size of reconstruction volume
    z_size = calc_zsize(*np.shape(im))
    if max_zsize is not None:
        z_size = min((max_zsize, z_size))
        
    cube = np.zeros((np.shape(im)[0], np.shape(im)[1], z_size))
    print('The estimated 3D density cube will have the shape', 
          np.shape(cube), '.')
    print('The list of threshold values contains',
          len(thres_list), 'entries.')
    
    z_center = int(z_size/2.)
    
    map_t = np.copy(im)
    
    for i in range(len(thres_list)-1):
        percent = int((i+1)/(len(thres_list)-1)*100)
        if (percent % 5 ==0):
            print('\r' + str(percent) + \
                  '% of threshold values calculated', 
                  end='')
        
        ### extraction of image component at a given level
        map_comp = np.copy(im)
        map_comp[im > thres_list[i+1]] = thres_list[i+1]
        map_comp = map_comp - thres_list[i]
        map_comp[map_comp < 0.] =0.
        mask_thres = (map_comp > 0)
        
        ### definition of substructures in image component
        (labels_ws, dist, dist_taxicab, 
         reg_max, num_labels_ws) = \
            label_image_ws(mask_thres,
                           include_edge_regions = \
                               include_edge_regions)
        
        ### calculation of bounding boxes of substructures
        regionprops = measure.regionprops(labels_ws, map_comp)
        region_bbox = [r.bbox for r in np.array(regionprops)]
        
        ### separation of substructures according to given 
        ### maximum volume
        if max_volume is not None:
            region_volume = \
                np.array([calc_voxels( \
                    (r.bbox[2]-r.bbox[0]), 
                    (r.bbox[3]-r.bbox[1])) for r in regionprops])
            mask_volume = (region_volume <= max_volume)
        else:
            mask_volume = np.ones(num_labels_ws, dtype='bool')
        
        ### small enough substructures are reconstructed
        for j in np.array(range(num_labels_ws))[mask_volume]:
            
            xmin, ymin, xmax, ymax = region_bbox[j]
            mask_ws = \
                (labels_ws == j+1)[xmin:xmax, ymin:ymax]
            
            ### setup of maps,
            ### removal of all substructures except current
            map_comp_ws = \
                np.copy(map_comp[xmin:xmax, ymin:ymax])
            map_comp_ws[~mask_ws] = 0.
            dist_taxicab_ws = \
                np.copy(dist_taxicab[xmin:xmax, ymin:ymax])
            dist_taxicab_ws[~mask_ws] = 0.
            dist_ws = \
                np.copy(dist[xmin:xmax, ymin:ymax])
            dist_ws[~mask_ws] = 0.
            reg_max_ws = \
                np.copy(reg_max[xmin:xmax, ymin:ymax])
            reg_max_ws[~mask_ws] = False
            
            ### calculation of maximum radius of substructure
            R1 = \
                np.ceil(np.nanmax( \
                            dist_taxicab_ws[reg_max_ws]))-0.5

            ### R1=0 if watershed region is entire map
            ### region is not processed in this case
            if (R1 <= 0): 
                continue
            
            ### calculation of substructure reconstruction
            shell = calc_Abel_shell(map_comp_ws, dist_ws, 
                                    reg_max_ws, R1, 
                                    max_zsize=z_size)

            
            ### removal of component from image
            map_t[xmin:xmax, ymin:ymax] -= \
                                        map_comp_ws[:,:]
            
            ### addition of substructure reconstruction to 
            ### total reconstruction volume
            c_zsize = np.shape(shell)[2]
            zmin = z_center-int(c_zsize/2.)
            zmax = z_center+int(c_zsize/2.+0.5)
            cube[xmin:xmax, ymin:ymax, zmin:zmax] += \
                shell
            
        ### large substructures are reconstructed
        for j in np.array(range(num_labels_ws))[~mask_volume]: 
            
            xmin, ymin, xmax, ymax = region_bbox[j]
            mask_ws = \
                (labels_ws == j+1)[xmin:xmax, ymin:ymax]
            
            ### setup of maps,
            ### removal of all substructures except current
            map_comp_ws = \
                np.copy(map_comp[xmin:xmax, ymin:ymax])
            map_comp_ws[~mask_ws] = 0.
            dist_taxicab_ws = \
                np.copy(dist_taxicab[xmin:xmax, ymin:ymax])
            dist_taxicab_ws[~mask_ws] = 0.
            dist_ws = \
                np.copy(dist[xmin:xmax, ymin:ymax])
            dist_ws[~mask_ws] = 0.
            reg_max_ws = \
                np.copy(reg_max[xmin:xmax, ymin:ymax])
            reg_max_ws[~mask_ws] = False
            
            ### all maps are resampled to be smaller than given 
            ### maximum volume
            volume_ex = region_volume[j]/max_volume
            block_side = \
                np.int(2**np.ceil(np.log(volume_ex)/np.log(8)))
            bsize = (block_side,block_side)
            
            mask_ws_red = \
                measure.block_reduce(mask_ws, 
                                     block_size=bsize, 
                                     func=np.nanmedian)
            map_comp_ws_red = \
                measure.block_reduce(map_comp_ws, 
                                     block_size=bsize, 
                                     func=np.nansum)
            dist_taxicab_ws_red = \
                measure.block_reduce(dist_taxicab_ws, 
                                     block_size=bsize, 
                                     func=np.nanmean) /block_side
            dist_ws_red = \
                measure.block_reduce(dist_ws, 
                                     block_size=bsize, 
                                     func=np.nanmean) /block_side
            reg_max_ws_red = \
                measure.block_reduce(reg_max_ws, 
                                     block_size=bsize, 
                                     func=np.nanmax)
            
            ### calculation of maximum radius of substructure
            R1_red = \
                np.ceil(np.nanmax( \
                            [reg_max_ws_red]))-0.5
            
            ### R1=0 if watershed region is entire map
            ### region is not processed in this case
            if (R1 <= 0):
                continue
              
            ### calculation of substructure reconstruction
            shell_red = calc_Abel_shell( \
                        map_comp_ws_red, dist_ws_red, 
                        reg_max_ws_red, R1_red, 
                        max_zsize=int(z_size/block_side))
            
            ### calculation of zoom factors for resampling of 
            ### substructure reconstruction to original size
            map_comp_ws_xy = np.shape(map_comp_ws)
            map_comp_ws_red_xy = np.shape(map_comp_ws_red)
            c_zsize_red = np.shape(shell_red)[2]
            zoom_x = \
                map_comp_ws_xy[0]/map_comp_ws_red_xy[0]
            zoom_y = \
                map_comp_ws_xy[1]/map_comp_ws_red_xy[1]  
            zoom_z = \
                np.mean([zoom_x, zoom_y])
            
            ### resampling and smoothing of 
            ### substructure reconstruction
            shell = ndimage.filters.gaussian_filter( \
                ndimage.zoom(shell_red, [zoom_x,zoom_y,zoom_z], 
                             order=0, prefilter=False)/ \
                    (zoom_x*zoom_y*zoom_z), sigma=block_side/2.)
            
            ### removal of component from image
            map_t[xmin:xmax, ymin:ymax] -= map_comp_ws[:,:]
            
            ### addition of substructure reconstruction to 
            ### total reconstruction volume
            c_zsize = np.shape(shell)[2]
            zmin = z_center-int(c_zsize/2.)
            zmax = z_center+int(c_zsize/2.+0.5)
            cube[xmin:xmax, ymin:ymax, zmin:zmax] += shell
            
    return cube, map_t


### calculation of threshold levels
def thres_list_stepsize(im, stepsize, mode='absolute'):
    thres_list_all = np.sort(np.unique(im.flatten()))
    
    ### ensure that zero is in level list
    thres_list_culled = [0.]
    last_valid = thres_list_all[0]
    
    ### difference between levels is at least stepsize
    if mode=='absolute':
        thres_diff_min = stepsize
        for n, t in enumerate(thres_list_all[1:]):
            diff = t - last_valid
            if diff >= thres_diff_min:
                last_valid = t
                thres_list_culled.append(t)
                
    ### difference between levels is at least stepsize*level
    if mode=='progressive':
        for n, t in enumerate(thres_list_all[1:]):
            diff = t - last_valid
            thres_diff_min = last_valid*stepsize
            if diff >= thres_diff_min:
                last_valid = t
                thres_list_culled.append(t)           
            
    ### ensure that highest value is in level list
    if thres_list_culled[-1] != thres_list_all[-1]:
        thres_list_culled.append(thres_list_all[-1])

    return np.array(thres_list_culled)
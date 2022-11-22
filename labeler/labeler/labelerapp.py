import copy
import os
import json
from utils import renormalize
from utils import labwidget, paintwidget
import cv2
import numpy as np
import math
from scipy.spatial import distance

##########################################################################
# UI
##########################################################################

class LabelerApp(labwidget.Widget):

    def __init__(self, save_image_dir=None, save_anotation_dir=None, images=[], max_height=256, max_width=256):
        super().__init__(style=dict(border="0", padding="0",
                                    display="inline-block", width="1000px",
                                    left="0", margin="0"
                                    ), className='rwa')
        self.current_index = 0
        self.max_index = len(images)-1
        self.max_height = max_height
        self.show_original = False

        self.images = images
        self.height, self.width = images[0]['img'].shape[:2]
        self.save_image_dir = save_image_dir
        self.save_anotation_dir = save_anotation_dir
        self.request = {}
        self.msg_out = labwidget.Div()
        self.loss_out = labwidget.Div()
        self.query_out = labwidget.Div()
        self.imgnum_textbox = labwidget.Textbox('%d' % (self.current_index)
                                                )
        
        self.images[0]['image_url'] = renormalize.as_url(renormalize.as_image_from_array(images[0]['img']))
        # self.images[0]['mask_url'] = renormalize.as_url(renormalize.as_image_from_array(images[0]['mask']))
        
        self.copy_canvas = paintwidget.PaintWidget(
            image=self.images[0]['image_url'], width=self.width, height=self.height
        )
        self.paste_canvas = paintwidget.PaintWidget(
            image=renormalize.as_url(renormalize.as_image_from_array(images[0]['mask'])), width=self.width, height=self.height,
            opacity=0.0, oneshot=True,
        )
        self.object_out = labwidget.Div(
            style={'display': 'inline-block',
                   'vertical-align': 'top',
                   'width': '%spx' % self.width,
                   'height': '%spx' % self.height})
        self.target_out = labwidget.Div(
            style={'display': 'inline-block',
                   'vertical-align': 'top',
                   'width': '%spx' % self.width,
                   'height': '%spx' % self.height})
        self.context_out = labwidget.Div(
            style={'display': 'inline-block',
                   'vertical-align': 'top',
                   'text-align': 'left',
                   'width': '%spx' % ((self.width + 2) * 3 // 2),
                   'height': '%spx' % (self.height * 3 // 8 + 20),
                   'white-space': 'nowrap',
                   'overflow-x': 'scroll'},
            className='ctx_tray')
        self.context_img_array = []
        self.keytray_div = labwidget.Div(style={'display': 'none'})
        
        self.keytray_canvas = paintwidget.PaintWidget(
            width=self.width, height=self.height,
            vanishing=False, disabled=True)
            
        inline = dict(display='inline')
        self.query_btn = labwidget.Button('Match Sel', style=inline
                                          ).on('click', self.query)
        self.next_btn = labwidget.Button('Next', style=inline
                                          ).on('click', self.next_image)
        self.prev_btn = labwidget.Button('Prev', style=inline
                                          ).on('click', self.prev_image)  
        self.revert_btn = labwidget.Button('Revert', style=inline
                                           ).on('click', self.revert)

        self.original_btn = labwidget.Button('Toggle Original', style=inline
                                             ).on('click', self.toggle_original)
        self.mask_btn = labwidget.Button('Generate mask', style=inline
                                          ).on('click', self.generate_mask)
        self.brushsize_textbox = labwidget.Textbox(10, desc='brush: ', size=3
                                                   ).on('value', self.change_brushsize)
        self.save_btn = labwidget.Button('Save', style=inline
                                         ).on('click', self.save)
        self.mask_mode_label = labwidget.Label('Masking Mode:')
        self.mask_mode = labwidget.Choice(choices=['Paint mask', 'Contour by points'], selection='Paint mask'
                                         ).on('selection', self.mask_mode)
        self.current_mask_item = None

    def get_image(self):
        if self.images[self.current_index].get('image_url')==None:
            self.images[self.current_index]['image_url'] = renormalize.as_url(renormalize.as_image_from_array(self.images[self.current_index]['img']))
        if self.images[self.current_index].get('mask_url')==None:
            self.images[self.current_index]['mask_url'] = ''#renormalize.as_url(renormalize.as_image_from_array(self.images[self.current_index]['mask']))

        self.height, self.width = self.images[self.current_index]['img'].shape[:2]
        self.copy_canvas.height = self.height
        self.copy_canvas.width = self.width

        self.paste_canvas.height = self.height
        self.paste_canvas.width = self.width
        self.imgnum_textbox.value = self.current_index

        self.copy_canvas.image = self.images[self.current_index]['image_url']
        self.copy_canvas.mask = self.images[self.current_index]['mask_url']

        if self.images[self.current_index].get('paste_img')!=None:
            self.paste_canvas.image = self.images[self.current_index]['paste_img']
        else:
            mask_img = np.zeros(self.images[self.current_index]['img'].shape, dtype=np.uint8)
            self.paste_canvas.image = renormalize.as_url(renormalize.as_image_from_array(mask_img))
    
    def next_image(self):
        self.images[self.current_index]['mask_url'] = self.copy_canvas.mask
        if self.max_index > self.current_index:
            self.current_index+=1

        self.get_image()

    def prev_image(self):
        self.images[self.current_index]['mask_url'] = self.copy_canvas.mask
        if self.current_index > 0:
            self.current_index-=1

        self.get_image()

    def clockwiseangle_and_distance(self, origin, point):
        
        x = point[0][0] - origin[0]
        y = point[0][1] - origin[1]

        dst = distance.euclidean((point[0][0], point[0][1]), (origin[0], origin[1]))

        angle = math.atan2(y, x)
        # Negative angles represent counter-clockwise angles so we need to subtract them 
        # from 2*pi (360 degrees)
        if angle < 0:
            return (2*math.pi)+angle,dst
        
        return angle,dst
    
    def generate_mask(self):
        quad = {"quad":[]}

        if not (self.copy_canvas.mask == ''):
            mask = renormalize.from_url(self.copy_canvas.mask, target='image')
            
            height, width = self.images[self.current_index]['img'].shape[:2]
            mask_3c = np.array(mask)
            mask_3c = cv2.resize(mask_3c, (width, height))

            mask_cv = cv2.cvtColor(mask_3c, cv2.COLOR_BGR2GRAY)
            mask_contour = np.zeros(mask_cv.shape, dtype=np.uint8)
        
            selection = self.mask_mode.selection
            if (selection == 'Contour by points'):
                thresh = 100
                canny_output = cv2.Canny(mask_cv, thresh, thresh * 2)
                
                # Find contours
                contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                # Approximate contours to polygons + get bounding rects and circles
                centers = [None]*len(contours)
                radius = [None]*len(contours)
                for i, c in enumerate(contours):
                    cntr, r = cv2.minEnclosingCircle(contours[i])
                    centers[i] = np.array([int(cntr[0]),int(cntr[1])])
                
                centers = np.array(centers)
                centers = centers[np.argsort(centers[:, 0]), :]

                centers_contour = np.array([[[int(c[0]), int(c[1])]] for c in centers])
                centers_contour = cv2.approxPolyDP(centers_contour, 10, True)

                c, r = cv2.minEnclosingCircle(centers_contour)
                centers_sorted_contour = np.array(sorted(centers_contour, key=lambda pt: self.clockwiseangle_and_distance(c, pt)))
                quad = {"quad":[[int(point[0][0]), int(point[0][1])] for point in centers_sorted_contour]}
                
                cv2.drawContours(mask_contour, [centers_sorted_contour], 0, 255, cv2.FILLED)

            mask_cv_3c = cv2.cvtColor(mask_contour, cv2.COLOR_GRAY2BGR)

            self.images[self.current_index]['mask'] = mask_cv_3c

            image_overlayed = self.overlay(self.images[self.current_index]['img'], mask_contour)
            
            self.images[self.current_index]['paste_img'] = renormalize.as_url(renormalize.as_image_from_array(image_overlayed))
            self.paste_canvas.image = self.images[self.current_index]['paste_img']

        self.images[self.current_index]['quad'] = quad
        
    def mask_mode(self):
        selection = self.mask_mode.selection
        if (selection == 'Contour by points'):
            self.copy_canvas.point = True
            brushsize = 5
            self.brushsize_textbox.value = self.copy_canvas.brushsize
        else:
            self.copy_canvas.point = False
            brushsize = 10

        self.copy_canvas.brushsize = brushsize
        self.brushsize_textbox.value = self.copy_canvas.brushsize
    
    def change_brushsize(self):
        brushsize = int(self.brushsize_textbox.value)
        self.copy_canvas.brushsize = brushsize

    def overlay(self, image, mask):
        image_overlayed = np.zeros(image.shape, dtype=np.uint8)
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if mask[x,y] > 0:
                    image_overlayed[x,y,:] = image[x,y,:]

        return image_overlayed
    
    def toggle_original(self):
        self.show_original = not self.show_original
        if self.show_original:
            self.original_btn.label = 'Toggle Changed'
            mask = self.images[self.current_index]['mask']
            image = self.images[self.current_index]['img']

            image_overlayed = self.overlay(image, mask)
            self.paste_canvas.image = renormalize.as_url(renormalize.as_image_from_array(image_overlayed))
        else:
            self.original_btn.label = 'Toggle Original'
            mask = self.images[self.current_index]['mask']
            mask_cv_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            self.paste_canvas.image = renormalize.as_url(renormalize.as_image_from_array(mask_cv_3c))
    
    def revert(self):
        with torch.no_grad():
            self.gw.model.load_state_dict(self.original_model.state_dict())
        self.repaint_canvas_array()
        # self.saved_list.value = ''
        self.show_msg('reverted to original')

    def query(self):
        self.images[self.current_index]['mask_url'] = self.copy_canvas.mask
        if (int(self.imgnum_textbox.value) >= 0) and (int(self.imgnum_textbox.value) < self.max_index):
            self.current_index=int(self.imgnum_textbox.value)

        self.get_image()

    def save(self):
        raw = self.images[self.current_index]['img']
        mask = cv2.cvtColor(self.images[self.current_index]['mask'], cv2.COLOR_BGR2GRAY)
        quad = self.images[self.current_index].get('quad')
        filename = self.images[self.current_index]['filename']

        self.save_as_image(raw, filename + '_' + str(self.current_index) + '.tif')
        self.save_as_anotaion(quad, filename + '_' + str(self.current_index) + '.json')

    def tryload(self):
        return

    def saved_names(self):
        return

    def save_as_image(self, image, output_filename):
        data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output_dir = self.save_image_dir
        
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_dir + output_filename , data)

        self.show_msg('saved image as ' + output_filename )

    def save_as_anotaion(self, data, output_filename):
        if not data is None:
            output_dir = self.save_anotation_dir
            
            os.makedirs(output_dir, exist_ok=True)
            with open(output_dir + output_filename , 'w') as outfile:
                json.dump(data, outfile)

            self.show_msg('saved anotation as ' + output_filename )

    def show_msg(self, msg):
        self.msg_out.clear()
        self.msg_out.print(msg, replace=True)

    def widget_html(self):
        def h(w):
            return w._repr_html_()
        return f'''<div {self.std_attrs()}>
        <div style="
            vertical-align:top
            width:100%;">
            <center>
                <span style="font-size:24px;margin-right:24px;vertical-align:bottom;">
                    Segmentation Labeler
                </span>
            </center>

            <hr style="border:1px solid gray; background-color: white">
            
            <div style="width:100%; margin-top: 8px; margin-bottom: 8px; display: table; overflow: hidden;">
                <div style="width: 100%; display: table;">
                    <div style="display: table-row">
                        <div style="width: 100%; 
                            display: table-cell;
                            vertical-align:middle;">
                            <div style="width:100%; display: table;">
                                <div style="display: table-row">
                                    <div style="display:inline-block; width:50%;;
                                        padding-bottom:20px;
                                        text-align:center">
                                        {h(self.mask_btn)}
                                    </div>
                                    <div style="display:inline-block; width:50%;;
                                        padding-bottom:20px;
                                        text-align:center">
                                        {h(self.original_btn)}
                                    </div>
                                </div>
                                <div style="display: table-row">
                                    <div style="display:inline-block;
                                        width:; width:50%;
                                        vertical-align:top;
                                        text-align:center;
                                        background:#f2f2f2">
                                        {h(self.copy_canvas)}
                                    </div>
                                    <div style="display:inline-block;
                                        width:; width:50%;
                                        vertical-align:top;
                                        text-align:center;
                                        background:#f2f2f2">
                                        {h(self.paste_canvas)}
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div style="display: table-cell;
                                    float: right;
                                    width:200px;">
                            <div style="height:{(self.height + 50)}px;
                                display: inline-block;
                                vertical-align: top;
                                border-left: 4px dashed gray;
                                padding-left: 5px;
                                margin-left: 5px;">
                                {h(self.mask_mode_label)}
                                {h(self.mask_mode)}
                                {h(self.brushsize_textbox)}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div style="width:100%;">
        <hr style="border:1px solid gray; background-color: white">
        </div>

        <div style="width:100%; text-align: center;
           margin-top:8px;padding-top:30px;">
        Images {h(self.imgnum_textbox)}
        {h(self.query_btn)}
        {h(self.prev_btn)}
        {h(self.next_btn)}
        {h(self.revert_btn)}
        &nbsp;
        {h(self.save_btn)}
        </div>

        {h(self.loss_out)}
        {h(self.msg_out)}
        </div>'''


##########################################################################
# Utility functions
##########################################################################

def positive_bounding_box(data):
    pos = (data > 0)
    v, h = pos.sum(0).nonzero(), pos.sum(1).nonzero()
    left, right = v.min().item(), v.max().item()
    top, bottom = h.min().item(), h.max().item()
    return top, left, bottom + 1, right + 1


def centered_location(data):
    t, l, b, r = positive_bounding_box(data)
    return (t + b) // 2, (l + r) // 2


def paste_clip_at_center(source, clip, center, area=None):
    target = source.clone()
    # clip = clip[:,:,:target.shape[2],:target.shape[3]]
    t, l = (max(0, min(e - s, c - s // 2))
            for s, c, e in zip(clip.shape[2:], center, source.shape[2:]))
    b, r = t + clip.shape[2], l + clip.shape[3]
    # TODO: consider copying over a subset of channels.
    target[:, :, t:b, l:r] = clip if area is None else (
        (1 - area)[None, None, :, :].to(target.device) * target[:, :, t:b, l:r] + area[None, None, :, :].to(target.device) * clip)
    return target, (t, l, b, r)


def crop_clip_to_bounds(source, target, bounds):
    t, l, b, r = bounds
    vr, hr = [ts // ss for ts, ss in zip(target.shape[2:], source.shape[2:])]
    st, sl, sb, sr = t // vr, l // hr, -(-b // vr), -(-r // hr)
    tt, tl, tb, tr = st * vr, sl * hr, sb * vr, sr * hr
    cs, ct = source[:, :, st:sb, sl:sr], target[:, :, tt:tb, tl:tr]
    return cs, ct, (st, sl, sb, sr), (tt, tl, tb, tr)


def projected_conv(weight, direction):
    if len(weight.shape) == 5:
        cosine_map = torch.einsum('goiyx, di -> godyx', weight, direction)
        result = torch.einsum('godyx, di -> goiyx', cosine_map, direction)
    else:
        cosine_map = torch.einsum('oiyx, di -> odyx', weight, direction)
        result = torch.einsum('odyx, di -> oiyx', cosine_map, direction)
    return result


def rank_one_conv(weight, direction):
    cosine_map = (weight * direction[None, :, None, None]).sum(1, keepdim=True)
    return cosine_map * direction[None, :, None, None]


def zca_from_cov(cov):
    evals, evecs = torch.symeig(cov.double(), eigenvectors=True)
    zca = torch.mm(torch.mm(evecs, torch.diag
                            (evals.sqrt().clamp(1e-20).reciprocal())),
                   evecs.t()).to(cov.dtype)
    return zca

"""Detection-box instance copy-paste for spall; preserve all contained annotations.

Box labels do not provide foreground masks, so the complete rectangular instance
crop is pasted. Original segmentation CopyPaste remains available for mask data.
"""
from copy import deepcopy
import random
import numpy as np
from ultralytics.data import augment
from ultralytics.utils.instance import Instances

OriginalCopyPaste=augment.CopyPaste

class BoxInstanceCopyPaste(OriginalCopyPaste):
    def __call__(self, labels):
        self.last_pasted=0
        if len(labels['instances'].segments):return super().__call__(labels)
        if self.p<=0 or random.random()>=self.p:return labels
        image=labels['img'];h,w=image.shape[:2]
        instances=deepcopy(labels['instances']);instances.convert_bbox('xyxy');instances.denormalize(w,h)
        boxes=instances.bboxes;classes=labels['cls']
        candidates=np.flatnonzero(classes.reshape(-1)==1).tolist();random.shuffle(candidates)
        for j in candidates:
            x1,y1,x2,y2=boxes[j];x1=max(0,int(np.floor(x1)));y1=max(0,int(np.floor(y1)));x2=min(w,int(np.ceil(x2)));y2=min(h,int(np.ceil(y2)))
            if min(x2-x1,y2-y1)<4 or (x2-x1)*(y2-y1)>.4*w*h:continue
            overlap=(np.minimum(boxes[:,2],x2)>np.maximum(boxes[:,0],x1)) & (np.minimum(boxes[:,3],y2)>np.maximum(boxes[:,1],y1))
            inside=(boxes[:,0]>=x1)&(boxes[:,1]>=y1)&(boxes[:,2]<=x2)&(boxes[:,3]<=y2)
            # Never paste an unlabelled fragment of an object cut by the crop boundary.
            if np.any(overlap & ~inside):continue
            dx1,dx2=w-x2,w-x1
            destination_overlap=(np.minimum(boxes[:,2],dx2)>np.maximum(boxes[:,0],dx1)) & (np.minimum(boxes[:,3],y2)>np.maximum(boxes[:,1],y1))
            # Keep every existing annotation visible; no synthetic occlusion labels.
            if destination_overlap.any():continue
            selected=np.flatnonzero(inside);added=deepcopy(instances[selected]);added.fliplr(w)
            result=image.copy();result[y1:y2,dx1:dx2]=image[y1:y2,x1:x2][:,::-1]
            labels['img']=result;labels['cls']=np.concatenate([classes,classes[selected]],axis=0)
            labels['instances']=Instances.concatenate([instances,added],axis=0)
            self.last_pasted=len(selected)
            return labels
        return labels

def install():
    augment.CopyPaste=BoxInstanceCopyPaste

def self_test():
    # A spall crop contains its rebar label. Both must be mirrored together.
    image=np.zeros((64,96,3),np.uint8);image[10:30,8:30]=[11,70,135]
    boxes=np.array([[8,10,30,30],[12,14,24,18]],np.float32)
    labels=dict(img=image,cls=np.array([[1],[4]],np.float32),instances=Instances(boxes,np.zeros((0,1000,2),np.float32),bbox_format='xyxy',normalized=False))
    op=BoxInstanceCopyPaste(p=1);out=op(deepcopy(labels))
    assert op.last_pasted==2 and out['cls'].reshape(-1).tolist()==[1,4,1,4]
    assert np.array_equal(out['img'][10:30,66:88],image[10:30,8:30][:,::-1])
    assert np.array_equal(out['instances'].bboxes[2:],[[66,10,88,30],[72,14,84,18]])
    assert np.array_equal(labels['instances'].bboxes,boxes) and np.all(image[:,40:]==0)
    # A destination occupied by another instance must not be overwritten.
    blocked=deepcopy(labels);blocked['instances']=Instances(np.concatenate([boxes,[[65,9,89,31]]]),np.zeros((0,1000,2),np.float32),bbox_format='xyxy',normalized=False);blocked['cls']=np.array([[1],[4],[0]],np.float32)
    out=op(blocked);assert op.last_pasted==0 and len(out['cls'])==3
    return dict(passed=True,box_copy_paste_active=True,contained_labels_preserved=True,existing_instances_not_occluded=True,original_image_unmodified=True)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 19:14:14 2022

@author: abhi
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage


sobel=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])


half_circle_filters_L=[]
half_circle_filters_R=[]


leung_malik_sigma_s_x=np.array([1,np.sqrt(2),2,2*np.sqrt(2)])
leung_malik_sigma_l_x=np.array([np.sqrt(2),2,2*np.sqrt(2)])

rotation=[]   
#def Rotation(i):
leung_malik_filter=[]
gabor_filter=[]
filters=[]

rotation=np.array(rotation)
size=29
sigma=1
#filter1=rotation@sobel
ax=np.linspace(-(size-1)/2,(size-1)/2,size)
ax,ay=np.meshgrid(ax,ax)
sigma_x=5
sigma_y=3*sigma_x
leung_malik_filter=[]

def gaussian_(sigma_x,scale=1):
    sigma_y=scale*sigma_x
    gaussian=(np.exp(-(np.square(ax)/(2*sigma_x*sigma_x)+np.square(ay)/(2*sigma_y*sigma_y))))/(np.sqrt(2*np.pi*sigma_x*sigma_y))
    return gaussian


def DOG():
    rotation_DOG=[]  
    i=0
    for scale in range(1,3):
        gaussian=gaussian_(scale,1)
        for angle in range(0,360,45):
            
            x=ndimage.convolve(gaussian,sobel)
            M=cv2.getRotationMatrix2D(((size-1)/2,(size-1)/2),angle,1)
            final_x=cv2.warpAffine(x,M,(size,size))
            rotation_DOG.append(final_x)
            
            #plt.axis("off")
            i=i+1
            
            
            plt.subplot(2,8,i)
            plt.imshow(final_x,cmap='gray')
            #print(angle)
            plt.axis("off")
            
    #filters=filters+rotation_DOG
    plt.savefig('../figure/DOG_filter.png')
    plt.show(block=False)    

    plt.pause(1)
    plt.close()    
    return np.array(rotation_DOG)
    







def LM_filter():
    i=0
    leung_malik_filter=[]
    Laplacian=np.array([[0,1,0],[1,-4,1],[0,1,0]])
    for scale in leung_malik_sigma_s_x:
        gaussian=gaussian_(scale,3)
        for angle in range(0,180,30):
            
            x=ndimage.convolve(gaussian,sobel)
            M=cv2.getRotationMatrix2D(((size-1)/2,(size-1)/2),angle,1)
            final_x=cv2.warpAffine(x,M,(size,size))
            y_orientation=ndimage.convolve(x,sobel)
            M=cv2.getRotationMatrix2D(((size-1)/2,(size-1)/2),angle,1)
            y_orientation=cv2.warpAffine(y_orientation,M,(size,size))
            leung_malik_filter.append(final_x)
            leung_malik_filter.append(y_orientation)
            
            #plt.axis("off")
            i=i+1
            
            
            plt.subplot(4,2*6,i)
            plt.imshow(final_x,cmap='gray')
            #print(angle)
            plt.axis("off")
            
            plt.subplot(4,2*6,6+i)
            plt.imshow(y_orientation,cmap='gray')
            #print(angle)
            plt.axis("off")
        i=i+6
    
    
    
    i=36    
    LOG_sigma_s_x=np.array([1,np.sqrt(2),2,2*np.sqrt(2)])  
    for j in range(1,4,2):
        for scale in LOG_sigma_s_x:
            gaussian =gaussian_(scale*j)
            #print(i)
            output_LOG=ndimage.convolve(gaussian,Laplacian)        
            i=i+1
            
            plt.subplot(4,2*6,i)
            plt.imshow(output_LOG,cmap='gray')
            #print(angle)
            plt.axis("off")
            
            leung_malik_filter.append(output_LOG)
    
    for scale in LOG_sigma_s_x:
        gaussian =gaussian_(scale)
        #print(i)
        #gaussian=ndimage.convolve(gaussian,Laplacian)        
        i=i+1
        
        plt.subplot(4,2*6,i)
        plt.imshow(gaussian,cmap='gray')
        #print(angle)
        plt.axis("off")
        leung_malik_filter.append(output_LOG)
        
    #filters=filters+leung_malik_filter
    leung_malik_filter=np.array(leung_malik_filter)
    plt.savefig('../figure/Leung_Malik_filter.png')
    plt.show(block=False)
    
    plt.pause(1)
    plt.close()
    return np.array(leung_malik_filter)        
    



def gabor():
    gabor_filter=[]
    sigma_gabor_array=[3,5,10,15]
    lambda_array=[3,5,7,9]
    gamma=1
    j =0
    for sigma_gabor,lambda_ in zip(sigma_gabor_array,lambda_array):
        for angle in range(0,179,int(45/2)):
                
            x_=ax*np.cos(angle*np.pi/180)+ay*np.sin(angle*np.pi/180)
            y_=-ax*np.sin(angle*np.pi/180)+ay*np.cos(angle*np.pi/180)
        
            gabor=(np.exp(-(np.square(x_)+np.square(gamma*y_))/(2*sigma_gabor*sigma_gabor)))*np.cos(2*np.pi*x_/(lambda_))
            gabor_filter.append(gabor)
            j=j+1
            #print(j)
            plt.subplot(4,9,j)
            plt.imshow(gabor,cmap='gray')
            plt.axis("off")
            #plt.show()
    plt.savefig('../figure/Gabor_filter.png')
    plt.show(block=False)  
    
    plt.pause(1)
    plt.close()
    #filters=filters+gabor_filter
    return np.array(gabor_filter)




#from scipy import ndimage
#x=ndimage.convolve(gaussian,sobel)
#x=ndimage.convolve(x,sobel)
#orientation_45=np.cos(90*np.pi/180)*sobel_x+np.sin(90*np.pi/180)*sobel_y
#sigma_x

from sklearn.cluster import KMeans
#import time
#s=time.time()


def texton_map(filters,img):
    texture=[]
    for i in range(filters.shape[0]):
        text=cv2.filter2D(img,-1,filters[i,:,:])
        text=text.reshape(text.shape[0]*text.shape[1])
        texture.append(text)    
    return np.array(texture)
    



def texture_id(texture,h,w,bins,txt=0):
    ext,_=im.split('.')

    if(txt==0):
        if(len(texture.shape)==2):
            n=1
            h,w=texture.shape
            name='../figure/texton_B'+ext+'.png'
        elif(len(texture.shape)==3):
            n=3
            h,w,c=texture.shape
            name='../figure/texton_C'+ext+'.png'
            
        texton=KMeans(n_clusters=bins,random_state=0).fit_predict(texture.reshape(h*w,n))
    else:
        name='../figure/texton_T'+ext+'.png'
        texton=KMeans(n_clusters=bins,random_state=0).fit_predict(texture.T)
    plt.imshow(texton.reshape((h,w)),cmap='hsv')
    plt.axis("off")
    plt.imsave(name,texton.reshape(h,w),cmap="hsv")

    plt.show(block=False)
    plt.pause(1)
    plt.close()
    return texton.reshape((h,w))
        


#h, w,c = img.shape
#texture=texture.reshape(h,w,nf)
#print(nf)
#texture=texture.reshape( (h*w,nf))
    


#M = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)

#y=ndimage.convolve(x,rotation[3])
#scales_hd=[2,3,4,5]
def half_disc():
    half_circle_filters_L=[]
    half_circle_filters_R=[]
    #scales_hd=[3,5,10,15]
    scales_hd=[1,3,5,7]
    #def half_disc_generator():
    j=0
    #i=0
    for scale in scales_hd:
        #left_h
        r=scale
        
        left_half_disc=np.zeros((2*r+1,2*r+1))
        x_c=(np.arange(-r,r+1,1).reshape(2*r+1,1))
        
        y_c=x_c.T
        
        #y_c[y_c>=0]=20
        mask=(x_c**2+y_c**2 <=r**2)  
        
        left_half_disc[mask]=1
        right_half_disc=left_half_disc.copy()
        left_half_disc[:,r+1:]=0
        right_half_disc[:,:r]=0
    #def circle_half()
        for angle in range(0,180,int(45/2)):
            
            M=cv2.getRotationMatrix2D(((2*r+1-1)/2,(2*r+1-1)/2),angle,1)
            final_LD=cv2.warpAffine(left_half_disc,M,(2*r+1,2*r+1))
            final_RD=cv2.warpAffine(right_half_disc,M,(2*r+1,2*r+1))
            #orientation_45=cv2.warpAffine(x,M,(size,size))
            half_circle_filters_L.append(final_LD.astype(np.uint8))
            half_circle_filters_R.append(final_RD.astype(np.uint8))
            
            #plt.imshow(left_half_disc)
            j=j+1
            
            plt.subplot(4,18,j)
            
            plt.imshow(final_LD,cmap='gray')
            plt.axis("off")
            j=j+1
            plt.subplot(4,18,j)
            plt.imshow(final_RD,cmap='gray')
            plt.axis("off")
    plt.savefig('../figure/half_circle.png')
    plt.show(block=False)
    
    plt.pause(1)
    plt.close()
    return half_circle_filters_L,half_circle_filters_R
            



import scipy.signal

#bins=16
import cv2
def gradients(Tg,half_circle_filters_L,half_circle_filters_R,bins):
    #final_tg=[None]*len(half_circle_filters_L)
    final_tg=np.zeros((len(half_circle_filters_L),Tg.shape[0],Tg.shape[1]))
    for i in range(len(half_circle_filters_L)):
        chi_f=np.zeros(Tg.shape)
        for b in range(bins):
            bin_img=np.zeros(Tg.shape)
            bin_mask=np.isin(Tg,b)
            bin_img[bin_mask]=1
            #lg=ndimage.convolve(Tg,half_circle_filters_L[i])
            lg =cv2.filter2D(bin_img,-1,  half_circle_filters_L[i])
            rg =cv2.filter2D(bin_img, -1, half_circle_filters_R[i])
            #rg=ndimage.convolve(Tg,half_circle_filters_R[i])
            #print(i)
            chi=np.square(lg-rg)/(lg+rg+0.0000001)
            chi_f=chi+chi_f
        final_tg[i,:,:]=chi_f
    return np.mean(final_tg,axis=0)

'''
bins=16
final_tg=np.zeros((len(half_circle_filters_L),Cg.shape[0],Cg.shape[1]))
for i in range(len(half_circle_filters_L)):
    chi_f=np.zeros(Tg.shape)
    for b in range(bins):
        bin_img=np.zeros(Cg.shape)
        bin_mask=np.isin(Cg,b)
        bin_img[bin_mask]=1
        #lg=ndimage.convolve(Cg,half_circle_filters_L[i])
        lg =scipy.signal.convolve2d(bin_img,  half_circle_filters_L[i], mode = 'same')
        rg =scipy.signal.convolve2d(bin_img,  half_circle_filters_R[i], mode = 'same')
        
        #rg=ndimage.convolve(Cg,half_circle_filters_R[i])
        chi=np.square(lg-rg)/(lg+rg+0.00001)
        chi_f=chi+chi_f
    final_tg[i,:,:]=chi_f
fx=(np.array(final_tg))
'''




        #bin_img[i//4:(i//4)+1,i%bnp.sqrt(bins):np.sqrti%4]
#Tgf=255*(Tgf-np.min(Tgf))/(np.max(Tgf)-np.min(Tgf))

#img_color
w1=0.5
w2=0.5


import os 
dir_parent=os.path.dirname(os.getcwd())
dir_image_folder= dir_parent+'/BSDS500/Images'
image_name=os.listdir(dir_image_folder)
#image_paths=dir_image_folder+image_name
#for i  in image_name:
#    image_paths=dir_image_folder+i
#left_half_disc[]
def main(dir_parent,im):
    
    img=cv2.imread(dir_parent+'/BSDS500/Images/'+im)
    img_color=img
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    rotation_DOG=DOG()
    leung_malik_filter=LM_filter()
    gabor_filter=gabor()
    filters=np.concatenate([rotation_DOG,leung_malik_filter,gabor_filter])
    
    texture=texton_map(filters,img)
    h,w=img.shape  
    Tg=texture_id(texture,h,w,64,1)
    #plt.imshow(texton.reshape((h,w)))
    #plt.savefig()
    Bg=texture_id(img,h,w,16)
    Cg=texture_id(img_color,h,w,16)
    
    
    Tg=Tg.astype(np.uint8)
    Cg=Cg.astype(np.uint8)
    Bg=Bg.astype(np.uint8)
    
    #plt.show()


    half_circle_filters_L,half_circle_filters_R=half_disc()
    
    #half_circle_filters_L=half_circle_filters_L.astype(np.uint8)
    #half_circle_filters_R=half_circle_filters_R.astype(np.uint8)
        
    ext,_=im.split('.')  
    Tgf=gradients(Tg,half_circle_filters_L,half_circle_filters_R,64)
    Cgf=gradients(Cg,half_circle_filters_L,half_circle_filters_R,16)
    Bgf=gradients(Bg,half_circle_filters_L,half_circle_filters_R,16)
    plt.imshow(Tgf,cmap='hsv')
    plt.axis("off")
    plt.imsave('../figure/Tgf_'+ext+'.png',Tgf,cmap="hsv")
    
    plt.show(block=False)
    
    plt.pause(1)
    plt.close()
    
    plt.imshow(Bgf,cmap='hsv')
    plt.axis("off")
    plt.imsave('../figure/Bgf_'+ext+'.png',Bgf,cmap="hsv")
    
    plt.show(block=False)
    
    plt.pause(1)
    plt.close()
    
    plt.imshow(Cgf,cmap='hsv')
    plt.axis("off")
    plt.imsave('../figure/Cgf_'+ext+'.png',Cgf,cmap="hsv")
    
    plt.show(block=False)
    
    plt.pause(1)
    plt.close()

    canny=cv2.imread(dir_parent+'/BSDS500/CannyBaseline/'+ext+'.png',cv2.IMREAD_GRAYSCALE)
    sobelPb=cv2.imread(dir_parent+'/BSDS500/SobelBaseline/'+ext+'.png',cv2.IMREAD_GRAYSCALE)
    so_ca=w1*canny+w2*sobelPb
    Pb_edges=np.multiply((Tgf+Bgf+Cgf)/3,so_ca)
    plt.imshow(Pb_edges,cmap='gray')
    plt.imsave('../figure/pb_lite/'+ext+'.png',Pb_edges,cmap="gray")
    plt.axis("off")
    plt.show(block=False)
    
    plt.pause(1)
    plt.close()


if __name__ == '__main__':
    import time
    s=time.time()    
    for im in image_name:
        main(dir_parent,im)
    print(time.time()-s)

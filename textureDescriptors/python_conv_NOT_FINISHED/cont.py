# Generated with SMOP  0.41-beta
from libsmop import *
# 

#C computes the VAR descriptor.
# J = CONT(I,R,N,LIMS,MODE) returns either a rotation invariant local 
# variance (VAR) image or a VAR histogram of the image I. The VAR values 
# are determined for all pixels having neighborhood defined by the input 
# arguments. The VAR operator calculates variance on a circumference of 
# R radius circle. The circumference is discretized into N equally spaced
# sample points. Function returns descriptor values in a continuous form or
# in a discrete from if the quantization limits are defined in the argument
# LIMS.
    
# Examples
# --------
    
#       im = imread('rice.png');
#       c  = cont(im,4,16); 
#       d  = cont(im,4,16,1:500:2000);
    
#       figure
#       subplot(121),imshow(c,[]), title('VAR image')
#       subplot(122),imshow(d,[]), title('Quantized VAR image')
    
    
@function
def cont(varargin=None,*args,**kwargs):
    varargin = cont.varargin
    nargin = cont.nargin

    # Version: 0.1.0
    
    # Check number of input arguments.
    error(nargchk(1,5,nargin))
    image=varargin[1]
    d_image=double(image)
    if nargin == 1:
        spoints=concat([[- 1,- 1],[- 1,0],[- 1,1],[0,- 1],[- 0,1],[1,- 1],[1,0],[1,1]])
        neighbors=8
        lims=0
        mode='i'
    
    if (nargin > 2) and (length(varargin[2]) == 1):
        radius=varargin[2]
        neighbors=varargin[3]
        spoints=zeros(neighbors,2)
        lims=0
        mode='i'
        a=dot(2,pi) / neighbors
        for i in arange(1,neighbors).reshape(-1):
            spoints[i,1]=dot(- radius,sin(dot((i - 1),a)))
            spoints[i,2]=dot(radius,cos(dot((i - 1),a)))
        if (nargin >= 4 and logical_not(ischar(varargin[4]))):
            lims=varargin[4]
        if (nargin >= 4 and ischar(varargin[4])):
            mode=varargin[4]
        if (nargin == 5):
            mode=varargin[5]
    
    if (nargin == 2) and ischar(varargin[2]):
        mode=varargin[2]
        spoints=concat([[- 1,- 1],[- 1,0],[- 1,1],[0,- 1],[- 0,1],[1,- 1],[1,0],[1,1]])
        neighbors=8
        lims=0
    
    # Determine the dimensions of the input image.
    ysize,xsize=size(image,nargout=2)
    miny=min(spoints(arange(),1))
    maxy=max(spoints(arange(),1))
    minx=min(spoints(arange(),2))
    maxx=max(spoints(arange(),2))
    # Block size, each LBP code is computed within a block of size bsizey*bsizex
    bsizey=ceil(max(maxy,0)) - floor(min(miny,0)) + 1
    bsizex=ceil(max(maxx,0)) - floor(min(minx,0)) + 1
    # Coordinates of origin (0,0) in the block
    origy=1 - floor(min(miny,0))
    origx=1 - floor(min(minx,0))
    # Minimum allowed size for the input image depends
    # on the radius of the used LBP operator.
    if (xsize < bsizex or ysize < bsizey):
        error('Too small input image. Should be at least (2*radius+1) x (2*radius+1)')
    
    # Calculate dx and dy;
    dx=xsize - bsizex
    dy=ysize - bsizey
    #Compute the local contrast
    
    for i in arange(1,neighbors).reshape(-1):
        y=spoints(i,1) + origy
        x=spoints(i,2) + origx
        fy=floor(y)
        cy=ceil(y)
        fx=floor(x)
        cx=ceil(x)
        ty=y - fy
        tx=x - fx
        w1=dot((1 - tx),(1 - ty))
        w2=dot(tx,(1 - ty))
        w3=dot((1 - tx),ty)
        w4=dot(tx,ty)
        N=dot(w1,d_image(arange(fy,fy + dy),arange(fx,fx + dx))) + dot(w2,d_image(arange(fy,fy + dy),arange(cx,cx + dx))) + dot(w3,d_image(arange(cy,cy + dy),arange(fx,fx + dx))) + dot(w4,d_image(arange(cy,cy + dy),arange(cx,cx + dx)))
        # ( http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#On-line_algorithm ).
        if i == 1:
            MEAN=zeros(size(N))
            DELTA=zeros(size(N))
            M2=zeros(size(N))
        DELTA=N - MEAN
        MEAN=MEAN + DELTA / i
        M2=M2 + multiply(DELTA,(N - MEAN))
    
    # Compute the variance matrix.
# Optional estimate for variance:
# VARIANCE_n=M2/neighbors;
    result=M2 / (neighbors - 1)
    # Quantize if LIMS is given
    if lims:
        q,r,s=size(result,nargout=3)
        quant_vector=q_(ravel(result),lims)
        result=reshape(quant_vector,q,r,s)
        if strcmp(mode,'h'):
            # Return histogram
            result=hist(result,length(lims) - 1)
    
    if strcmp(mode,'h') and logical_not(lims):
        # Return histogram
    #epoint = round(max(result(:)));
        result=hist(ravel(result),arange(0,10000.0,1))
    
    return result
    
if __name__ == '__main__':
    pass
    
    
@function
def q_(sig=None,partition=None,*args,**kwargs):
    varargin = q_.varargin
    nargin = q_.nargin

    nRows,nCols=size(sig,nargout=2)
    indx=zeros(nRows,nCols)
    for i in arange(1,length(partition)).reshape(-1):
        indx=indx + (sig > partition(i))
    
    return indx
    
if __name__ == '__main__':
    pass
    
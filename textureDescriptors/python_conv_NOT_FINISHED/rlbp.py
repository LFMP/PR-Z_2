# Generated with SMOP  0.41-beta
from libsmop import *
# 

    #LBP returns the local binary pattern image or LBP histogram of an image.
#  J = LBP(I,R,N,MAPPING,MODE) returns either a local binary pattern
#  coded image or the local binary pattern histogram of an intensity
#  image I. The LBP codes are computed using N sampling points on a 
#  circle of radius R and using mapping table defined by MAPPING. 
#  See the getmapping function for different mappings and use 0 for
#  no mapping. Possible values for MODE are
#       'h' or 'hist'  to get a histogram of LBP codes
#       'nh'           to get a normalized histogram
#  Otherwise an LBP code image is returned.
    
    #  J = LBP(I) returns the original (basic) LBP histogram of image I
    
    #  J = LBP(I,SP,MAPPING,MODE) computes the LBP codes using n sampling
#  points defined in (n * 2) matrix SP. The sampling points should be
#  defined around the origin (coordinates (0,0)).
    
    #  Examples
#  --------
#       I=imread('rice.png');
#       mapping=getmapping(8,'u2'); 
#       H1=LBP(I,1,8,mapping,'h'); #LBP histogram in (8,1) neighborhood
#                                  #using uniform patterns
#       subplot(2,1,1),stem(H1);
    
    #       H2=LBP(I);
#       subplot(2,1,2),stem(H2);
    
    #       SP=[-1 -1; -1 0; -1 1; 0 -1; -0 1; 1 -1; 1 0; 1 1];
#       I2=LBP(I,SP,0,'i'); #LBP code image using sampling points in SP
#                           #and no mapping. Now H2 is equal to histogram
#                           #of I2.
    
    
@function
def RLBP(varargin=None,*args,**kwargs):
    varargin = RLBP.varargin
    nargin = RLBP.nargin

    # Version 0.3.2
# Authors: Marko Heikkila and Timo Ahonen
    
    # Changelog
# Version 0.3.2: A bug fix to enable using mappings together with a
# predefined spoints array
# Version 0.3.1: Changed MAPPING input to be a struct containing the mapping
# table and the number of bins to make the function run faster with high number
# of sampling points. Lauge Sorensen is acknowledged for spotting this problem.
    
    # Check number of input arguments.
    narginchk(1,5)
    image=varargin[1]
    d_image=double(image)
    if nargin == 1:
        spoints=concat([[- 1,- 1],[- 1,0],[- 1,1],[0,- 1],[- 0,1],[1,- 1],[1,0],[1,1]])
        neighbors=8
        mapping=0
        mode='h'
    
    if (nargin == 2) and (length(varargin[2]) == 1):
        error('Input arguments')
    
    if (nargin > 2) and (length(varargin[2]) == 1):
        radius=varargin[2]
        neighbors=varargin[3]
        spoints=zeros(neighbors,2)
        a=dot(2,pi) / neighbors
        for i in arange(1,neighbors).reshape(-1):
            spoints[i,1]=dot(- radius,sin(dot((i - 1),a)))
            spoints[i,2]=dot(radius,cos(dot((i - 1),a)))
        if (nargin >= 4):
            mapping=varargin[4]
            if (isstruct(mapping) and mapping.samples != neighbors):
                error('Incompatible mapping')
        else:
            mapping=0
        if (nargin >= 5):
            mode=varargin[5]
        else:
            mode='h'
    
    if (nargin > 1) and (length(varargin[2]) > 1):
        spoints=varargin[2]
        neighbors=size(spoints,1)
        if (nargin >= 3):
            mapping=varargin[3]
            if (isstruct(mapping) and mapping.samples != neighbors):
                error('Incompatible mapping')
        else:
            mapping=0
        if (nargin >= 4):
            mode=varargin[4]
        else:
            mode='h'
    
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
    # Fill the center pixel matrix C.
    C=image(arange(origy,origy + dy),arange(origx,origx + dx))
    d_C=double(C)
    bins=2 ** neighbors
    # Initialize the result matrix with zeros.
    result=zeros(dy + 1,dx + 1)
    #Compute the LBP code image
    
    for i in arange(1,neighbors).reshape(-1):
        y=spoints(i,1) + origy
        x=spoints(i,2) + origx
        fy=floor(y)
        cy=ceil(y)
        ry=round(y)
        fx=floor(x)
        cx=ceil(x)
        rx=round(x)
        if (abs(x - rx) < 1e-06) and (abs(y - ry) < 1e-06):
            # Interpolation is not needed, use original datatypes
            N=image(arange(ry,ry + dy),arange(rx,rx + dx))
            D=N >= C
        else:
            # Interpolation needed, use double type images
            ty=y - fy
            tx=x - fx
            w1=dot((1 - tx),(1 - ty))
            w2=dot(tx,(1 - ty))
            w3=dot((1 - tx),ty)
            w4=dot(tx,ty)
            N=dot(w1,d_image(arange(fy,fy + dy),arange(fx,fx + dx))) + dot(w2,d_image(arange(fy,fy + dy),arange(cx,cx + dx))) + dot(w3,d_image(arange(cy,cy + dy),arange(fx,fx + dx))) + dot(w4,d_image(arange(cy,cy + dy),arange(cx,cx + dx)))
            D=N >= d_C
        # Update the result matrix.
        v=2 ** (i - 1)
        result=result + dot(v,D)
    
    x=load('tabela_RLBP.csv')
    vetor_final=zeros(1,mapping.num)
    #Apply mapping if it is defined
    if isstruct(mapping):
        bins=mapping.num
        for i in arange(1,size(result,1)).reshape(-1):
            for j in arange(1,size(result,2)).reshape(-1):
                #result(i,j) = mapping.table(result(i,j)+1);
                vetor_final[x(result(i,j) + 1,1) + 1]=vetor_final(x(result(i,j) + 1,1) + 1) + 1
                if x(result(i,j) + 1,2) >= 0:
                    vetor_final[x(result(i,j) + 1,2) + 1]=vetor_final(x(result(i,j) + 1,2) + 1) + 1
                    if x(result(i,j) + 1,3) >= 0:
                        vetor_final[x(result(i,j) + 1,3) + 1]=vetor_final(x(result(i,j) + 1,3) + 1) + 1
    
    if (strcmp(mode,'h') or strcmp(mode,'hist') or strcmp(mode,'nh')):
        # Return with LBP histogram if mode equals 'hist'.
    #result=hist(result(:),0:(bins-1));
        if (strcmp(mode,'nh')):
            result=vetor_final / sum(vetor_final)
    else:
        #Otherwise return a matrix of unsigned integers
        if ((bins - 1) <= intmax('uint8')):
            result=uint8(result)
        else:
            if ((bins - 1) <= intmax('uint16')):
                result=uint16(result)
            else:
                result=uint32(result)
    
    clear('x')
    return result
    
if __name__ == '__main__':
    pass
    
# Generated with SMOP  0.41-beta
from libsmop import *
# 

#GETMAPPING returns a structure containing a mapping table for LBP codes.
#  MAPPING = GETMAPPING(SAMPLES,MAPPINGTYPE) returns a
#  structure containing a mapping table for
#  LBP codes in a neighbourhood of SAMPLES sampling
#  points. Possible values for MAPPINGTYPE are
#       'u2'   for uniform LBP
#       'ri'   for rotation-invariant LBP
#       'riu2' for uniform rotation-invariant LBP.
    
#  Example:
#       I=imread('rice.tif');
#       MAPPING=getmapping(16,'riu2');
#       LBPHIST=lbp(I,2,16,MAPPING,'hist');
#  Now LBPHIST contains a rotation-invariant uniform LBP
#  histogram in a (16,2) neighbourhood.
    
    
@function
def getmapping(samples=None,mappingtype=None,*args,**kwargs):
    varargin = getmapping.varargin
    nargin = getmapping.nargin

# Version 0.2
# Authors: Marko Heikkil?, Timo Ahonen and Xiaopeng Hong
    
# Changelog
# 0.1.1 Changed output to be a structure
# Fixed a bug causing out of memory errors when generating rotation
# invariant mappings with high number of sampling points.
# Lauge Sorensen is acknowledged for spotting this problem.
    
# Modified by Xiaopeng HONG and Guoying ZHAO
# Changelog
# 0.2
# Solved the compatible issue for the bitshift function in Matlab
# 2012 & higher
    
    matlab_ver=ver('MATLAB')
    matlab_ver=str2double(matlab_ver.Version)
    if matlab_ver < 8:
        mapping=getmapping_ver7(samples,mappingtype)
    else:
        mapping=getmapping_ver8(samples,mappingtype)
    
    return mapping
    
if __name__ == '__main__':
    pass
    
    
@function
def getmapping_ver7(samples=None,mappingtype=None,*args,**kwargs):
    varargin = getmapping_ver7.varargin
    nargin = getmapping_ver7.nargin

    disp('For Matlab version 7.x and lower')
    table=arange(0,2 ** samples - 1)
    newMax=0
    
    index=0
    if strcmp(mappingtype,'u2'):
        newMax=dot(samples,(samples - 1)) + 3
        for i in arange(0,2 ** samples - 1).reshape(-1):
            j=bitset(bitshift(i,1,samples),1,bitget(i,samples))
            numt=sum(bitget(bitxor(i,j),arange(1,samples)))
            #0->1 transitions
                                                    #in binary string
                                                    #x is equal to the
                                                    #number of 1-bits in
                                                    #XOR(x,Rotate left(x))
            if numt <= 2:
                table[i + 1]=index
                index=index + 1
            else:
                table[i + 1]=newMax - 1
    
    if strcmp(mappingtype,'ri'):
        tmpMap=zeros(2 ** samples,1) - 1
        for i in arange(0,2 ** samples - 1).reshape(-1):
            rm=copy(i)
            r=copy(i)
            for j in arange(1,samples - 1).reshape(-1):
                r=bitset(bitshift(r,1,samples),1,bitget(r,samples))
                #left
                if r < rm:
                    rm=copy(r)
            if tmpMap(rm + 1) < 0:
                tmpMap[rm + 1]=newMax
                newMax=newMax + 1
            table[i + 1]=tmpMap(rm + 1)
    
    if strcmp(mappingtype,'riu2'):
        newMax=samples + 2
        for i in arange(0,2 ** samples - 1).reshape(-1):
            j=bitset(bitshift(i,1,samples),1,bitget(i,samples))
            numt=sum(bitget(bitxor(i,j),arange(1,samples)))
            if numt <= 2:
                table[i + 1]=sum(bitget(i,arange(1,samples)))
            else:
                table[i + 1]=samples + 1
    
    mapping.table = copy(table)
    mapping.samples = copy(samples)
    mapping.num = copy(newMax)
    return mapping
    
if __name__ == '__main__':
    pass
    
    
@function
def getmapping_ver8(samples=None,mappingtype=None,*args,**kwargs):
    varargin = getmapping_ver8.varargin
    nargin = getmapping_ver8.nargin

    #disp('For Matlab version 8.0 and higher');
    
    table=arange(0,2 ** samples - 1)
    newMax=0
    
    index=0
    if strcmp(mappingtype,'u2'):
        newMax=dot(samples,(samples - 1)) + 3
        for i in arange(0,2 ** samples - 1).reshape(-1):
            i_bin=dec2bin(i,samples)
            j_bin=circshift(i_bin.T,- 1).T
            numt=sum(i_bin != j_bin)
            #0->1 transitions
                                                    #in binary string
                                                    #x is equal to the
                                                    #number of 1-bits in
                                                    #XOR(x,Rotate left(x))
            if numt <= 2:
                table[i + 1]=index
                index=index + 1
            else:
                table[i + 1]=newMax - 1
    
    if strcmp(mappingtype,'ri'):
        tmpMap=zeros(2 ** samples,1) - 1
        for i in arange(0,2 ** samples - 1).reshape(-1):
            rm=copy(i)
            r_bin=dec2bin(i,samples)
            for j in arange(1,samples - 1).reshape(-1):
                r=bin2dec(circshift(r_bin.T,dot(- 1,j)).T)
                if r < rm:
                    rm=copy(r)
            if tmpMap(rm + 1) < 0:
                tmpMap[rm + 1]=newMax
                newMax=newMax + 1
            table[i + 1]=tmpMap(rm + 1)
    
    if strcmp(mappingtype,'riu2'):
        newMax=samples + 2
        for i in arange(0,2 ** samples - 1).reshape(-1):
            i_bin=dec2bin(i,samples)
            j_bin=circshift(i_bin.T,- 1).T
            numt=sum(i_bin != j_bin)
            if numt <= 2:
                table[i + 1]=sum(bitget(i,arange(1,samples)))
            else:
                table[i + 1]=samples + 1
    
    mapping.table = copy(table)
    mapping.samples = copy(samples)
    mapping.num = copy(newMax)
    return mapping
    
if __name__ == '__main__':
    pass
    
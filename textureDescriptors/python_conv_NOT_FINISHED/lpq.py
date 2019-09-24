# Generated with SMOP  0.41-beta
from libsmop import *
# 


#from matlabFunctions import *
    
@function
def lpq(img=None,winSize=None,decorr=None,freqestim=None,mode=None,*args,**kwargs):
    varargin = lpq.varargin
    nargin = lpq.nargin

# Funtion LPQdesc=lpq(img,winSize,decorr,freqestim,mode) computes the Local Phase Quantization (LPQ) descriptor
# for the input image img. Descriptors are calculated using only valid pixels i.e. size(img)-(winSize-1).
    
# Inputs: (All empty or undefined inputs will be set to default values)
# img = N*N uint8 or double, format gray scale image to be analyzed.
# winSize = 1*1 double, size of the local window. winSize must be odd number and greater or equal to 3 (default winSize=3 ou 7 pelo Yandre).
# decorr = 1*1 double, indicates whether decorrelation is used or not. Possible values are:
#                      0 -> no decorrelation, 
#            (default) 1 -> decorrelation
# freqestim = 1*1 double, indicates which method is used for local frequency estimation. Possible values are:
#               (default) 1 -> STFT with uniform window (corresponds to basic version of LPQ)
#                         2 -> STFT with Gaussian window (equals also to Gaussian quadrature filter pair)
#                         3 -> Gaussian derivative quadrature filter pair.
# mode = 1*n char, defines the desired output type. Possible choices are:
#        (default) 'nh' -> normalized histogram of LPQ codewords (1*256 double vector, for which sum(result)==1)
#                  'h'  -> un-normalized histogram of LPQ codewords (1*256 double vector)
#                  'im' -> LPQ codeword image ([size(img,1)-r,size(img,2)-r] double matrix)
    
# Output:
# LPQdesc = 1*256 double or size(img)-(winSize-1) uint8, LPQ descriptors histogram or LPQ code image (see "mode" above)
    
# Example usage:
# img=imread('cameraman.tif');
# LPQhist = lpq(img,3);
# figure; bar(LPQhist);
    
# Version published in 2010 by Janne Heikkila, Esa Rahtu, and Ville Ojansivu 
# Machine Vision Group, University of Oulu, Finland
    
## Defaul parameters
# Local window size
    if nargin < 2 or isempty(winSize):
        winSize=3
    
    # Decorrelation
    if nargin < 3 or isempty(decorr):
        decorr=1
    
    rho=0.9
    
    # Local frequency estimation (Frequency points used [alpha,0], [0,alpha], [alpha,alpha], and [alpha,-alpha])
    if nargin < 4 or isempty(freqestim):
        freqestim=1
    
    STFTalpha=1 / winSize
    
    sigmaS=(winSize - 1) / 4
    
    sigmaA=8 / (winSize - 1)
    
    # Output mode
    if nargin < 5 or isempty(mode):
        mode='nh'
    
    # Other
    convmode='valid'
    
    ## Check inputs
    if size(img,3) != 1:
        error('Only gray scale image can be used as input')
    
    if winSize < 3 or (winSize%2 != 1):
        error('Window size winSize must be odd number and greater than equal to 3')
    
    if decorr not in [0,1]:
        error('decorr parameter must be set to 0->no decorrelation or 1->decorrelation. See help for details.')
    
    if freqestim not in [1,2,3] == 0:
        error('freqestim parameter must be 1, 2, or 3. See help for details.')
    
    if mode not in ['nh','h','im'] == 0:
        error('mode must be nh, h, or im. See help for details.')
    
    ## Initialize
    img=double(img)
    
    r=(winSize - 1) / 2
    
    x=arange(- r,r)
    
    u=arange(1,r)
    
    ## Form 1-D filters
    if freqestim == 1:
        # Basic STFT filters
        w0=(dot(x,0) + 1)
        w1=exp(complex(0,dot(dot(dot(- 2,pi),x),STFTalpha)))
        w2=conj(w1)
    else:
        if freqestim == 2:
            # Basic STFT filters
            w0=(dot(x,0) + 1)
            w1=exp(complex(0,dot(dot(dot(- 2,pi),x),STFTalpha)))
            w2=conj(w1)
            gs=exp(dot(- 0.5,(x / sigmaS) ** 2)) / (multiply(sqrt(dot(2,pi)),sigmaS))
            w0=multiply(gs,w0)
            w1=multiply(gs,w1)
            w2=multiply(gs,w2)
            w1=w1 - mean(w1)
            w2=w2 - mean(w2)
        else:
            if freqestim == 3:
                # Frequency domain definition of filters
                G0=exp(dot(- x ** 2,(dot(sqrt(2),sigmaA)) ** 2))
                G1=concat([zeros(1,length(u)),0,multiply(u,exp(dot(- u ** 2,sigmaA ** 2)))])
                G0=G0 / max(abs(G0))
                G1=G1 / max(abs(G1))
                w0=real(fftshift(ifft(ifftshift(G0))))
                w1=fftshift(ifft(ifftshift(G1)))
                w2=conj(w1)
                w0=w0 / max(abs(concat([real(max(w0)),imag(max(w0))])))
                w1=w1 / max(abs(concat([real(max(w1)),imag(max(w1))])))
                w2=w2 / max(abs(concat([real(max(w2)),imag(max(w2))])))
    
    ## Run filters to compute the frequency response in the four points. Store real and imaginary parts separately
# Run first filter
    filterResp=conv2(conv2(img,w0.T,convmode),w1,convmode)
    # Initilize frequency domain matrix for four frequency coordinates (real and imaginary parts for each frequency).
    freqResp=zeros(size(filterResp,1),size(filterResp,2),8)
    # Store filter outputs
    freqResp[arange(),arange(),1]=real(filterResp)
    freqResp[arange(),arange(),2]=imag(filterResp)
    # Repeat the procedure for other frequencies
    filterResp=conv2(conv2(img,w1.T,convmode),w0,convmode)
    freqResp[arange(),arange(),3]=real(filterResp)
    freqResp[arange(),arange(),4]=imag(filterResp)
    filterResp=conv2(conv2(img,w1.T,convmode),w1,convmode)
    freqResp[arange(),arange(),5]=real(filterResp)
    freqResp[arange(),arange(),6]=imag(filterResp)
    filterResp=conv2(conv2(img,w1.T,convmode),w2,convmode)
    freqResp[arange(),arange(),7]=real(filterResp)
    freqResp[arange(),arange(),8]=imag(filterResp)
    # Read the size of frequency matrix
    freqRow,freqCol,freqNum=size(freqResp,nargout=3)
    ## If decorrelation is used, compute covariance matrix and corresponding whitening transform
    if decorr == 1:
        # Compute covariance matrix (covariance between pixel positions x_i and x_j is rho^||x_i-x_j||)
        xp,yp=meshgrid(arange(1,winSize),arange(1,winSize),nargout=2)
        pp=concat([ravel(xp),ravel(yp)])
        dd=dist(pp,pp.T)
        C=rho ** dd
        q1=dot(w0.T,w1)
        q2=dot(w1.T,w0)
        q3=dot(w1.T,w1)
        q4=dot(w1.T,w2)
        u1=real(q1)
        u2=imag(q1)
        u3=real(q2)
        u4=imag(q2)
        u5=real(q3)
        u6=imag(q3)
        u7=real(q4)
        u8=imag(q4)
        M=concat([[ravel(u1).T],[ravel(u2).T],[ravel(u3).T],[ravel(u4).T],[ravel(u5).T],[ravel(u6).T],[ravel(u7).T],[ravel(u8).T]])
        D=dot(dot(M,C),M.T)
        A=diag(concat([1.000007,1.000006,1.000005,1.000004,1.000003,1.000002,1.000001,1]))
        U,S,V=svd(dot(dot(A,D),A),nargout=3)
        freqResp=reshape(freqResp,concat([dot(freqRow,freqCol),freqNum]))
        freqResp=(dot(V.T,freqResp.T)).T
        freqResp=reshape(freqResp,concat([freqRow,freqCol,freqNum]))
    
    ## Perform quantization and compute LPQ codewords
    LPQdesc=zeros(freqRow,freqCol)
    
    for i in arange(1,freqNum).reshape(-1):
        LPQdesc=LPQdesc + dot((double(freqResp(arange(),arange(),i)) > 0),(2 ** (i - 1)))
    
    ## Switch format to uint8 if LPQ code image is required as output
    if strcmp(mode,'im'):
        LPQdesc=uint8(LPQdesc)
    
    ## Histogram if needed
    if strcmp(mode,'nh') or strcmp(mode,'h'):
        LPQdesc=hist(ravel(LPQdesc),arange(0,255))
    
    ## Normalize histogram if needed
    if strcmp(mode,'nh'):
        LPQdesc=LPQdesc / sum(LPQdesc)
    
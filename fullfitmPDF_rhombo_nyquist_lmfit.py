import scipy
from scipy import interpolate
from scipy.optimize.minpack import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
from lmfit import Parameters, minimize, fit_report


from mcalculator_mod import calculateMPDF
from getmPDF import j0calc

def cv(x1,y1,x2,y2):
    '''
    Module to compute convolution of functions y1 and y2.
    
    Inputs: array y1, x1, y2, x2. Should have the same grid spacing to be safe.
    
    Returns: arrays ycv and xcv giving the convolution.
    '''
    dx=x1[1]-x1[0]
    ycv = dx*np.convolve(y1,y2,'full')
    xcv=np.linspace(x1[0]+x2[0],x1[-1]+x2[-1],len(ycv))

    return xcv,ycv

def costransform(q,fq,rmin=0.0,rmax=50.0,rstep=0.1): # does not require even q-grid
    '''
    Module to compute cosine Fourier transform of f(q). Uses direct integration rather than FFT and does not require an even q-grid.
    
    Inputs: array q (>=0 only), array f(q) to be transformed, optional arguments giving rmin, rmax, and rstep of output r-grid.
    
    Returns: arrays r and fr, where fr is the cosine Fourier transform of fq.
    '''
    lostep = int(np.ceil((rmin - 1e-8) / rstep))
    histep = int(np.floor((rmax + 1e-8) / rstep)) + 1
    r = np.arange(lostep,histep)*rstep
    qrmat=np.outer(r,q)
    integrand=fq*np.cos(qrmat)
    fr=np.sqrt(2.0/np.pi)*np.trapz(integrand,q)
    return r,fr


def fitfunc(x,lowerb,upperb,scale1,scale2,width,damp,rSr,Sr,para):
    '''
    x: meaningless array, can be simply np.array([0]), just needed to make the curve_fit module work
    scale1: scale factor of the correlated part of d(r) (coming from the ideal mPDF)
    scale2: scale factor of the "paramagnetic" part of d(r)
    width: smoothing factor when calculating the mPDF
    theta, phi: angles giving direction of "up"-spin in the cubic coordinate system
    damp: full-width half max of overall gaussian envelope applied to mPDF.
    '''
    [r,fr]=calculateMPDF(atoms,spins,uclist,rstep,rcalcmax,width) ### ideal f(r)
    rDr,Dr = cv(r,fr,rSr,Sr) ### correlated term in d(r)
    Dr = scale1*Dr
    Dr[:len(para)] += scale2*para ### adding paramagnetic term
    #th=(np.sin(qmax*r)-np.sin(qmin*r))/np.pi/r ### convolution to simulate effects of finite qmin and qmax
    #rDrcv, Drcv = cv(rDr,Dr,r,th)
    rDrcv,Drcv=rDr,Dr
    dampf=np.exp(-(rDrcv/2.0/damp)**2)
    Drcv=dampf*Drcv
    #rDrcv, Drcv = rDr,Dr
    return Drcv[np.logical_and(rDrcv>lowerb+0.5*rstep,rDrcv<=upperb+0.5*rstep)]

### Preparatory stuff
[dq,q1,q2] = [0.01,0.00,10.00]
q=np.arange(q1,q2,dq)   
fMn=j0calc(q,[0.422,17.684,0.5948,6.005,0.0043,-0.609,-0.0219])

r1,r2,dr=-5.0266734,5.027141,0.0897662 #nyquist grid spacing
rsr, sr=costransform(q,fMn,rmin=r1,rmax=r2,rstep=dr)
sr = np.sqrt(np.pi/2.0)*sr

rSr,Sr = cv(rsr,sr,rsr,sr)

para = -1.0*np.sqrt(2.0*np.pi)*np.gradient(Sr,rSr[1]-rSr[0]) ### paramagnetic term in d(r)

### Prepare details for mPDF calculation
uclist=np.arange(32)
rstep=dr #nyquist grid spacing
rmin=0.09
rmax=20.0
rcalcmin = 0
rcalcmax=25.0

qmin=0.85 ### bounds for convolution to simulate effects of finite qmin and qmax
qmax=35.0

### Prepare structure
print 'Preparing initial structure...'

### Rhombohedral structure

### lattice parameters of magnetic unit cell, and some basis transformation stuff
e1=np.array([-0.5,0.5,0])
e1n=e1/np.linalg.norm(e1)
e2=np.array([0,-0.5,0.5])
e2n=e2/np.linalg.norm(e2)
e3=np.array([1,1,1])
e3n=e3/np.linalg.norm(e3)

Rbasis=np.array([e1,e2,e3])
IRbasis=np.linalg.inv(Rbasis)

Rbasisn=np.array([e1n,e2n,e3n])

#Tvals=['15']
Tvals=['15','40','70','90','100','110','112p5','115','117p5','120','125','130','200','300']
rhombo20avals={'15':3.150321,'40':3.1501,'70':3.14879,'90':3.14742,'100':3.14643,'110':3.14517,'112p5':3.1444,'115':3.14342,'117p5':3.14299,'120':3.1429284,'125':3.143242,'130':3.1434673,'200':3.146904,'300':3.1513434} ### Nyquist refinement values
rhombo20cvals={'15':7.5944326,'40':7.59639,'70':7.60716,'90':7.61846,'100':7.6268,'110':7.63692,'112p5':7.64238,'115':7.65051,'117p5':7.65622,'120':7.6574473,'125':7.657119,'130':7.6568855,'200':7.65727,'300':7.6611397}

#T='112p5'
for T in Tvals:
    a=rhombo20avals[T]
    c=rhombo20cvals[T]

    latpars=np.array([[a],[a],[c]])

    distorted=Rbasisn*latpars

    cell=2*np.dot(IRbasis,distorted)

    ### positions of atoms in magnetic unit cell
    basis=np.array([[0,0,0],[0.5,0,0],[0.25,0.25,0],[0,0.5,0],[0.25,0.75,0],[0.5,0.5,0],[0.75,0.25,0],[0.75,0.75,0],[0,0.25,0.25],[0.25,0,0.25],[0,0.75,0.25],[0.25,0.5,0.25],[0.5,0.25,0.25],[0.75,0,0.25],[0.5,0.75,0.25],[0.75,0.5,0.25],[0,0,0.5],[0,0.5,0.5],[0.25,0.25,0.5],[0.5,0,0.5],[0.25,0.75,0.5],[0.5,0.5,0.5],[0.75,0.25,0.5],[0.75,0.75,0.5],[0,0.25,0.75],[0.25,0,0.75],[0,0.75,0.75],[0.25,0.5,0.75],[0.5,0.25,0.75],[0.75,0,0.75],[0.5,0.75,0.75],[0.75,0.5,0.75]])
    atomcell=np.dot(basis,cell)


    ### spin orientations in same order as atomic positions.

    svec=np.array([1,-1,0])
    svec=svec/np.linalg.norm(svec)
    theta=np.pi/2
    phi=-np.pi/4

    width=0.2
    damp=100


    S=2.5

    spincell=S*np.array([svec,-svec,-svec,-svec,svec,svec,svec,-svec,
    -svec,-svec,svec,svec,svec,svec,-svec,-svec,
    -svec,svec,svec,svec,-svec,-svec,-svec,svec,
    svec,svec,-svec,-svec,-svec,-svec,svec,svec])


    ### how big to make the box
    radius=40.
    dim1=np.round(radius/np.linalg.norm(cell[0]))
    dim2=np.round(radius/np.linalg.norm(cell[1]))
    dim3=np.round(radius/np.linalg.norm(cell[2]))

    ### generate the coordinates of each unit cell 
    latos=np.dot(np.mgrid[-dim1:dim1+1,-dim2:dim2+1,-dim3:dim3+1].transpose().ravel().reshape((2*dim1+1)*(2*dim2+1)*(2*dim3+1),3),cell)

    ### select points within a desired radius from origin
    latos=latos[np.where(np.apply_along_axis(np.linalg.norm,1,latos)<=(rcalcmax+10.0))]

    ## rearrange latos array so that [0,0,0] is the first one (for convenience)
    latos[np.where(np.all(latos==[0,0,0],axis=1))]=latos[0]
    latos[0]=np.array([0,0,0])

    ### create list of all Mn atomic positions and spin directions
    atoms=np.empty([len(latos)*len(atomcell),3])
    spins=np.empty([len(latos)*len(spincell),3])
    index=0
    for i in range(len(latos)):
        for j in range(len(atomcell)):
            atoms[index]=latos[i]+atomcell[j]
            spins[index] = spincell[j]
            index+=1

    ### Record positions of up-spins and down-spins to quickly change spin orientation
    ups=np.where(np.all(spins==S*svec,axis=1))
    downs=np.where(np.all(spins==-1.0*S*svec,axis=1))

    ### Experimental data
    dataFile='Tdep/results_R3m_20A_nyquist/R3m_20A_Ny_'+T+'K.diff'
    #dataFile='Tdep/results_R3m_20A/correctedData_stdFF_'+T+'K.txt'
    #dataFile='Tdep/results_R3m_20A/scaleCorrected_R3m_20A_'+T+'K.diff'
    expr,expDr,err = np.loadtxt(dataFile,unpack=True)


    print 'Refining...'

    def residual(pars, x, data=None):
        vals = pars.valuesdict()
        scalePara=vals['scalePara']
        scaleCorr=vals['scaleCorr']
        width=vals['width']
        damp=vals['damp']

        para = -1.0*np.sqrt(2.0*np.pi)*np.gradient(Sr,rSr[1]-rSr[0]) ### paramagnetic term in d(r)
        #paraExpGrid=np.interp(expr,rSr,para)
        
        model = fitfunc(x,rmin,rmax,scaleCorr,scalePara,width,damp,rSr,Sr,para)
        if data is None:
            return model
        return (model - data)

    fit_params = Parameters()
    fit_params.add('scalePara', value=3.77156,vary=False)
    fit_params.add('scaleCorr', value=np.random.uniform(5,25),min=0,max=30)
    fit_params.add('width', value=0.2,vary=False)
    fit_params.add('damp', value=100,vary=False)#1.5,min=1.0,max=100)


    x=np.array([0])
    data=expDr[np.logical_and(expr>rmin+0.5*rstep,expr<=rmax+0.5*rstep)]
    dataErr=err[np.logical_and(expr>rmin+0.5*rstep,expr<=rmax+0.5*rstep)]
    fit_kws={'sigma':err,'absolute_sigma':True}
    out = minimize(residual, fit_params, args=(x,), kws={'data':data},**fit_kws)

    fit = residual(fit_params, x)
    print fit_report(fit_params)


    rcomp = expr[np.logical_and(expr>rmin+0.5*rstep,expr<=rmax+0.5*rstep)]
    diff = data-fit
    chisq=np.sum((diff)**2/len(diff))
    print chisq

    rfull=expr[np.logical_and(expr>rcalcmin+0.5*rstep,expr<=rcalcmax+0.5*rstep)]
    datafull=expDr[np.logical_and(expr>rcalcmin+0.5*rstep,expr<=rcalcmax+0.5*rstep)]

    offset = 1.25*np.abs(np.min(data))

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(rfull,datafull,marker='o',mfc='none',mec='b',linestyle='none')
    ax.plot(rcomp,fit,'r-',lw=2)
    ax.plot(rcomp,np.zeros_like(rcomp)-offset,'k-',rcomp,diff-offset,'g-')
    ax.set_xlim(xmin=rcalcmin,xmax=rcalcmax)
    ax.set_xlabel('r ($\AA$)')
    ax.set_ylabel('d(r) ($\AA^{-2}$)')

#    plt.show()

    vals=fit_params.valuesdict()
    scalePara=vals['scalePara']
    scaleCorr=vals['scaleCorr']
    width=vals['width']
    damp=vals['damp']


    
    ###### provide options to save data
    fitstring=' Experimental data: '+dataFile+\
                '\n Chi-squared: '+str(chisq)+\
                '\n Correlated scale: '+str(scaleCorr)+\
                '\n Paramagnetic scale: '+str(scalePara)+\
                '\n Broadening factor: '+str(width)+\
                '\n Theta: '+str(theta)+\
                '\n Phi: '+str(phi)+\
                '\n Damp: '+str(damp)+\
                '\n '+\
                '\n Column format: r, Obs., Calc., Diff., Uncertainty'
    savefile='mPDFfit_R3m_nyquist_0-20_fixedParaScale_'+T+'.txt'
    #savefile='mPDFfit_correctedData_noDamp_R3m_0-20_'+T+'.txt'
    np.savetxt(savefile,np.column_stack((rcomp,data,fit,diff,dataErr)),header=fitstring)

    # if (raw_input('Save fitted function? (y/n) ')=='y'):
    ##### provide options to save data
        # fitstring=' Experimental data: '+dataFile+\
                    # '\n Chi-squared: '+str(chisq)+\
                    # '\n Correlated scale: '+str(scaleCorr)+\
                    # '\n Paramagnetic scale: '+str(scalePara)+\
                    # '\n Broadening factor: '+str(width)+\
                    # '\n Theta: '+str(theta)+\
                    # '\n Phi: '+str(phi)+\
                    # '\n Damp: '+str(damp)+\
                    # '\n '+\
                    # '\n Column format: r, Obs., Calc., Diff.'
        # savefile=raw_input('File name: ')
        # np.savetxt(savefile,np.column_stack((rcomp,data,fit,diff)),header=fitstring)

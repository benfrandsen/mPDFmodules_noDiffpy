import numpy as np
import matplotlib.pyplot as plt
from mcalculator import *

# Create the unit cell and atomic basis
a=4.446
MnOunitCell=np.array([[a,0,0],[0,a,0],[0,0,a]])
atomBasis=np.array([[0,0,0],[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]])

rmax=30.0
aXYZ=generateAtomsXYZ(MnOunitCell,atomBasis,rmax)

svec=2.5*np.array([1.0,-1.0,0])/np.sqrt(2)
k=np.array([0.5,0.5,0.5])
sXYZ=generateSpinsXYZ(MnOunitCell,aXYZ,aXYZ[0],k,svec)

# Calculate the mPDF
r,fr=calculatemPDF(aXYZ,sXYZ,rmax=30,psigma=0.15,qmin=0.1,qmax=30.)

# Calculate the unnormalized mPDF D(r)
q=np.arange(0,10,0.01)
ff=j0calc(q,[0.422,17.684,0.5948,6.005,0.0043,-0.609,-0.0219])
Dr=calculateDr(r,fr,q,ff)

# Plot the mPDF
fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(r,fr,'r-',r,Dr,'b-')
ax.set_xlabel('r ($\AA$)')
ax.set_ylabel('f ($\AA^{-2}$)')

plt.show()


# Now an alternative way to do it using the mPDFcalculator class:
mc=mPDFcalculator(MnOunitCell,atomBasis,rmax=30.0,svec=2.5*np.array([1.0,-1.0,0])/np.sqrt(2),kvec=np.array([0.5,0.5,0.5]),qmin=0.1,qmax=30.0,gaussPeakWidth=0.15)
mc.makeAtoms()
mc.spinOrigin=mc.atoms[0]
mc.makeSpins()

mc.ffqgrid=np.arange(0,10,0.01)
mc.ff=j0calc(mc.ffqgrid,[0.422,17.684,0.5948,6.005,0.0043,-0.609,-0.0219]),

r,fr,Dr=mc.calc(both=True)
fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(r,fr,'r-',r,Dr,'b-')
ax.set_xlabel('r ($\AA$)')
ax.set_ylabel('f ($\AA^{-2}$)')

plt.show()


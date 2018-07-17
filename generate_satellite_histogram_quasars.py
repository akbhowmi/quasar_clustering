import numpy
import numpy as np
import pickle
import itertools
from readsubhalo import *
from kdcount import correlate
import scipy
root_folder='/physics/yfeng1/mb2'
from multiprocessing import Pool
global z,mid,width,rvir
import scipy.interpolate
from scipy.integrate import quad
snapshots=['085','083','079','073','068','063','058']
#,'055','050','042','039','037','035','034','031','029','027','026','025','024','023','022','021','020','019','018','017','016','015','014','013','012','011']

NBINS=25
log_ymin=-3
log_ymax=1
BOXSIZE=100
WRAP=True
y_space=numpy.logspace(log_ymin,log_ymax,NBINS+1)


c=3e8
mass_unit_conv=1e10/980/1e6
mass_sun=2e30
yr_to_sec=3.15e7
lamb=0.1
h=0.7
joule_to_ergs=1e7
total_conv=mass_unit_conv*mass_sun/yr_to_sec*c**2*joule_to_ergs*lamb


data=numpy.loadtxt('../comparison_with_observations/k_correction_g_band.txt')
#plt.plot(data[:,0],data[:,1])
K= scipy.interpolate.interp1d(data[:,0],data[:,1],fill_value='extrapolate')
global c,h,omm,oml,omk
omk=0
oml=0.693
omm=0.307
cc=3*10**5
h=0.697

def DC(z0, z1):
    # Comoving distance in Mpc                                                                                          $
    def integrand(z):
        return 1.0/EE(z)
    return cc/(100.0)*quad(integrand,z0,z1)[0]

def EE(z):
        #normalized hubble parameter                                                                                    $
    return np.sqrt(omm*(1.0+z)**3 + oml + omk*(1.0+z)**2)

def DL(z):
    #Luminosity distance in Mpcs                                                                                        $
    return (1.0+z)*DC(0,z)

def DM(z):
        #Distance Modulus                                                                                               $
        return 5*np.log10((DL(z)*1e6)/10.0)
def m_to_f_SDSS(m):
    return 10**((m-22.5)/(-2.5))


def mtoM(m,z):
    print DM(z)
    print z
    return m - DM(z) - K(z)

def Mtom(M, z):
    
    return M + DM(z) + K(z)

def LtoM(L):
    M = -2.5 * np.log10(L*1e-7)+34.1
    return M

def MtoL(M):
    return 10**((-1/2.5*(M-34.1)))*1e7


def correlate_info(data1,data2, NBINS = NBINS, RMIN=1, RMAX=2, BOXSIZE = BOXSIZE, WRAP = WRAP):
    if data1 is not None:
        if RMAX is None:
            RMAX = BOXSIZE

        if WRAP:
            wrap_length = BOXSIZE
        else:
            wrap_length = None

        dataset1 = correlate.points(data1, boxsize = wrap_length)
        dataset2 = correlate.points(data2, boxsize = wrap_length)

        binning = correlate.RBinning(np.logspace(np.log10(RMIN),np.log10(RMAX),NBINS+1))

        DD = correlate.paircount(dataset1,dataset2, binning, np=0)
        DD = DD.sum1
        N=len(dataset1)-1
        
#        if (sum(DD)!=N):
#            print data1,data2
        
        return DD,N
    else:
        return None, None,None
    

def get_DD(group_index):
        sub_halo_SM=h[group_index]['massbytype'][:,4]*1e10
        sub_halo_pos=h[group_index]['pos'][:]/1e3
        extract_central=sub_halo_SM==max(sub_halo_SM)
        central_pos=sub_halo_pos[extract_central]
        
        temp=np.array([sum(dat) for dat in sub_halo_pos])
        rem_nan=~np.isnan(temp)
        mask_SM=(sub_halo_SM>10**(lSM_bin-width))&(sub_halo_SM<10**(lSM_bin+width))
        sub_halo_pos=sub_halo_pos[mask_SM&rem_nan]

        #sub_halo_SM=sub_halo_SM[mask_SM&rem_nan]
        #print numpy.array(central_pos)
        #print sub_halo_pos
        
        
        return correlate_info(sub_halo_pos,central_pos,RMIN=10**log_ymin*rvir,RMAX=10**log_ymax*rvir)
        #return correlate_info(sub_halo_pos,sub_halo_pos,RMIN=10**log_ymin*rvir,RMAX=10**log_ymax*rvir)

def get_DD(group_index):
        global bhmdot,lSM_cut,z,bhmpos,M_cut,bol_to_Mr,bhmass

	Mr_space,L_bol_space=pickle.load(open('../comparison_with_observations/Mr_vs_L_bol.pickle'))
#	bol_to_Mr=scipy.interpolate.interp1d(numpy.log10(L_bol_space),Mr_space,fill_value='extrapolate')

#        M_cut=mtoM(FLUX_CUT,z)
        bhmdot_group=bhmdot[group_index]
        bhmpos_group=bhmpos[group_index]
        bhmass_group=bhmass[group_index]
        bhlum_group=bhmdot_group*total_conv
 

#	bhMr_group=bol_to_Mr(numpy.log10(bhlum_group))
        
        
        temp=np.array([sum(dat) for dat in bhmpos_group])
        rem_nan=~np.isnan(temp)

        mask_SM=bhlum_group>10**lcut
	
	if (len(bhlum_group)>0):
        	extract_central=bhmass_group==max(bhmass_group)
        	central_pos=bhmpos_group[extract_central]
        	bhmpos_group=bhmpos_group[mask_SM&rem_nan]
        	return correlate_info(bhmpos_group,central_pos,RMIN=10**log_ymin*rvir,RMAX=10**log_ymax*rvir)
	else:
		return numpy.array([0.]*NBINS)

def construct_halo_mass_bin(snapshot):
    global h,lSM_cut,z,bhmdot,bhmpos,indices,bhmass,Nhalo
    snap = SnapDir(snapshot, root_folder)
    z=snap.redshift
    g = snap.readgroup()
    group_HM=g['massbytype'][:,1]*1e10
#    indices=np.arange(0,len(group_HM))

    bhmdot=snap.load(5,'bhmdot',g)
    bhmpos=snap.load(5,'pos',g)/1e3
    bhmass=snap.load(5,'mass',g)
    indices=np.arange(0,len(bhmdot))
    
    mask_HM_group=(group_HM>10**(mid-width))&(group_HM<10**(mid+width))
    group_indices=indices[mask_HM_group]
    Nhalo=len(group_indices)
    p=Pool(16)
    out=p.map(get_DD,group_indices)
    p.close()
    #DD_total,N=get_DD(group_indices[0])
    DD_total=numpy.array([0.]*NBINS)
    N=0
    for o in out:
        DD_total+=o[0]
        N+=o[1]
    return DD_total,N
    
    #
    #sub_halo_pos=h['pos'][:]/1e3
    #mask_SM=sub_halo_SM>10**lSM_cut
    


def poiss(rmin,rmax):
    p=4./3*scipy.pi*(rmax**3-rmin**3)/BOXSIZE**3
    return p


rho_crit=2.775e11
mids=[12.5,12.0,11.5,11.0]
lcuts=[41,41.5,42,42.5,43,43.5,44,44.5]
width=0.25
for snapshot in reversed(snapshots):
    for mid in mids:
        for lcut in lcuts:
            rvir=(10**mid/(4./3*3.14*200*rho_crit*0.2814))**(1./3)
            DD,N=construct_halo_mass_bin(snapshot)
            y_cen=correlate.RBinning(y_space).centers

    	    dy=y_cen*numpy.diff(numpy.log(y_cen))[0]
    	    dr=rvir*dy
    	    r_cen=rvir*y_cen
    	    pre_factor=4.*3.14
#    print da

#    	    print sum(no_of_sat)
    	    density=DD/dr/r_cen**2/pre_factor
    	    density_err=numpy.sqrt(DD)/dr/r_cen**2/pre_factor
    	    pickle.dump([y_cen,r_cen,DD,density,density_err,Nhalo],open('raw_satellite_histograms_z_%.2f_mid_%.1f_lcut_%.1f.pickle'%(z,mid,lcut),'w'))
















#            pickle.dump([y_space_cen,DD],open('raw_satellite_histograms_z_%.2f_mid_%.1f_FLUX_CUT_%.1f.pickle'%(z,mid,FLUX_CUT),'w'))
    

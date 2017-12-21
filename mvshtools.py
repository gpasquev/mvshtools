#!/usr/bin/env python
#coding: utf-8

import matplotlib.pyplot as pyp
import numpy as np
import lmfit

""" 
**********
mvshtools
**********
--------------------------------------
Simple tools for MvsH cycle operation
--------------------------------------

This python-module gives some tools to quantitatively analyze magnetization vs. 
magnetic-applied-field cycles.

    i-   Split conventional two branches loops into two independent branches.
    ii-  Retrieve magnetization saturation using asymptotic behavior. 
    iii- Calculate (and remove) the superimposed lineal contribution. (:func:`removepara`)
    iv-  Calculate coercivity field.
    v-   Calculate initial susceptibility. 
    iv-  Make Chantrell Popplewelwell and Charles analysis (:func:`cpc`)         

Control figures can be disabled macking global variable FIGS False::

    >>> mvshtools.FIGS = False



Dependencies
-------------
numpy, lmfit, matplotlib


"""

# ------------------------------------------------------------------------------
# I have make a big effort to put all the docstrings
# in English. However, most part comments in the code are still in Spanish. 
# I should also apologize because the quality of my English. Sorry. GAP.
# ------------------------------------------------------------------------------
__author__  = 'Gustavo Pasquevich'
__version__ = 0.171031  # History at the end of module.
_debug      = False
FIGS        = True
NUMFIG      = 9000         # variable for function __newfig__

 

def splitcycle(H,M):
    """ Separete a M vs H cycle into two branches.

        H  and M  are expect to  be  close MvsH cycle,  without  initial 
        magentization curve. Both are  lists or numpy.arrays of the same length.

        Returns:: 
            
            H1, M1, H2, M2 
    
        where "1" indicate negative  dH/dt branch while "2" indicate the other 
        one.
    """
    # ------------------------------------------------------------------------------
    # SEPARA EL LOS DATOS EN AMBAS RAMAS:
    #
    # H1 y M1 son la rama desde H_max a H_min, y H2 y M2 correspondene a la rama 
    # siguiente, de Hmin a Hmax.
     
    if H[0] > H[len(H)/2]:     # Rama inicial decreciente
        _ind_ret = np.where(H == min(H))[0]
        if _debug:
            print _ind_ret
        if len(_ind_ret) == 2:
            if (_ind_ret[1]-_ind_ret[0]) == 1:
                i0 = _ind_ret[0]
                i1 = _ind_ret[1]
        else:
            i0 = _ind_ret[0]
            i1 = i0

        if _debug:
            print i0, i1

        H1 = H[:i0+1]
        M1 = M[:i0+1]
        H2 = H[i1:]
        M2 = M[i1:]
    else:
        _ind_ret = np.where(H == max(H))[0]
        if len(_ind_ret) == 2 and (_ind_ret[1]-_ind_ret[0]) == 1:
            i0 = _ind_ret[0]
            i1 = _ind_ret[1]
        H2 = H[:i0+1]
        M2 = M[:i0+1]
        H1 = H[i1:]
        M1 = M[i1:]

    return H1,M1,H2,M2

def fitfunc(pars, H, data=None, eps=None):
    """
    Function to be fitted to dataset. Prepared for lmfit module. 

                          (      a       b    )     
    y  =  Ms  * sign(H) * (1 - ----- - -----  ) +  Xi * H   + offset
                          (     |H|     H^2   )   

    This function represents the asintotic behaviour of a magnetic 
    ferromagnetic component in addition to a lineal contribution, Xi*x.

    """
    # unpack parameters:
    # extract .value attribute for each parameter
    Ms = pars['Ms'].value
    Xi = pars['Xi'].value
    ydc = pars['offset'].value
    a = pars['a'].value
    b = pars['b'].value

    y = (Ms*np.sign(H))*(1-abs(a)/abs(H)-abs(b)/H**2) + Xi*H + ydc

    if data is None:
        return y
    if eps is None:
        return (y - data)
    return (y - data)/eps



def __fourpoints__(P1,P2,P3,P4):
    """ Given four points P1, P2 P3 and P4 of the M vs H curve this function
        obtain apoximate charactesristic parameters of the curve.


                             _ ----------     ^
                            / P3       P4     | 
                           /                  | Step
                         _-                   | 
                --------                      |
               P1     P2                      v 


        P1 and P2 should be points in the negative and saturated part of the 
        curve while P3,and P4 in the positive part. This function returns 
        stmators for "step", superimpose linear contribution, and offset.
    """


    for k in [P1,P2,P3,P4]:
        #print 'puntos del triángulo::::: ',k
        if FIGS:
            pyp.plot(k[0],k[1],'o')
    # vector a = P1P2 con origen en P1 y punto final en P2
    a=[P2[0]-P1[0],P2[1]-P1[1]]
    # vector b = P1P3 con origen en P1 y punto final en P3
    b=[P3[0]-P1[0],P3[1]-P1[1]]

    x1,y1 = P1
    x2,y2 = P2
    x3,y3 = P3
    x4,y4 = P4

    d3 = (y3 - y1)-(y2 - y1)*(x3 -x1)/(x2-x1)  # d stands for delta ($\delta M$)
    d4 = (y4 - y1)-(y2 - y1)*(x4 -x1)/(x2-x1)
    d1 = (y1 - y4)-(y3 - y4)*(x1 -x4)/(x3-x4)
    d2 = (y2 - y4)-(y3 - y4)*(x2 -x4)/(x3-x4)

    slope12 = (y2 -y1)/(x2-x1)
    slope34 = (y3 -y4)/(x3-x4)

    step = (-d1-d2+d3+d4)/4.
    slope = (slope12 + slope34)/2.
    centro = ((y2-y1)/(x2-x1)*(-x1) + (y3-y4)/(x3-x4)*(-x4) + (y1+y4) )/2.

    return step, slope, centro


def Xi_and_Hc(H1,M1,H2,M2,limx=188):
    """ Initial susceptibility and coercive magnetic field.

        args:
        H1,M1 determine the cycle downward-branch [dH/dt < 0] 
        H2,M2 determine the cycle rising-branch [dH/dt >0] 

        kwarg:
        limx, H-limit to perform the linear fit. It is performed 
              in [-limx,limx]. 
       """

    j1 = np.where(np.abs(H1)<limx)[0]
    p1 = np.poly1d(np.polyfit(H1[j1],M1[j1],1))
    j2 = np.where(np.abs(H2)<limx)[0]
    p2 = np.poly1d(np.polyfit(H2[j2],M2[j2],1))

    X = np.mean([p1[1],p2[1]])
    Hc = (-p2[0]/p2[1]+p1[0]/p1[1] )/2.

    print '=============================================='
    print 'Initial Susceptibility and coercive field' 
    print '----------------------------------------------'
    print ' SLOPE: dM/dH at M=0:                         '
    print ' [dH/dt<0--branch] :', p1[1]
    print ' [dH/dt>0--branch] :', p2[1]
    print ' mean slope:       :', X
    print '                                              '
    print ' Zero croosing magnetic field, H at M=0: '
    print ' [dH/dt<0--branch]:', -p1[0]/p1[1] 
    print ' [dH/dt>0--branch]:', -p2[0]/p2[1]
    print ' mean-half-difference: (Hc1-Hc2)/2: Hc=',Hc,'Oe'
    print '=============================================='

    if FIGS:
        __newfig__(500)
        pyp.cla()
        pyp.plot(H1,M1,'.-r',label='dH/dt<0')
        pyp.plot(H1,p1(H1),color='green')
        pyp.plot(H2,M2,'.-b',label='dH/dt>0')
        pyp.plot(H2,p2(H2),color='green')
        pyp.axvline(-limx,color='k')
        pyp.axvline( limx,color='k')
        pyp.xlim([-188,188])
        pyp.ylim( [ np.min( [ M1[j1].min() , M2[j2].min() ] ) , np.max( [ M1[j1].max() , M2[j2].max() ] ) ] )
        pyp.axhline(0,color='k')
        pyp.axvline(0,color='k')
        pyp.legend(loc=0)

    return X,Hc


# INICIO PROCESO DE AJUSTE RAMA dH/dt < 0 *************************************
def linealcontribution(H,M,HLIM,E=None,label = 'The M vs H Curve',
                       fixed=dict(),initial=dict()):
    """
    This function determines the lineal contribution in the M vs H curve. As 
    well as the asymptotic behaviour at high fields. See :func:`fitfunc`.

    Returns
    =======
    lmfit.Parameter instance result of the tail-branch fit of magnetization 
    curve with :func:`fit func`

    IMPORTANT
    ========= 
    **H** and **M** input arguments corresponds only to one of the two 
    MvsH-cycle branchs. For a complete cycle it should be called twice, one 
    time for each branch. 

    Arguments
    =========
    args:
    -----    
    1) H, magnetic field. 1d-numpy array.  
    2) M, magnetization.  1d-numpy array.
    3) HLIM = [L1,L2]
        L1 and L2 are two limits defining the fitting zone. The analisis is
        done in the regions [-L1,-L2] and [L2,L1], i.e where  L2 < |H| < L1.

    kwargs:
    -------
    1) eps: weight values for fittng process. Numpy-array same size as **H**. 
    1) label: label for the figures. kwarg for pyplot.plot(). 
              [FIGS global variable should be True]
    2) fixed: dictionary with the parametesrs that should be fixed:
              e.g. fixed = {'Ms':value,'a':0,'b':None}
                
              valid keys: 'Ms','Xi', 'offset', 'a' and 'b'
              values can be numbers (the values), or None which indicate 
              that the value of the parametesr is the automatically obtained 
              but it should be a fix parameter.
    3) initial: dictinary with initail values. If None (or not defined) then 
              they are guessed.              
       
    """

    # Control de entrada, si H esta ordenado de manera que diff H > 0 lo invierte:
    #INVERT = False      #<<< SUPUESTA VARIABLE INSERVIBLE
    if H[-1] > H[0]:
        H = H[::-1]
        M = M[::-1]
        #INVERT = True    #<<< SUPUESTA VARIABLE INSERVIBLE

    if E is None:
        E = np.ones(H.size)

    H_LIM_1, H_LIM_2 = HLIM # BUSCAMOS INDICES INICIO Y FINAL REGIONES DE SATURACIÓN.
                            # Las rectas de aprox. lineal se tomaran analizando el ciclo
                            # entre los campos H_LIM_2 y H_LIM_1.
    
    # L1 es el límite de alto campo. L1a corresponde a la región de campos positivos
    # y L1b a la de negativos.
    if H_LIM_1 >= max(abs(H)):
        L1a = 0
        L1b = len(H)-1
    else:
        L1a = np.where( H > H_LIM_1)[0][-1]        
        L1b = np.where( H < -H_LIM_1)[0][0]

    # L2 es el limite de bajo campo. L2a corresponde a la region de campos positivos
    # y L2b a la de negativos.
    L2a = np.where( H > H_LIM_2)[0][-1]     
    L2b = np.where( H < -H_LIM_2)[0][0]

    if FIGS:
        __newfig__(249)
        pyp.plot(H,M,'.-')
        pyp.axvline(H[L1a],color='k')
        pyp.axvline(H[L1b],color='k')
        pyp.axvline(H[L2a],color='r')
        pyp.axvline(H[L2b],color='r')
        pyp.title('Posiciones de las cotas para %s'%label)

    # Estimation of initial parameters
    salto, pendiente, centro  = __fourpoints__( [H[L1b],M[L1b]],
                                              [H[L2b],M[L2b]],
                                              [H[L2a],M[L2a]],
                                              [H[L1a],M[L1a]])
        
    #pini = [ salto/2.,pendiente, centro, 0.0001, 0.0001]



    h_fit = np.concatenate([H[L1a:L2a],H[L2b:L1b]])
    m_fit = np.concatenate([M[L1a:L2a],M[L2b:L1b]])
    e_fit = np.concatenate([E[L1a:L2a],E[L2b:L1b]])

    # inicio del trabajo con lmfit ----------------------------
    # 
    # Armamos diccionario de parámetros. lmfit.Parameters
    params = lmfit.Parameters()
    params.add_many(('Ms',salto/2., True, None,None,None),
                    ('Xi',pendiente,True, None,None,None),
                    ('offset',centro,True, None,None,None),
                    ('a',0.0079,True, None,None,None),
                    ('b',0.0001,True, None,None,None))

    # Se fijan y definen los parámetros segun las entrdas *fixed* y *initial*
    for k in initial.keys():
        params[k].value = initial[k]
    for k in fixed.keys():
        params[k].vary = False
        if fixed[k] != None:
            params[k].value = fixed[k]
    


    if FIGS:
        __newfig__(250)
        pyp.axvline(H[L1a],color='k')
        pyp.axvline(H[L1b],color='k')
        pyp.axvline(H[L2a],color='r')
        pyp.axvline(H[L2b],color='r')
        pyp.plot(H,M,'.')
        pyp.plot(H,fitfunc(params,H),'-r',lw=2,alpha=0.8,label='pre fit curve')

    out = lmfit.minimize(fitfunc, params, args=(h_fit,m_fit,e_fit), kws=None, method='leastsq')

    print 'Information on the pre fit and pos fit parameters'
    print '-------------------------------------------------'
    for k in params.values():
        print k 

    lmfit.report_fit(out.params) # imprime reporte de los parámetros y el ajsute


    if FIGS:
        pyp.plot(h_fit,m_fit,'x',label='selected data to fit')
        pyp.ylim([M.min(),M.max()])
        pyp.legend(loc=0)
        pyp.title(label)  
        pyp.plot(H,fitfunc(out.params,H),color='k',lw=1,ls='--')
        pyp.ylim([M.min(),M.max()])
    
    return out.params


def removepara(H,M,Hmin = '1/2',Hmax = 'max'):
    """ Retrive lineal contribution to cycle and remve it from cycle.


        H y M corresponden a un ciclo completo. Es decir H comienza y termina
        en el mismo valor (o un valor aproximado).

        El ciclo M vs H se separa en sus dos ramas. H1,M1 y H2,M2, según:: 

            H1,M1: curva con dH/dt < 0. El campo decrece con el tiempo.
            H2,M2: curva con dH/dt > 0. El campo aumenta con el tiempo.

        Con la variable global FIGS = True muestra gráficas intermedias del 
        proceso de determinación y eliminación de la contribución lineal.

        La Figura 249 muestra las posiciones Hmin y Hmax en el ciclo. 

        Returns: H1,M1,H2,M2,[pendiente,salto,desp]
                
    """
    print '**********************************************************'
    print 'removepara '
    print '**********************************************************'

    if Hmax == 'max':
        Hmax = max(abs(H))
    if Hmin == '1/2':
        Hmin = 0.5*max(abs(H))

    H1,M1,H2,M2 = splitcycle(H,M)

    p1 = linealcontribution(H1,M1,[Hmax,Hmin],label='dH/dt < 0')
    p2 = linealcontribution(H2,M2,[Hmax,Hmin],label='dH/dt > 0')


    # Fin de ajustes 

    print 'pendiente 1:',p1['Xi']
    print 'pendiente 2:',p2['Xi']
    print 'salto 1    :',p1['Ms']
    print 'salto 2    :',p2['Ms']
    print 'despl.  1  :',p1['offset']
    print 'despl.  2  :',p2['offset']
    print 'a  1       :',p1['a']
    print 'a  2       :',p2['a']
    print 'b  1       :',p1['b']
    print 'b  2       :',p2['b']


    # Armamos una pendiente promedio a partir de la obtenida para cada rama.
    # Corregimos ambas ramas eliminando esta pendiente.

    pend =(p1['Xi']+p2['Xi'])/2.
    salto=(p1['Ms']+p2['Ms'])/2.
    desp =(p1['offset']+p2['offset'])/2.
    M1 = (M1-H1*pend)
    M2 = (M2-H2*pend)

    if FIGS:
        __newfig__()
        pyp.plot(H1,M1,'b.-',label = 'dH/dt < 0')
        pyp.plot(H2,M2,'r.-',label = 'dH/dt > 0')
        pyp.axhline(salto,color = 'k', alpha =0.5)
        pyp.axhline(-salto,color= 'k', alpha =0.5)
        pyp.legend(loc=0)
    return H1,M1,H2,M2,[pend,salto,desp]


def cpc(H, M, Hmin = '1/2', Hmax = 'max', clin=None, T=300, limx=10, 
              weight='None', rhr=False, aini=100.):
    """ Given a superparamagnetic cycle M vs H, assuming distribution of Lagevin 
        functions calculate <mu>, N and <mu^2> from M vs H cycle. 
        It use Chantrell, Popplewelwell, Charles proposal. 

        Arguments:
        ==========
        H and M: 
                They are assumed to be in cgs units. H in Oe and M in emu/sth.
                **sth** stands for something, indicating the fact that the
                moment or magnetization bvalues can be given by grams, cm^3,
                whatever or even nothing.
                H and M are np.arrays that with a whole cycle, i.e they include
                boths branches. 

        Returns: 
        ========
                H1, M1, H2, M2, [slop,step,offset]

        kwargs:
        =======
        Hmin,Hmax: 
                They define the high field region for the asymptotic analysis.
                **Hmin** can be a value or the string '1/2' (default), which 
                means is taken as the half of the maximum values of |H|.
                **Hmax**, can be a value or the string 'max' (default) which 
                indicates that Hmax must be taken as the maximum of |H|.

        clin:   propuesta de constante lineal a mantener constante 
                (if None [default], then is fitted) 

        a:      proposing initial value for "a" parametre in asympthotic process.
 
        T:      Temperature used for Chanterell calculations,
        limx :  max H field for calculation dM/dH at H=0.
        weight: String que indica el modo de tomar el peso de los datos en el 
                ajuste del comportamiento asintótico.
                    'None': Sin peso (o peso uniforme).
                    'sep' : inversamente proporcional a la separación entre 
                            puntos.
        rhr:    remove-H-remanenet. Before the analysis shift the values of H so 
                they not have remanent field. That is aceptable is measurement 
                provides of an SQUID, and is known that the sample behaves like
                superparamagnet. However here is only a technical parameter 
                to better suceptibility at low fields. 

        El ciclo M vs H se separa en sus dos ramas. H1,M1 y H2,M2, según:: 

            H1,M1: curva con dH/dt < 0. El campo decrece con el tiempo.
            H2,M2: curva con dH/dt > 0. El campo aumenta con el tiempo.

        Con la variable global FIGS = True muestra gráficas intermedias del 
        proceso de determinación y eliminación de la contribución lineal, los 
        cuales son útiles como herramientas para refinar los parámetros de 
        entrada. 

        It perform similar analysis as proposed by R. Chantrell, J. Popplewell 
        and S. Charles in "Measurements of particle size distribution parameters 
        in ferrofluids"  IEEE Transactions on Magnetics ( Volume: 14, Issue: 5, 
        September 1978). pag. 975 - 977 DOI:10.1109/TMAG.1978.1059918.  

        The function has been called first as the first author of the paper
        in which is base on. However, I thought it wasn't correct name only one 
        of the names of a three authors of the paper. So the functon name was 
        named by the first letter of each one of the authors: **cpc**. 

               
    """
    # Physics Constants
    kB  = 1.3806488e-16  #erg/K Boltzmann Constant in cgs units
    muB = 9.27400968e-21 #erg/G Bohr Magneton in cgs units.

    H1,M1,H2,M2 = splitcycle(H,M)

    if rhr:
        print 'rhr1'
        H1 = remove_H_remanent(H1,M1)
        print 'rhr2'
        H2 = remove_H_remanent(H2,M2)


    H1max = max(np.abs(H1))
    H2max = max(np.abs(H2))

    if Hmax == 'max':
        Hmax = max(H1max,H2max)
    if Hmin == '1/2':
        Hmin = 0.5*max(H1max,H2max)

     
    if weight == 'sep':
        E1 = __makeEdiff__(H1)
        E2 = __makeEdiff__(H2)
    else:
        E1 = None
        E2 = None

    # Low magnetic field analysis===============================================    
    X,HC = Xi_and_Hc(H1,M1,H2,M2,limx=limx)

    # ==========================================================================
    #Tail zone analysis of tail zone 
    fixed = dict()
    initial=dict()
    initial['a']=aini
    if clin is not None:
        fixed['Xi'] = clin
    fixed['b']  = 1e-10
    
    print('\n\nWorking whit dH/dt < 0 branch \n----------------------------- \n')
    p1 = linealcontribution(H1, M1, [Hmax,Hmin], label='dH/dt < 0', fixed=fixed, E=E1, initial = initial)
    print('\n\nWorking whit dH/dt > 0 branch \n----------------------------- \n')
    p2 = linealcontribution(H2,M2,[Hmax,Hmin],label='dH/dt > 0',fixed=fixed,E=E2,initial=initial)


    pend  = (p1['Xi']     + p2['Xi'])/2.
    salto = (p1['Ms']     + p2['Ms'])/2.
    desp  = (p1['offset'] + p2['offset'])/2.
    M1 = (M1-H1*pend)
    M2 = (M2-H2*pend)


    

    # Finish wit cycle anaysis 
    # ========================================================================= 
    print '========================'
    print 'cycle-analysis results'
    print '========================'
    print 'slope 1:',p1['Xi'].value
    print 'slope 2:',p2['Xi'].value
    print 'step 1    :',p1['Ms'].value
    print 'step 2    :',p2['Ms'].value
    print 'offset  1  :',p1['offset'].value
    print 'offset  2  :',p2['offset'].value
    print 'a  1       :',p1['a'].value
    print 'a  2       :',p2['a'].value
    print 'b  1       :',p1['b'].value
    print 'b  2       :',p2['b'].value
    print 'HC         :',HC
    print 'X          :',X



    mu1      = kB*T/p1['a']/muB
    mu2      = kB*T/p2['a']/muB
    mu       = (mu1+mu2)/2.
    N1       = p1['Ms']/mu1/muB
    N2       = p2['Ms']/mu2/muB
    N        = (N1+N2)/2.
    mumu     = (X-pend)*3*kB*T/N/muB**2 
    
    rhoA = mumu/mu**2
    STD      = np.sqrt(mumu-mu**2)
    sigma    = np.sqrt(np.log(rhoA))


    print('')
    print('----------------------------------')
    print('1: <mu> = %.1f muB'%(mu1))
    print('2: <mu> = %.1f muB'%(mu2))
    print('1: N    = %.2e'%(N1))
    print('2: N    = %.2e'%(N2))
    print('----------------------------------')
    print('<mu> .......... = %.1f mB'%(mu)) 
    print('sqrt(<mu^2>)... = %.1f mB'%(np.sqrt(mumu)))
    print('N ............. = %.2e num/sth.'%(N))
    print('rho ........... = %.2f '%(rhoA))
    print('STD ........... = %.1f mB'%(STD))
    print('sigma-lognormal = %.2f '%(sigma))
    print('----------------------------------')
    print('<mu>_mu ....... = %.1f mB'%(mumu/mu))


    # Armamos una pendiente promedio a partir de la obtenida para cada rama.
    # Corregimos ambas ramas eliminando esta pendiente.

    if FIGS:
        __newfig__()
        pyp.plot(H1,M1,'b.-',label = 'dH/dt < 0')
        pyp.plot(H2,M2,'r.-',label = 'dH/dt > 0')
        pyp.axhline(salto,color = 'k', alpha =0.5)
        pyp.axhline(-salto,color= 'k', alpha =0.5)
        pyp.legend(loc=0)

    return H1,M1,H2,M2,[pend,salto,desp]

def remove_H_remanent(H,M):
    """ H and M should be an unique branch. """
    # decreasing field franch
    if np.mean(np.diff(H)<0):
        Hr = np.interp(0,M[::-1],H[::-1])
    elif np.mean(np.diff(H)>0):
        Hr = np.interp(0,M,H)

    return H-Hr
    

def __makeEdiff__(H,smallvalue=1e-10 ):
    """ Calculated the inverse of the mean difference of each point with their
        neigbours. 
        All points have a value given by their two neigbours, except
        extrem points that have only a neigbour. 
        
        To avoid division by zero all values are incremented in samllvalue. 
        Default value should work without problem. """
    D = np.diff(H)        
    D1 = np.append(D[0],D)
    D2 = np.append(D,D[-1])
    D = (D1+D2)/2.+smallvalue 
    return 1/D


def __newfig__(num=None):
    """ Auxiliar function make new figures."""
    global NUMFIG
    if num == None:
        pyp.figure(NUMFIG)
        NUMFIG += 1
    else:
        pyp.figure(num)
        pyp.cla()

# ==============================================================================
# Historial de versiones
# =======================
# 171129     Se modifican docstrings pensando en subirlo a github
# 171031     Se agregan lineas al docstring. También hace unas semasn se agregó
#            el cálculo de Chantrell. 
# 1603       Inicio de mvsh_tools. Antes fuertemente heredado de mvsh_removepara.py
#
# 151211     CAMBIO IMPORTANTE se modifica para tener en cuenta cambio del 
#            funcionamiento de lmfit. Ahora funciona con la version 0.9.1. 
#            La diferencia esta en donde se encuentra el resultado del ajsute. 
#            lmfit cambia esa posición de la verisión posterior a la nueva. 
#            Todos los cambios ocurrieron en "linealcontribution"
#           
#            Se agrega diccionario *initial* a "linealcontribution".
#
# 150920     Mejoro el docstring de removepara y agrego líneas de Ms en la 
#            figura 9000.
#            Agrego la variable __version__
#               
# 28/11/13   Corrijo linealcontribution que tenía problemas con L1 y L2. Al final
#            parece que estaba bien. Así que quedo igual que antes, 
#            pero mejor explicado.
#            Cambio la rutina de ajuste de scipy.leastq a lmfit. 
#            Agrego el kwarg fixed a linealcontribution



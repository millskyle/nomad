

     ******************************************
     **    PROGRAM:              EXPTVL      **
     **    PROGRAM VERSION:      5.5.2b      **
     **    DISTRIBUTION VERSION: 5.9.a       **
     ******************************************

        Calculation of expectation values of one-electron properties.

 This Version of Program EXPTVL   is Maintained by:
     Thomas Mueller
     Juelich Supercomputing Centre (JSC)
     Institute of Advanced Simulation (IAS)
     D-52425 Juelich, Germany 
     Email: th.mueller@fz-juelich.de

 workspace allocation information: lcore=  65536000 mem1=          0 ifirst=          1

echo of the input file:
 ------------------------------------------------------------------------
  &input
   quadrup=1
   lvlprt=1,
   moment=2
   mofilen='mocoef'
   nofilen='mocoef_prop'
  &end
 ------------------------------------------------------------------------

echo of the input file:
 ------------------------------------------------------------------------
 ------------------------------------------------------------------------
 rdgeom: refpoint:  0.000000000000000E+000  0.000000000000000E+000
  0.000000000000000E+000

     Shifted molecular geometry:

 H     1.0    0.00000000    0.00000000   -0.70052142
 H     1.0    0.00000000    0.00000000    0.70052142

  Charge of molecule -0.00

Header information of AO-integral file:

Hermit Integral Program : SIFS version  localhost.localdo 22:20:23.079 02-Jan-16

irrep                  A  
basis functions        10
  basis function labels
   1:  1H1s         2:  2H1s         3:  3H1px        4:  4H1py        5:  5H1pz        6:  6H2s         7:  7H2s         8:  8H2px 
   9:  9H2py       10: 10H2pz 
One electron integral record length =   4096 real*8 words
Number of 1-e integrals per record  =   3272

 energy( 1)=  7.137540490910E-01, ietype=   -1,    core energy of type: Nuc.Rep.

nonzero typea=0 expectation values:
type             fcore              <type>               total
--------      ----------          ----------          ----------
S1(*)   :  0.000000000000E+00  2.000000000000E+00  2.000000000000E+00
T1(*)   :  0.000000000000E+00  1.083817284569E+00  1.083817284569E+00
V1(*)   :  0.000000000000E+00 -2.870109135005E+00 -2.870109135005E+00

total <h1> = -1.786291850436D+00


################################################################################

          O U T P U T:

          Diagonal dipole moment integrals in the NO basis.

   Symmetry:  A  
               X              Y              Z   

    1    -0.00000000    -0.00000000    -0.00000000
    2     0.00000000     0.00000000    -0.00000000
    3    -0.00000000    -0.00000000    -0.00862151
    4    -0.00000000    -0.00000000     0.00862151
    5     0.00000024     0.00000041     0.00000012
    6    -0.00000000    -0.00000041    -0.00010014
    7     0.00000026    -0.00000000     0.00385563
    8    -0.00000024     0.00000000    -0.00376119
    9     0.00000007     0.00074609    -0.00000067
   10    -0.00000033    -0.00074609     0.00000625


          Diagonal second moment integrals in the NO basis.

   Symmetry:  A  
               XX             XY             XZ             YY             YZ             ZZ  

    1     1.90066632     0.00000000    -0.00000000     1.90066632     0.00000000     5.95977417
    2     0.51201712     0.00000000    -0.00000000     0.51201712    -0.00000000     0.67595891
    3     0.90786962     0.00000000    -0.00000000     0.90786962    -0.00000000     2.99591258
    4     2.31119049     0.00000000     0.00000000     2.31119049     0.00000000     2.91421536
    5     1.15192610     0.00000000     0.00000002     1.15192610    -0.00002092     2.17577953
    6     0.34400426     0.00354513    -0.00000002     1.03151156     0.00000000     0.67334520
    7     1.03152991    -0.00028620    -0.00000001     0.34398591    -0.00000000     1.30583434
    8     1.03161838    -0.00356575    -0.00000000     0.34389744    -0.00000000     0.67325021
    9     0.40814145     0.00000000    -0.00000000     0.40817878     0.00000000     2.15981159
   10     0.34388258     0.00030682     0.00000001     1.03160289     0.00002092     1.30598261


  The following moments are calculated in a.u. relative to the point:      0.00000000      0.00000000      0.00000000

           Dipole moments:

                     X               Y               Z   
   nuclear       0.00000000      0.00000000      0.00000000
   electronic    0.00000000      0.00000000      0.00000000
   total         0.00000000      0.00000000      0.00000000
   total Dipole moment =   0.00000000    Debye

           Second moments:

                     XX              XY              XZ              YY  
   nuclear       0.00000000      0.00000000      0.00000000      0.00000000
   electronic   -2.41268931      0.00000000      0.00000000     -2.41268931
   total        -2.41268931      0.00000000      0.00000000     -2.41268931

                     YZ              ZZ  
   nuclear       0.00000000      0.98146052
   electronic    0.00000000     -6.63572780
   total         0.00000000     -5.65426728

           Quadrupole moment:  

                    QXX             QXY             QXZ             QYY             Q
   nuclear      -0.49073026      0.00000000      0.00000000     -0.49073026
   electronic    2.11151925      0.00000000      0.00000000      2.11151925
   total         1.62078899      0.00000000      0.00000000      1.62078899
  
                    QYZ             QZZ             Q
   nuclear       0.00000000      0.98146052
   electronic    0.00000000     -4.22303849
   total         0.00000000     -3.24157797

          Diagonal dipole moment integrals in the MO basis.

   Symmetry:  A  
               X              Y              Z   

    1     0.00000000     0.00000000     0.00000000
    2    -0.00000000    -0.00000000    -0.00000000
    3    -0.00000000     0.00000000     0.00000000
    4     0.00000000     0.00000000     0.00000000
    5    -0.00000000    -0.00000000    -0.00000000
    6    -0.00000000     0.00000000     0.00000000
    7     0.00000000    -0.00000000     0.00000000
    8     0.00000000    -0.00000000    -0.00000000
    9    -0.00000000    -0.00000000     0.00000000
   10     0.00000000    -0.00000000    -0.00000000


          Diagonal second moment integrals in the MO basis.

   Symmetry:  A  
               XX             XY             XZ             YY             YZ             ZZ  

    1     0.66524989    -0.00000000     0.00000000     0.66524989    -0.00000000     0.90121640
    2     1.86098926    -0.00000000    -0.00000000     1.86098926     0.00000000     5.84519552
    3     2.79862449     0.00000000     0.00000000     2.79862449     0.00000000     2.94866697
    4     1.01816369     0.00000000     0.00000000     1.01816369     0.00000000     3.46574616
    5     0.34839982     0.05557715    -0.00000000     1.02711600    -0.00000000     0.67324674
    6     1.02711600    -0.05557715     0.00000000     0.34839982    -0.00000000     0.67324674
    7     0.51133393     0.00000000     0.00000000     0.51133393     0.00000000     1.91606609
    8     1.03163673    -0.00030682    -0.00000000     0.34387909     0.00000000     1.30593626
    9     0.34387909     0.00030682    -0.00000000     1.03163673    -0.00000000     1.30593626
   10     0.33745333    -0.00000000    -0.00000000     0.33745333    -0.00000000     1.80460735



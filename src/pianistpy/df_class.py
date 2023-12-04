import sys
import math
from mpmath import hyp2f1
import scipy.integrate as integrate
from numpy import loadtxt

# compute the Euclidean distance between two spatial points
def distance_cal(self,reference_origin, point_of_interest):
   origin_x = float(reference_origin[0])
   origin_y = float(reference_origin[1])
   origin_z = float(reference_origin[2])
   pos_x    = float(point_of_interest[0])
   pos_y    = float(point_of_interest[1])
   pos_z    = float(point_of_interest[2])
   return math.sqrt((origin_x-pos_x)**2.0+(origin_y-pos_y)**2.0+(origin_z-pos_z)**2.0)
# parse strings of particle attributes
def par_attr_parse(self,pos_vector, vel_vector, CM_coords):
   pos_vector = pos_vector-CM_coords
   # (1) compute distance to the system's CM
   To_CM_distance = distance_cal([0.0,0.0,0.0], pos_vector)
   # (2) determine the radial component of the velocity via vector projection
   vel_rad = pos_vector*np.dot(vel_vector, pos_vector)/np.dot(pos_vector, pos_vector)
   # (3) determine the tangential component of the velocity via vector subtraction
   vel_tan = np.subtract(vel_vector, vel_rad)
   if (np.linalg.norm(vel_tan) > np.linalg.norm(vel_vector)):
      print("Error!!")
   # (4) output [radial distance to CM, \sigma_{radial}^2, \sigma_{tangential}^2]
   output_list = [To_CM_distance, np.linalg.norm(vel_rad)**2.0, np.linalg.norm(vel_tan)**2.0]
   return output_list
def round_off_rating(input_num):
   return round(float(input_num)*2.0)/2.0

# generic DF construction/truncation methods
class df_methods:
   #in units of "U_G = NFW_Mass_virial = NFW_R_scale = 1"
   def __init__(self, df_mode="Isotropic", Z_trun=0.0):
      """
      Initialize ... with ...

      Parameters
      ----------
      df_mode : (str) the choice of DF class from "Isotropic", "betap0p5", "betam0p5", or "OsipkovMerritt"
            Default: "Isotropic"
      Z_trun : (float) dimensionless DF truncation parameter that ranges from 0.0 (no truncation) to 1.0
            Default: 0.0
      """
      # input parameter range checks
      if (df_mode not in ["Isotropic", "betap0p5", "betam0p5", "OsipkovMerritt"]):
         print("%s is UNDEFINED and NOT in [Isotropic, betap0p5, betam0p5, OsipkovMerritt]\n"%df_mode)
         raise NotImplementedError

      if not (0.0 <= Z_trun < 1.0):
         print("0.0 <= %.2f < 1.0 is NOT satisfied\n"%Z_trun)
         raise NotImplementedError

      self.df_mode              = df_mode
      self.Z_trun               = Z_trun

   # truncate the input one-dimensional df_input(Z) at Z = Z_trun
   def df_trunc(self, Z, df_input):
      df_at_Z_trun = df_input(self.Z_trun)
      if (1.0-self.Z_trun >= Z >= 0.0):
            return df_input(Z+self.Z_trun) - df_at_Z_trun
      else:
            return 0.0

   def df_to_Potential_integrand(self, Z, Pmax):
      if (self.df_mode in ["Isotropic", "OsipkovMerritt"]): # RHS integrand in Eq. (15) of Drakos, Taylor, and A. J. Benson 2017
            if (Pmax >= Z >= 0.0):
               return df_trunc(Z)*(2.0*(Pmax-Z))**(0.5)
            else:
               return 0.0
      elif (self.df_mode == "betap0p5"): # Eq. (4.66) of Binney & Tremaine 2008
         if (Pmax >= Z >= 0.0):
            return df_trunc(Z)
         else:
            return 0.0
      elif (self.df_mode == "betam0p5"): # Eq. (4.70) of Binney & Tremaine 2008
         if (Pmax >= Z >= 0.0):
            return df_trunc(Z)*(Pmax-Z)
         else:
            return 0.0


   def df_to_Potential_integrand(self, Z, Pmax):
      if (self.df_mode in ["Isotropic", "OsipkovMerritt"]): # RHS integrand in Eq. (15) of Drakos, Taylor, and A. J. Benson 2017
         if (Pmax >= Z >= 0.0):
            return df_trunc(Z)*(2.0*(Pmax-Z))**(0.5)
         else:
            return 0.0
      elif (self.df_mode == "betap0p5"):# Eq. (4.66) of Binney & Tremaine 2008
         if (Pmax >= Z >= 0.0):
            return df_trunc(Z)
         else:
            return 0.0
      elif (self.df_mode == "betam0p5"): # Eq. (4.70) of Binney & Tremaine 2008
         if (Pmax >= Z >= 0.0):
            return df_trunc(Z)*(Pmax-Z)
         else:
            return 0.0

   def p_RHS(P): # construct an interpolated function for the RHS integral in Eq. (15) of Drakos, Taylor, and A. J. Benson 2017
      p_sampling = np.linspace(0.0,1.0-self.Z_trun,2.5e4,endpoint=True,dtype=float)
      inte_value = [0.0]                                    # the first entry integrating from 0 to 0
      for p_now in p_sampling[1:-1]:
         inte_value += [integrate.quad(lambda Z: df_to_Potential_integrand(Z,p_now), 0.0, p_now)[0]]
      inte_value += [inte_value[-1]*2.0 - inte_value[-2]]   # the approximated last entry, which tends to be numerically ill-behaved
      # the final interpolation function
      if (1.0-self.Z_trun >= P >= 0.0):
         return interp1d(p_sampling, inte_value, kind='cubic')(P)[()]
      else:
         return 0.0



# analytical/fitted DFs
class df_special:
   #in units of "U_G = NFW_Mass_virial = NFW_R_scale = 1"
   def df_isotropic_NFW_Lane2022(self,E): # dimensionless isotropic NFW DF from Lane, Bovy, and Mackereth 2022
      E2 = 1.0 - E
      if (1.0 > E > 0.0):
         return ((E**(1.5)/E2**(2.5))*(-math.log(E)/E2)**(-2.75)*(0.0926081086527954+0.0232272797489981*E
         +0.258346829924101*E**2.0-2.97848277491979*E**3.0+14.9776586391246*E**4.0-46.6587221550258*E**5.0
         +92.6397009471828*E**6.0-117.647787290798*E**7.0+92.5144063082258*E**8.0-41.0268009529576*E**9.0
         +7.84806318891231*E**10.0))
      else:
         return 0.0
   def df_Osipkov_Merritt_NFW_Lane2022(self,Q): # dimensionless Osipkov-Merritt anisotropic NFW DF from Lane, Bovy, and Mackereth 2022
      Q2 = 1.0 - Q
      if (1.0 > Q > 0.0):
         return (1.0/(Ra**2.0*Q**(2.0/3.0)))*(Q2/math.log(Q))**2.0*(0.0802975743915827-0.0408426627412238*Q
         -0.217968733177408*Q**2.0+0.831302363461598*Q**3.0-3.6920719890718*Q**4.0+7.03132348658788*Q**5.0
         -7.60690467091859*Q**6.0+4.29052661245253*Q**7.0-0.995895790138335*Q**8.0) + df_isotropic_NFW_Lane2022(Q)
      else:
         return 0.0
   def df_constant_anisotropy_NFW_beta_0p5(E): # dimensionless constant-anisotropy (beta = 0.5) NFW DF
      E2 = 1.0 - E
      if (1.0 > E > 0.0):
         return ((E/E2)*(-math.log(E)/E2)**(-1.818)*(0.04571587989836464+0.03208845828622633*E-0.03534507340877377*E**2.0
         -0.04927791659804768*E**3.0+0.19478642057773052*E**4.0-0.6462503895541302*E**5.0+1.1346302879113916*E**6.0
         -1.2600084471092619*E**7.0+0.8602282173899513*E**8.0-0.3320256387598663*E**9.0+0.055458201382062866*E**10.0))
      else:
         return 0.0

# generic density profile construction/truncation methods
class den_solver:#(...):
   #in units of "U_G = NFW_Mass_virial = NFW_R_scale = 1"
   def __init__(self, den_alpha=1.0, den_beta=3.0, den_gamma=1.0, concentration=10.0):
      '''
      Initialize ... with ...

      Parameters
      ----------
      den_alpha : (float) alpha index, i.e. the transition sharpness, of the double-power-law density profile
            Default: 1.0
      den_beta : (float) beta index, i.e. the outer logarithmic slope, of the double-power-law density profile
            Default: 3.0
      den_gamma : (float) beta index, i.e. the inner logarithmic slope, of the double-power-law density profile
            Default: 3.0
      concentration : (float) used to define (effective) r_vir = concentration*r_scale
            Default: 10.0
      '''
      # input parameter range checks
      if (0.5 > den_alpha):
         den_alpha = 0.5
         print("den_alpha is physical ONLY for >= 0.5\n")
         print("resetting den_alpha = %1f\n"%den_alpha)

      if (2.0 >= den_beta):
         den_beta = 2.0
         print("den_beta is physical ONLY for > 2.0\n")
         print("resetting den_beta = %1f\n"%den_beta)

      round_off_den_gamma       = round_off_rating(den_gamma)
      if (abs(round_off_den_gamma - den_gamma) > 1.0e-3 or not (0.0 <= round_off_den_gamma <= 1.5)):
         print("den_gamma is currently defined ONLY for \{0.0, 0.5, 1.0, 1.5\}\n")
         print("resetting den_gamma = %1f"%round_off_den_gamma)
      den_gamma                 = round_off_den_gamma # to be precise up to machine precision

      self.den_alpha            = den_alpha
      self.den_beta             = den_beta
      self.den_gamma            = den_gamma
      self.concentration        = concentration

   # unit normalization constant for the dimensionless potential of double-power-law density profiles
   def den_potential_normalization_C(self):
      if (abs(0.0 - self.den_gamma) > 1.0e-3):
         return ((self.den_beta-2.0)*math.gamma(self.den_beta/self.den_alpha))/(math.gamma(2.0/self.den_alpha)*math.gamma(((self.den_beta-2.0)/self.den_alpha)+1.0))
      elif (abs(0.5 - self.den_gamma) > 1.0e-3):
         return ((self.den_beta-2.0)*math.gamma((self.den_beta-0.5)/self.den_alpha))/(math.gamma(1.5/self.den_alpha)*math.gamma(((self.den_beta-2.0)/self.den_alpha)+1.0))
      elif (abs(1.0 - self.den_gamma) > 1.0e-3):
         return (math.gamma((self.den_beta-1.0)/self.den_alpha))/(math.gamma(1.0+(1.0/self.den_alpha))*math.gamma((self.den_beta-2.0)/self.den_alpha))
      elif (abs(1.5 - self.den_gamma) > 1.0e-3):
         return ((self.den_beta-2.0)*math.gamma((self.den_beta-1.5)/self.den_alpha))/(math.gamma(0.5/self.den_alpha)*math.gamma(((self.den_beta-2.0)/self.den_alpha)+1.0))

   # dimensionless potential for double-power-law density profiles
   def den_potential(self, R):
      if (abs(0.0 - self.den_gamma) > 1.0e-3):
         return (R**2.0/3.0)*(hyp2f1(3.0/self.den_alpha, self.den_beta/self.den_alpha, (3.0/self.den_alpha)+1.0, -R**self.den_alpha) + (3.0*R**(-self.den_beta)/(self.den_beta-2.0))
                              *hyp2f1((self.den_beta-2.0)/self.den_alpha, self.den_beta/self.den_alpha, ((self.den_beta-2.0)/self.den_alpha)+1.0, -R**(-self.den_alpha)))
      elif (abs(0.5 - self.den_gamma) > 1.0e-3):
         return (R**1.5)*(0.4*hyp2f1(2.5/self.den_alpha, (self.den_beta-0.5)/self.den_alpha, (2.5/self.den_alpha)+1.0, -R**self.den_alpha) + (R**(0.5-self.den_beta)/(self.den_beta-2.0))
                              *hyp2f1((self.den_beta-2.0)/self.den_alpha, (self.den_beta-0.5)/self.den_alpha, ((self.den_beta-2.0)/self.den_alpha)+1.0, -R**(-self.den_alpha)))
      elif (abs(1.0 - self.den_gamma) > 1.0e-3):
         return (math.gamma(1.0+(1.0/self.den_alpha))*math.gamma((self.den_beta-2.0)/self.den_alpha)/math.gamma((self.den_beta-1.0)/self.den_alpha)
                              +R*(0.5*hyp2f1(2.0/self.den_alpha, (self.den_beta-1.0)/self.den_alpha, (2.0/self.den_alpha)+1.0, -R**self.den_alpha)
                              -hyp2f1(1.0/self.den_alpha, (self.den_beta-1.0)/self.den_alpha, (1.0/self.den_alpha)+1.0, -R**self.den_alpha)))
      elif (abs(1.5 - self.den_gamma) > 1.0e-3):
         return (((2.0*R**0.5)/3.0)*hyp2f1(1.5/self.den_alpha, (self.den_beta-1.5)/self.den_alpha, (1.5/self.den_alpha)+1.0, -R**self.den_alpha) + (R**(2.0-self.den_beta)/(self.den_beta-2.0))
                              *hyp2f1((self.den_beta-2.0)/self.den_alpha, (self.den_beta-1.5)/self.den_alpha, ((self.den_beta-2.0)/self.den_alpha)+1.0, -R**(-self.den_alpha)))

   # dimensionless double-power-law density profiles
   def den_profile(self, R):
      return R**(-self.den_gamma)*(1.0+R**self.den_alpha)**((self.den_gamma-self.den_beta)/self.den_alpha)

   def NFW_density_Dimless(self,R): # dimensionless NFW density profile with "R = r_physical/NFW_R_scale"
      return 1.0/(R*(1+R)**2.0)
   def NFW_potential_relative_Dimless(self,R): # dimensionless NFW gravitational potential; see Widrow 2000
      return (1.0/R)*math.log(1.0+R)

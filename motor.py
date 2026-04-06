import numpy as np

R_GAS = 8.31446261815324
MW_AIR = 0.02897
GAMMA_AIR = 1.4
CP_AIR = R_GAS/MW_AIR*(GAMMA_AIR/(GAMMA_AIR - 1))

    
class Species:

    def __init__(self, mw:float, gamma: float):
        self.mw = mw
        self.gamma = gamma
        self.id = 0

    @staticmethod
    def ideal_cp(mw:float, gamma:float):
        return R_GAS/mw*(gamma/(gamma - 1))

    def specific_enthalpy(self, temperature: float) -> float:
        return Species.ideal_cp(self.mw, self.gamma)*temperature
    
    def temperature_from_enthalpy(self, enthalpy: float) -> float:
        return enthalpy/Species.ideal_cp(self.mw, self.gamma)

class Fuel:

    def __init__(self, density: float = 1.0, heating_value: float = 0):
        self.density = density # in kg/m^3
        self.heating_value = heating_value # in J/kg
        self.frozen_flow_products = Species(MW_AIR, GAMMA_AIR)

    def get_burn_rate(self, pressure:float, temperature:float) -> float:
        # in m/s
        return 0
    
class Fuel_StRoberts(Fuel):

    def __init__(self, density: float, heating_value: float, a: float, n: float, a0: float = 0):
        super().__init__(density, heating_value)
        self.a = a # must be in m/s/Pa^n
        self.n = n
        self.a0 = a0

    def get_burn_rate(self, pressure:float, temperature:float) -> float:
        return self.a0 + self.a*((pressure/101325)**self.n)
    

class Fuel_LUT(Fuel):

    def __init__(self, density: float, heating_value: float, pressures: np.ndarray, burn_rates: np.ndarray):
        super().__init__(density, heating_value)
        if(len(pressures) != len(burn_rates)):
            raise ValueError("Burn rates must be same length as pressures")
        if(len(pressures) < 2):
            raise ValueError("At least 2 points needed")
        if not np.all(pressures[1:] > pressures[:-1]):
            raise ValueError("pressures must be monotomically increasing")

        self.pressures = pressures
        self.burn_rates = burn_rates
        self._n = len(pressures)
        self.dbdp = np.zeros(self._n)
        for i in range(1, self._n):
            self.dbdp[i-1] = (self.burn_rates[i] - self.burn_rates[i-1]) / (self.pressures[i] - self.pressures[i-1])
        self.dbdp[-1] = 0 # explicitly setting this for clarity
        self._r = range(self._n-2,0,-1)

    def get_burn_rate(self, pressure:float, temperature:float) -> float:
        if pressure <= self.pressures[0]:
            return self.burn_rates[0]
        elif pressure >= self.pressures[-1]:
            return self.burn_rates[-1]

        idx = 0
        for i in self._r:
            if pressure > self.pressures[i]:
                idx = i
                break

        return self.burn_rates[idx] + (pressure - self.pressures[idx]) * self.dbdp[idx]


class Case:

    def __init__(self):
        self.length:float = 0
        self.radius:float = 0

    def volume(self):
        return np.pi*self.length*self.radius**2
        
class Nozzle:

    NOZZLE_STATE_SUBSONIC = 0
    NOZZLE_STATE_SHOCK_IN_NOZZLE = 1
    NOZZLE_STATE_OVEREXPANDED = 2
    NOZZLE_STATE_UNDEREXPANDED = 3

    def __init__(self, throat_radius, chamber_radius, exit_radius, discharge_coefficient, nozzle_efficiency, 
                shape_x: np.ndarray, shape_radius: np.ndarray, chamber_angle = np.pi/4, chamber_curve_radius_ratio: float = 1.5):

        if throat_radius > chamber_radius or throat_radius > exit_radius:
            raise ValueError("Throat area must be smaller than chamber radius and exit radius")
        
        if len(shape_x) != len(shape_radius):
            raise ValueError("shape_x and shape_radius must be same size")
        
        if chamber_angle <= 0 or chamber_angle >= np.pi/2:
            raise ValueError("chamber angle must be between (0,pi/2)")
        
        self.throat_radius:float  = throat_radius
        self.chamber_radius:float  = chamber_radius
        self.exit_radius:float  = exit_radius
        self.discharge_coefficient:float  = discharge_coefficient
        self.nozzle_efficiency:float  = nozzle_efficiency
        self.chamber_angle:float  = chamber_angle
        self._throat_area:float  = np.pi * self.throat_radius ** 2
        self._chamber_area:float  = np.pi * self.chamber_radius ** 2
        self._exit_area:float  = np.pi * self.exit_radius ** 2
        self._bell_shape_x:np.ndarray  = shape_x
        self._bell_shape_radius:np.ndarray  = shape_radius
        self.chamber_curve_radius_ratio:float  = chamber_curve_radius_ratio
        self._chamber_shape_x = np.zeros(0)
        self._chamber_shape_radius = np.zeros(0)
        self._chamber_shape_x, self._chamber_shape_radius = self.generate_chamber_shape()

    def generate_chamber_shape(self, num_points: int = 20) -> tuple:

        chamber_turn_radius = self.chamber_curve_radius_ratio*self.throat_radius
        turn_ends_at_chamber_radius = False
        if chamber_turn_radius*(1 - np.cos(self.chamber_angle)) >= (self.chamber_radius - self.throat_radius):
            turn_ends_at_chamber_radius = True

        if turn_ends_at_chamber_radius:
            chamber_angle = np.acos(1 - (self.chamber_radius - self.throat_radius)/chamber_turn_radius)
            return Nozzle.generate_turn(self.throat_radius, chamber_angle, 
                                     chamber_turn_radius, num_points - 1)
            
        (x,r) = Nozzle.generate_turn(self.throat_radius, self.chamber_angle, 
                                     chamber_turn_radius, num_points - 1)

        dr = (self.chamber_radius - r[-1])
        drdx = np.tan(self.chamber_angle)
        dx = dr/drdx
        xf = x[-1] + dx

        x = np.append(x, xf)
        r = np.append(r, self.chamber_radius)

        return (-np.flip(x),np.flip(r))

    def chamber_volume(self):
        # rough estimate for now will fix
        sum = 0
        for i in range(1,len(self._chamber_shape_x)):
            dx = self._chamber_shape_x[i] - self._chamber_shape_x[i-1]
            sum += dx*(self._chamber_shape_radius[i]**2 + self._chamber_shape_radius[i-1]**2
                        + self._chamber_shape_radius[i]*self._chamber_shape_radius[i-1])
            
        return sum*np.pi/3
    
    def full_shape(self, num_chamber: int = 12, num_nozzle: int = 24, growth_factor: float = 1.05) -> tuple:
        
        xChamber = np.zeros(num_chamber)
        xBell = np.zeros(num_nozzle+1)
        if abs(growth_factor - 1) > 0.0001:
            a_chamber = self._chamber_shape_x[0]*(1 - growth_factor)/(1 - growth_factor**num_chamber)
            
            xChamber[0] = a_chamber
            for i in range(1, num_chamber):
                xChamber[i] = xChamber[i-1] + a_chamber*growth_factor**i

            xChamber = np.flip(xChamber)

            a_bell = self._bell_shape_x[-1]*(1 - growth_factor)/(1 - growth_factor**num_nozzle)
            xBell[0] = 0
            for i in range(1,num_nozzle+1):
                xBell[i] = xBell[i-1] + a_bell*growth_factor**(i-1)
        else:
            dx = self._chamber_shape_x[0]/num_chamber
            xChamber = np.array(range(num_chamber,1,-1))*dx
            dx = self._bell_shape_x[-1]/num_nozzle
            xBell = np.array(range(num_nozzle))*dx

        rChamber = np.interp(xChamber, self._chamber_shape_x, self._chamber_shape_radius)
        rBell = np.interp(xBell, self._bell_shape_x, self._bell_shape_radius)

        return (np.append(xChamber, xBell), np.append(rChamber, rBell))
    
    def bell_length(self) -> float:
        return self._bell_shape_x[-1]
    
    def exit_area(self) -> float:
        return self._exit_area
    
    def throat_area(self) -> float:
        return self._throat_area
    
    @staticmethod
    def generate_turn(throat_radius: float, angle:float, curvature_radius: float = 0, n_divisions: int = 32) -> tuple:
        angles = np.linspace(0, angle, num=n_divisions)

        if curvature_radius <= 0:
            curvature_radius = throat_radius*0.434

        x = np.zeros(n_divisions)
        r = np.zeros(n_divisions)

        x[0] = 0
        r[0] = throat_radius

        R = throat_radius + curvature_radius
        for i in range(1,n_divisions):
            r[i] = R - np.cos(angles[i])*curvature_radius
            x[i] = np.sin(angles[i])*curvature_radius

        return (x,r)
        
    
    @staticmethod
    def generate_conical(throat_radius:float, exit_radius:float, angle:float, turn_radius: float = 0) -> tuple:
        if angle <= 0:
            raise ValueError("Angle must be greater than 0")

        if turn_radius > 0:
            x, r = Nozzle.generate_turn(throat_radius, angle, turn_radius)

            dr = exit_radius - r[-1]
            dx = dr/np.tan(angle)
            x = np.append(x, x[-1] + dx)
            r = np.append(r, exit_radius)

            return (x,r)
        
        dr = exit_radius - throat_radius
        dx = dr/np.tan(angle)
        
        x = np.array([0, dx])
        r = np.array([throat_radius, exit_radius])
        return (x,r)
    
    @staticmethod
    def rao_turn_angles(length_fraction: float) -> tuple:
        if length_fraction <= 0:
            raise ValueError("percent_nozzle must be greater than 0")

        # Common bell-nozzle length presets used as Rao-style approximations.
        pct = np.array([0.60, 0.80, 0.90])
        turn_deg = np.array([20.0, 30.0, 35.0])
        exit_deg = np.array([25.0, 15.0, 10.0])

        p = np.clip(length_fraction, pct[0], pct[-1])
        turn_angle = np.interp(p, pct, turn_deg) * np.pi / 180.0
        exit_angle = np.interp(p, pct, exit_deg) * np.pi / 180.0
        return (turn_angle, exit_angle)
    
    @staticmethod
    def generate_rao(throat_radius:float, expansion_ratio:float, 
                        length_fraction: float = 0.85, num_divisions: int = 16) -> tuple:

        turn_radius = 0.382*throat_radius

        turn_angle, exit_angle = Nozzle.rao_turn_angles(length_fraction)

        expansion_ratio_term = np.sqrt(expansion_ratio)
        L_N_cone = (expansion_ratio_term - 1)*throat_radius/np.tan(15*np.pi/180)
        L_N = length_fraction*L_N_cone

        x, r = Nozzle.generate_turn(throat_radius, turn_angle, turn_radius)

        Ex = L_N
        Ey = throat_radius*expansion_ratio_term

        Nx = x[-1]
        Ny = r[-1]

        m1 = np.tan(turn_angle)
        m2 = np.tan(exit_angle)

        C1 = Ny - m1*Nx
        C2 = Ey - m2*Ex
        
        Qx = (C2 - C1)/(m1 - m2)
        Qy = (m1*C2 - m2*C1)/(m1 - m2)

        xBell = np.zeros(num_divisions)
        rBell = np.zeros(num_divisions)

        ts = np.linspace(0, 1, num_divisions)
        for i in range(num_divisions):
            pt = 1 - ts[i]
            xBell[i] = pt**2*Nx + ts[i]*(2*pt*Qx + ts[i]*Ex)
            rBell[i] = pt**2*Ny + ts[i]*(2*pt*Qy + ts[i]*Ey)

        return (np.append(x[:-1],xBell), np.append(r[:-1], rBell))
    
    @staticmethod
    def generate_parabolic(throat_radius:float, exit_radius:float, turn_angle: float, turn_radius: float, num_divisions: int = 16) -> tuple:
        
        if turn_radius <= 0:
            turn_radius = throat_radius*0.4

        if turn_angle <= 0:
            turn_angle = 33*np.pi/180.0

        xs, rs = Nozzle.generate_turn(throat_radius, turn_angle, turn_radius)

        r0 = rs[-1]
        x0 = xs[-1]
        drdx = np.tan(turn_angle)

        a = 2*r0*drdx
        b = r0*r0 - a*x0
        xf = (exit_radius**2 - b)/a

        dx = (xf-x0)/num_divisions

        xBell = np.zeros(num_divisions)
        rBell = np.zeros(num_divisions)
        for i in range(num_divisions):
            xBell[i] = x0 + (i + 1)*dx
            rBell[i] = np.sqrt(a*xBell[i] + b)

        xs = np.append(xs, xBell)
        rs = np.append(rs, rBell)
        return (xs,rs)

    @staticmethod
    def critical_pressure_ratio(gamma:float) -> float:
        return (2/(gamma + 1))**(gamma/(gamma - 1))
    
    @staticmethod
    def isentropic_temperature_ratio(mach:float, gamma:float = 1.4) -> float:
        return 1/(1 + (gamma-1)*0.5*mach**2)

    @staticmethod
    def isentropic_pressure_ratio(mach:float, gamma:float = 1.4) -> float:
        return Nozzle.isentropic_temperature_ratio(mach, gamma)**(gamma/(gamma - 1))

    @staticmethod
    def mach_for_isentropic_pressure_ratio(pressure_ratio: float, gamma:float = 1.4) -> float:
        g1 = gamma - 1
        return np.sqrt((2/g1)*(pressure_ratio**(-g1/gamma) - 1))
    
    @staticmethod
    def isentropic_area_ratio(mach: float, gamma: float = 1.4) -> float:
        gp1 = (gamma + 1)*0.5
        gm1 = gamma - 1
        ex = gp1/gm1
        return (((1 + gm1*0.5*mach*mach)/gp1)**ex)/mach
    
    @staticmethod
    def mach_for_supersonic_area_ratio(area_ratio: float, gamma: float = 1.4, n_bisections: int = 16) -> float:
        if area_ratio <= 1:
            raise ValueError("Area ratio must be greater than 1 for supersonic flow")
        
        low = 1.0001
        high = 10
        for i in range(n_bisections):
            mid = (low + high)/2
            if Nozzle.isentropic_area_ratio(mid, gamma) < area_ratio:
                low = mid
            else:
                high = mid

        
        mach_guess = (low + high)/2

        dM = 1e-6
        f_0 = Nozzle.isentropic_area_ratio(mach_guess, gamma) - area_ratio
        f_high = Nozzle.isentropic_area_ratio(mach_guess + dM, gamma) - area_ratio

        dfdM = (f_high - f_0)/dM

        mach_guess -= 0.9*f_0/dfdM
        
        return mach_guess
    
    @staticmethod
    def mach_for_subsonic_area_ratio(area_ratio: float, gamma: float = 1.4, n_bisections: int = 16) -> float:
        if area_ratio <= 1:
            raise ValueError("Area ratio must be greater than 1 for subsonic flow")
        
        low = 1e-6
        high = 0.999999
        for i in range(n_bisections):
            mid = (low + high)/2
            if Nozzle.isentropic_area_ratio(mid, gamma) > area_ratio:
                low = mid
            else:
                high = mid

        return (low + high)/2
    
    @staticmethod
    def downstream_mach_normal_shock(mach: float, gamma: float = 1.4) -> float:
        g1 = gamma - 1
        m2 = mach*mach
        return np.sqrt((g1*m2 + 2)/(2*gamma*m2 - g1))
    
    @staticmethod
    def downstream_pressure_normal_shock(mach: float, gamma: float = 1.4) -> float:
        return (2*mach*mach*gamma - gamma + 1)/(gamma + 1)
    
    @staticmethod
    def ideal_pressure_ratio(area_ratio:float, gamma:float = 1.4) -> float:
        m_exit_ideal = Nozzle.mach_for_supersonic_area_ratio(area_ratio, gamma)
        return Nozzle.isentropic_pressure_ratio(m_exit_ideal, gamma)
    
    @staticmethod
    def shock_at_exit_pressure_ratio(area_ratio:float, gamma:float = 1.4) -> float:
        m_ideal = Nozzle.mach_for_supersonic_area_ratio(area_ratio, gamma)
        p_ratio_ideal = Nozzle.isentropic_pressure_ratio(m_ideal, gamma)
        p_ratio_shock = Nozzle.downstream_pressure_normal_shock(m_ideal, gamma)
        return p_ratio_ideal*p_ratio_shock
    
    @staticmethod
    def choked_pressure_ratio(area_ratio:float, gamma:float = 1.4) -> float:
        m_exit_choked = Nozzle.mach_for_subsonic_area_ratio(area_ratio, gamma)
        return Nozzle.isentropic_pressure_ratio(m_exit_choked, gamma)
    
    @staticmethod
    def ideal_choked_mass_flux(total_pressure:float, total_temperature:float, gamma:float, mw:float) -> float:
        g1 = gamma + 1
        return total_pressure*np.sqrt(gamma*mw/(R_GAS*total_temperature)*(2/g1)**(g1/(gamma - 1)))

    @staticmethod
    def ideal_subsonic_frozen_flow_mass_flux(total_pressure:float, total_temperature:float, gamma:float, mw:float, ambient_pressure:float) -> float:
        R_s = R_GAS/mw
        p_ratio = ambient_pressure/total_pressure
        m_exit = Nozzle.mach_for_isentropic_pressure_ratio(p_ratio, gamma)
        T_exit = total_temperature*Nozzle.isentropic_temperature_ratio(m_exit, gamma)
        rho_exit = ambient_pressure/(R_s*T_exit)
        v_exit = np.sqrt(gamma*R_s*T_exit)*m_exit
        return rho_exit*v_exit
        
    @staticmethod
    def nozzle_condition(total_pressure:float, gamma:float, ambient_pressure:float, area_ratio:float) -> int:
        if ambient_pressure > Nozzle.choked_pressure_ratio(area_ratio, gamma)*total_pressure:
            return Nozzle.NOZZLE_STATE_SUBSONIC
        
        if ambient_pressure > Nozzle.shock_at_exit_pressure_ratio(area_ratio, gamma)*total_pressure:
            return Nozzle.NOZZLE_STATE_SHOCK_IN_NOZZLE
        
        if ambient_pressure > Nozzle.ideal_pressure_ratio(area_ratio, gamma)*total_pressure:
            return Nozzle.NOZZLE_STATE_OVEREXPANDED
        
        return Nozzle.NOZZLE_STATE_UNDEREXPANDED
    


class NozzleMesh:

    def __init__(self, nozzle: Nozzle):
        xs, rs = nozzle.full_shape()
        self.xs = xs
        self.rs = rs
        nx = len(xs)
        self.n_cells = nx - 1

        # geometry
        self.A_face = np.pi*np.square(self.rs) # face area
        self.V = np.zeros(self.n_cells) # volume
        self.dA = self.A_face[1:] - self.A_face[:-1] # change in area
        self.dx = self.xs[1:] - self.xs[:-1]
        self.x_centers = (self.xs[1:] + self.xs[:-1])/2 # x center

        for i in range(self.n_cells):
            self.V[i] = self.dx[i]*(self.rs[i]**2 + self.rs[i+1]**2 + self.rs[i]*self.rs[i+1])

        self.V =(np.pi/3)*self.V

    def num_cells(self) -> int:
        return self.n_cells



class GrainCrossSection:

    def __init__(self):
        pass

    def get_max_burn_length(self) -> float:
        return 0

    def get_cross_section_area(self, burn_length: float) -> float:
        return 0
    
    def get_exposed_length(self, burn_length: float) -> float:
        return 0
    
    def get_second_moments_of_area(self, burn_length: float) -> np.ndarray:
        return np.zeros(2)
    
class HollowCylinderCrossSection(GrainCrossSection):

    def __init__(self, outer_radius: float, inner_radius: float):
        if inner_radius > outer_radius:
            raise ValueError("inner_radius must be smaller than outer radius")
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        self._max_burn_length = outer_radius - inner_radius
        self._r1_sq = outer_radius*outer_radius

    def get_max_burn_length(self) -> float:
        return self._max_burn_length

    def get_cross_section_area(self, burn_length: float) -> float:
        if burn_length > self._max_burn_length:
            return 0
        
        r0 = self.inner_radius + burn_length
        return np.pi*(self._r1_sq - r0*r0)
    
    def get_exposed_length(self, burn_length: float) -> float:
        if burn_length > self._max_burn_length:
            return 0
        
        r0 = self.inner_radius + burn_length
        return np.pi*2*r0
    
    def get_second_moments_of_area(self, burn_length: float) -> np.ndarray:
        if burn_length > self._max_burn_length:
            return np.zeros(2)
        
        r0 = self.inner_radius + burn_length
        I = (np.pi/4)*(self._r1_sq*self._r1_sq - r0**4)
        return np.array((I,I))


class GrainSegment:

    def __init__(self, fuel: Fuel, cross_section: GrainCrossSection, n_exposed_caps: int = 2):
        self.length = 0
        self.fuel: Fuel = fuel
        self.cross_section: GrainCrossSection = cross_section
        self.n_exposed_caps = n_exposed_caps
        self.burning = True
        self._current_burn_length = 0
        self._current_length = 0
        self._current_mass = 0
        self._current_surface_area = 0
        self._current_volume = 0
        self._current_cross_section_area = 0
        self._current_inertia = np.zeros(3)

    def update(self):
        self._current_length = self._compute_length()
        self._current_cross_section_area = self.cross_section.get_cross_section_area(self._current_burn_length)
        self._current_volume = self._current_cross_section_area*self._current_length
        self._current_mass = self._current_volume*self.fuel.density

        self._current_surface_area = self.cross_section.get_exposed_length(self._current_burn_length)*self._current_length
        self._current_surface_area += self.n_exposed_caps*self._current_cross_section_area

        self._current_inertia = self._compute_inertia()

    def init(self):
        self._current_burn_length = 0
        self.update()

    def _compute_length(self) -> float:
        L = self.length - self.n_exposed_caps*self._current_burn_length
        return max(L,0)
    
    def _compute_inertia(self) -> np.ndarray:
        Ix, Iy = self.cross_section.get_second_moments_of_area(self._current_burn_length)
        J = Ix + Iy
        L = self._current_length
        AL_sq = self._current_cross_section_area*L*L/12
        rhoL = self.fuel.density*L
        Ixx = rhoL*(Ix + AL_sq)
        Iyy = rhoL*(Iy + AL_sq)
        Izz = rhoL*J
        return np.array((Ixx, Iyy, Izz)) # kg m2
    
    def current_mass(self) -> float:
        return self._current_mass
    
    def current_volume(self) -> float:
        return self._current_volume
    
    def current_inertia(self) -> np.ndarray:
        return self._current_inertia
    
    def burn(self, pressure: float, temperature: float, dt: float) -> float:
        mass0 = self._current_mass
        self._current_burn_length += self.fuel.get_burn_rate(pressure, temperature) * dt
        self._current_burn_length = min(self._current_burn_length, 
                                        self.cross_section.get_max_burn_length())
        self.update()
        return mass0 - self._current_mass

class Motor:

    def __init__(self, nozzle: Nozzle, case: Case):
        self.segments: list[GrainSegment] = []
        self.nozzle: Nozzle = nozzle
        self.case: Case = case
        self._thrust = []
        self._mass = []
        self._times = []
        self._inertia = []
        self._chamber_pressure = []
        self._chamber_temperature = []
        self._ndata = 0

    def compute_constant_gamma_ideal_gas(self, pressure_ambient: float, temperature_ambient: float, 
                               gamma: float, mw: float, CFL:float = 0.85, dt_record = 1e-2, 
                               record_full_output: bool = False, MAX_STEPS:int = 10000000):
        
        #Check Geometry
        for segment in self.segments:
            segment.init()

        volume_internal = self.case.volume()

        volume_segments = 0
        mass_fuel = 0
        inertia = np.zeros(3)
        for segment in self.segments:
            volume_segments += segment.current_volume()
            mass_fuel += segment.current_mass()
            inertia += segment.current_inertia()

        pressure_chamber = pressure_ambient
        temperature_chamber = temperature_ambient
        volume_chamber = volume_internal - volume_segments

        if volume_chamber <= 0:
            raise ValueError("Initial grain volume cannot exceed internal volume of motor")
        
        # Gas Constants
        R_s = R_GAS/mw
        cv = R_s/(gamma - 1)
        density_ambient = pressure_ambient/(R_s*temperature_ambient) #kg/m^3

        # Chamber Values
        mass_chamber = volume_chamber*density_ambient
        density_chamber = density_ambient
        specific_internal_energy = cv*temperature_ambient
        energy_chamber = mass_chamber*specific_internal_energy

        # Create Mesh
        mesh = NozzleMesh(self.nozzle)

        # Mesh values
        nc = mesh.num_cells() # 2 boundary conditions
        pressure = np.ones(nc)*pressure_ambient # pressure (Pa)
        rho = np.ones(nc)*density_ambient # density (kg/m3)
        E = np.ones(nc)*specific_internal_energy # specific internal energy (J/kg)
        vx = np.zeros(nc) # axial speed (m/s)

        nf = mesh.num_cells() + 1

        Q = np.zeros((3, nf + 1)) # Cell integrated conserved quantities
        speed = np.zeros(nf + 1)
        sound_speed = np.ones(nf + 1)*np.sqrt(gamma*R_s*temperature_ambient)
        F = np.zeros((3, nf + 1)) # Euler Flux 

        interior_idx = range(1, nf)
        left_idx = range(nf)
        right_idx = range(1, nf + 1)
        left_face_idx = range(nc)
        right_face_idx = range(1, nf)

        half_A = 0.5*np.array([mesh.A_face, mesh.A_face, mesh.A_face])

        Q[0,interior_idx] = rho
        Q[1,interior_idx] = rho*vx
        Q[2,interior_idx] = rho*E

        CHAMBER_IDX = 0
        Q[0,CHAMBER_IDX] = density_ambient
        Q[1,CHAMBER_IDX] = 0
        Q[2,CHAMBER_IDX] = density_ambient*specific_internal_energy

        Q[0,-1] = density_ambient
        Q[1,-1] = 0
        Q[2,-1] = density_ambient*specific_internal_energy
        
        # Constants
        const_p = gamma - 1
        const_a = gamma*const_p
        min_energy = np.ones(nc)*1e-6
        chamber_net_A = np.pi*self.nozzle.chamber_radius**2

        self._times = [0.0]
        self._mass = [mass_fuel]
        self._inertia = [inertia]
        self._thrust = [0.0]
        self._chamber_pressure = [pressure_chamber]
        self._chamber_temperature = [temperature_chamber]
        self._cells = []
        
        burnout = False
        time_burnout = 0.0
        time = 0.0
        time_record = dt_record
        for i in range(MAX_STEPS):

            # Reconstruct primatives
            rho = Q[0,interior_idx]
            vx = Q[1,interior_idx]/rho
            E = Q[2,interior_idx]/rho
            
            ke = vx*vx*0.5
            e = np.maximum(E - ke, min_energy)  # Ensure non-negative energy
            pressure = const_p*rho*e
            sound_speed[interior_idx] = np.sqrt(const_a*e)
            speed[interior_idx] = np.abs(vx)
            speed[-1] = speed[-2]
            sound_speed[-1] = sound_speed[-2]
            lax = sound_speed + speed

            # CFL Condition
            dt = min(CFL*np.min(mesh.dx/lax[interior_idx]), dt_record)
            time += dt

            # Cell conserved quantities
            momentum = rho*vx
            energy = rho*E

            # Note these are temporarily used with intrinsic values
            Q[0,interior_idx] = rho
            Q[1,interior_idx] = momentum
            Q[2,interior_idx] = energy

            # Face values for flux
            F[0,interior_idx] = momentum
            F[1,interior_idx] = momentum*vx + pressure
            F[2,interior_idx] = vx*(energy + pressure)

            # Exit boundary
            Q[:,-1] = Q[:,-2]
            F[:,-1] = F[:,-2]
            if vx[-1] < sound_speed[-1]:
                dp = pressure_ambient - pressure[-1]
                F[1,-1] += dp
                F[2,-1] += vx[-1]*dp

            # Chamber Boundary (temporarily use intrinsic values)
            Q[0,CHAMBER_IDX] = density_chamber
            Q[2,CHAMBER_IDX] = energy_chamber/volume_chamber

            # Chamber velocity gradient is zero (BC) to allow for proper mass flow out of chamber
            F[0,0] = density_chamber*vx[0]
            F[1,0] = density_chamber*vx[0]*vx[0] + pressure_chamber
            F[2,0] = vx[0]*(Q[2,CHAMBER_IDX] + pressure_chamber) 

            # Lax Friedrich
            s = np.maximum(lax[left_idx], lax[right_idx])
            face_flux = half_A*((F[:,left_idx] + F[:,right_idx]) - s*(Q[:,right_idx] - Q[:,left_idx]))

            # Cell Step Residual
            R = face_flux[:,left_face_idx] - face_flux[:,right_face_idx]
            R[1,:] = R[1,:] + pressure*mesh.dA # Momentum Source term. Note dA is A_{i+1/2} - A_{i-1/2}
            R = R/mesh.V

            # Euler Step, note we're ignoring chamber and exit conditions 
            Q[:,interior_idx] = Q[:,interior_idx] + R*dt

            # TODO: higher order integration like rk4?

            # chamber
            dm_in = 0
            de_in = 0
            if not burnout:
                volume_segments = 0
                mass_fuel = 0
                inertia = np.zeros(3)            
                for segment in self.segments:
                        dm = segment.burn(pressure_chamber, temperature_chamber, dt)
                        de = dm*segment.fuel.heating_value
                        dm_in += dm
                        de_in += de

                        volume_segments += segment.current_volume()
                        mass_fuel += segment.current_mass()
                        inertia += segment.current_inertia()

                volume_chamber = volume_internal - volume_segments

                if mass_fuel <= 0:
                    time_burnout = time
                    burnout = True
                    volume_chamber = volume_internal
                    mass_fuel = 0

            else:

                if vx[-1] < 1 or thrust <= 0:
                    break

            mass_chamber += dm_in - face_flux[0,0]*dt
            energy_chamber += de_in - face_flux[2,0]*dt

            if mass_chamber <= 0:
                print("Oops, somehow we achieved negative masses")
                break

            density_chamber = mass_chamber/volume_chamber
            pressure_chamber = const_p*energy_chamber/volume_chamber
            specific_internal_energy = energy_chamber/mass_chamber
            temperature_chamber = specific_internal_energy/cv
            
            # Set this to integrated values
            Q[0,CHAMBER_IDX] = density_chamber
            Q[2,CHAMBER_IDX] = density_chamber*specific_internal_energy

            if time > time_record:

                print("Time: ", time)
                self._times.append(time)
                self._mass.append(mass_fuel)
                self._inertia.append(inertia)
                self._chamber_pressure.append(pressure_chamber)
                self._chamber_temperature.append(temperature_chamber)

                guage_pressure = pressure - pressure_ambient
                thrust = np.sum(guage_pressure*mesh.dA) + (pressure_chamber - pressure_ambient)*chamber_net_A
                # thrust is entirely driven by pressure on body, ie, no viscous forces
                self._thrust.append(thrust)
                print("Thrust: ", thrust)
                print("Mass Fuel: ", mass_fuel)
                if record_full_output:
                    self._cells.append(Q.copy())

                time_record += dt_record

        if record_full_output:
            self._cells.append(Q.copy())
            
        print("Time Burnout")
        print(time_burnout)
                
    def thrusts(self) -> list:
        return self._thrust
    
    def masses(self) -> list:
        return self._mass
    
    def times(self) -> list:
        return self._times
    
    def inertias(self) -> list:
        return self._inertia
    
    def chamber_pressures(self) -> list:
        return self._chamber_pressure
    
    def chamber_temperatures(self) -> list:
        return self._chamber_temperature
    
    def cells(self) -> list[np.ndarray]:
        return self._cells

    def compute(self, pressure_ambient: float, temperature_ambient: float, dt:float = 1e-4, MAX_STEPS:int = 1000000):
        density_ambient = pressure_ambient*MW_AIR/(R_GAS*temperature_ambient) #kg/m^3

        for segment in self.segments:
            segment.init()

        volume_internal = self.case.volume() + self.nozzle.chamber_volume()

        species = [Species(MW_AIR, GAMMA_AIR)]
        species_id = [0]
        species_fraction = [1]
        volume_segments = 0
        mass_fuel = 0
        inertia = np.zeros(3)
        for segment in self.segments:
            volume_segments += segment.current_volume()
            mass_fuel += segment.current_mass()
            inertia += segment.current_inertia()
            if segment.fuel.frozen_flow_products.id not in species_id:
                species_id.append(segment.fuel.frozen_flow_products.id)
                species.append(segment.fuel.frozen_flow_products)
                species_fraction.append(0)

        
        # TODO



        

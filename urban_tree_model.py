'''
Paul Voit, 01 May 2025
This function follows the paper of Tams et al. 2024 and 2023 (Urban tree drought stress: Sap flow measurements, model validation, and
water management simulations & Impact of shading on evapotranspiration and water stress of
urban trees) which contained a full model description and the parameters.

With this model we can calculate the water budget of a tree including interception, infiltration,
runoff, seasonal leaves, water stress etc.

Additionally I implemented an irrigation function.

Todo: Constant soil evaporation could be added as in Urban Tree Drought Stress (Tams et. al)
'''
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

#todo add pheno multiplier

#Todo: add area/volume of tree and pit
# add irrigation input option
class Tree:
    def __init__(self, age):

        #we assume that all the street trees grow in compacted soil
        #values from Predicting Water Supply and Evapotranspiration of Street Trees Using Hydro-Pedo-Transfer Functions (HPTFs),
        #Gerd Wessolek and Björn Kluge 2021. Table 6

        self.age = age

        #get rooting depth and crown protection area according to age
        if age == "young":
            rooting_depth = np.arange(0.3, 0.75, 0.05).tolist()
            self.rooting_depth = np.round(random.sample(rooting_depth, 1)[0], 2)
            self.cpa = random.randint(2, 12)

        if age == "medium":
            rooting_depth = np.arange(0.7, 1.3, 0.05).tolist()
            self.rooting_depth = np.round(random.sample(rooting_depth, 1)[0], 2)
            self.cpa = random.randint(13, 50)

        if age == "old":
            rooting_depth = np.arange(1.2, 2.05, 0.05).tolist()
            self.rooting_depth = np.round(random.sample(rooting_depth, 1)[0], 2)
            self.cpa = random.randint(51, 100)


        self.soil = random.sample(["med_sandy", "fine_sandy", "loamy_sand", "mineral_substrate"], 1)[0]

        #get soil available water old version
        # if self.soil == "med_sandy":
        #     self.saw = random.sample(np.arange(8, 12.2, 0.2).tolist(), 1)[0]
        #
        # elif self.soil == "fine_sandy":
        #     self.saw = random.sample(np.arange(12, 15.2, 0.2).tolist(), 1)[0]
        #
        # elif self.soil == "loamy_sand":
        #     self.saw = random.sample(np.arange(12, 16.2, 0.2).tolist(), 1)[0]
        #
        # elif self.soil == "sandy_loam":
        #     self.saw = random.sample(np.arange(13, 17.2, 0.2).tolist(), 1)[0]
        #
        # elif self.soil == "mineral_substrate":
        #     self.saw = random.sample(np.arange(21, 25, 1).tolist(), 1)[0]

        #this is all in Vol% saw = Soil Available Water (water between pF 1.8-4.2), basically the nFK (nutzbare Feldkapazität)
        #taken from Wessolek & Kluge Predicting Water Supply and Evapotranspiration of Street Trees Using Hydro-Pedo-Transfer Functions (HPTFs)
        if self.soil == "med_sandy":
            self.saw = random.sample(np.arange(8, 12.2, 0.2).tolist(), 1)[0]
            self.pwp = 40 #mm

        elif self.soil == "fine_sandy":
            self.saw = random.sample(np.arange(12, 15.2, 0.2).tolist(), 1)[0]
            self.pwp = 50 #mm

        elif self.soil == "loamy_sand":
            self.saw = random.sample(np.arange(12, 16.2, 0.2).tolist(), 1)[0]
            self.pwp = 60 #mm

        elif self.soil == "mineral_substrate":
            self.saw = random.sample(np.arange(21, 25, 1).tolist(), 1)[0]
            self.pwp = 30 #mm

        self.kc = np.round(np.random.normal(0.6, 0.05, 1)[0], 2)

        self.k_ea = np.round(random.sample(np.arange(0.4,  0.62, 0.02).tolist(), 1)[0], 2)
        self.k_s = random.sample(np.arange(0.2, 0.42, 0.02).tolist(), 1)[0]

        #because we're going to milimeters, we're multiplying by 10
        self.fc = np.round(self.saw * 10 + self.pwp, 1)

        self.surface_sealing = np.round(random.sample(np.arange(0, 1.05, 0.05).tolist(), 1)[0], 2)

        self.sealing_class = random.sample([1, 2, 3, 4], 1)[0]

        self.start_water_content = np.round(random.sample(np.arange(0.1, 1.05, 0.05).tolist(), 1)[0], 2)

        self.sky_view_factor = np.round(np.random.normal(1, 0.125, 1)[0], 2)

        self.c_inf = np.round(random.sample(np.arange(17, 24, 1).tolist(), 1)[0])



def get_pheno_multiplier(doy):

    '''
    Calculates the fraction of leaves on the tree depending on the day of year.

    :param doy: np.array containing the day of year
    :return: np.array with pheno multipliers
    '''

    if doy < 114:
        cp = 0
    elif ((doy >= 114) and (doy < 128)):
        cp = 0.5
    elif ((doy >= 128) and (doy < 274)):
        cp = 1
    elif ((doy >= 274) and (doy < 288)):
        cp = 0.5
    elif doy >= 274:
        cp = 0

    return cp

def calc_interception(rain, et_pot, doy, l_max=10, c_i=0.5):
    '''
    Calculates the interception storage and effective rainfall (available for infiltration) for a given day of year.
    :param rain: np.array containing rainfall values
    :param et_pot: np.array containing values for potential evapotranspiration. These were calculated with
    the package pyet in the script et_pot_package.py.
    :param doy: np.array containing the day of year
    :param l_max: int/float, maximum interception storage in [mm]
    :param c_i: float/int, interception coefficient
    :return: np.array of interception storage, effective rainfall and interception losses (for the water balance) [mm]
    '''

    #print("Calculating interception")
    current_day = doy[0]
    c_p = get_pheno_multiplier(current_day)
    interception = np.zeros_like(rain)
    eff_rain = np.zeros_like(rain)
    interception_loss = np.zeros_like(rain)

    if rain[0] > l_max:
        interception[0]  = l_max
        eff_rain[0] = np.round(rain[0] - l_max, 1)

    else:
        interception[0] = rain[0] * c_p * c_i
        eff_rain[0] = rain[0] - interception[0]
        interception[0] = np.max([0, interception[0] - et_pot[0]]) #check this: how quickly would it be empty

    for i in range(1, len(rain)):

        current_day = doy[i]
        prev_interception = interception[i - 1]

        if rain[i] > 0:
            c_p = get_pheno_multiplier(current_day)
            pot_interception = rain[i] * c_p * c_i
            available_storage = l_max - prev_interception

            if pot_interception > available_storage:
                interception[i] = l_max
                eff_rain[i] = rain[i] - available_storage
                interception_loss[i] = rain[i] - eff_rain[i]

            else:
                interception[i] = np.max([0, prev_interception + pot_interception]) #subtracting Et_pot here
                eff_rain[i] = rain[i] - pot_interception
                interception_loss[i] = rain[i] - eff_rain[i]

        #finally I will take out Et_pot. This could be argued, but the daily data
        # does not give information about the sunny hours and when the rain actually did fall
        interception[i] = np.max([0, interception[i] - et_pot[i]])


    return interception, eff_rain, interception_loss

def calc_surface_runoff(eff_rain, c_r=0.2, c_rt=10):
    '''
    Calculates infiltration excess runoff
    :param eff_rain: np.array from calculate_interception()
    :param c_r: float/int, runoff coefficient
    :param c_rt: float/int, runoff threshold
    :return: np.array with surface runoff [mm]
    '''
    #print("Calculating surface_runoff")
    surface_runoff = np.zeros_like(eff_rain)

    surface_runoff[eff_rain < c_rt] = 0
    surface_runoff[eff_rain >= c_rt] = eff_rain[eff_rain >= c_rt]* c_r #check and compare to old loop version

    return surface_runoff

def calc_infiltration(eff_rain, surface_runoff, c_inf=20.0):
    '''
    Calculates infiltration into soil
    :param eff_rain: np.array of effective rainfall from calc_interception()
    :param surface_runoff: np.array of runoff from calc_surface_runoff()
    :param c_inf: float/int, infiltration coefficient
    :return: np.array of infiltration [mm]
    '''
    #print("Calculating infiltration")
    infiltration = np.zeros_like(eff_rain)

    for i in range(len(infiltration)):

        pot_infiltration = eff_rain[i] - surface_runoff[i]
        infiltration[i] = np.min([c_inf, pot_infiltration])

    return infiltration

def check_water_stress(swc, pwp, afc):

    #scale swc back to 1m root depth to compare with PWP
    swc = np.round(swc, 1)

    if swc < pwp + 0.3 * afc:
        if swc <= (pwp + 0.3): #add a little buffer of 0.3 mm
            return 2
        else:
            return 1
    else:
        return 0

def calc_et_actual_and_swc(et_pot, infiltration, datetime_array, rain,
                           k_c=0.6, k_ea=0.5, k_s=0.3, fc=220, pwp=40,
                           crown_area=4, surface_sealing=0, sealing_class=2, rooting_depth=0.8, start_water_content=0.5,
                           cistern_volume=2000, cistern_catchment=20, loss_factor_in=0.5,
                           loss_factor_out=0.2, irrigation_rate=300):
    '''
    This function follows the paper of Tams et al. 2024 and 2023 (Urban tree drought stress: Sap flow measurements, model validation, and
    water management simulations & Impact of shading on evapotranspiration and water stress of
    urban trees) which contained a full model description and the parameters.

    :param et_pot: np.array, potential evaporation [mm]
    :infiltration: np.array, infiltration [mm] from calc_infiltration().
    :param k_c: crop coefficient for calculating the actual evapotranspiration according to FAO
    :param datetime_array: np.array, dates of et_pot in rain in datetime format
    :param k_ea: E_ta reduction coefficient during water stress
    :param k_s: water stress_coefficient
    :param fc: field capacity [mm]: Often its expressed in  [m³/m³] which is Vol %. So if we assume
    a cubic meter e.g. 26 m³/m³ actually mean 26% are water, for one cubic meter that would be 260 mm. If the soil
    is deeper than a meter, than we need to multiply this value
    :param pwp: permanent wilting point [mm]. Explanation see above
    :param crown_area: [m²] area of the crown. We need to know this, because tree crown and pit surface might be different,
    which means that the tree has more surface for evapotranspiration than for infiltration.
    :param surface_sealing, [-] fraction of crown area that is sealed
    :param sealing_class [1,2,3,4]: describes which material is used for sealing. there are for classes:
    1: not really sealed, 2:small cobblestones, 3: large "Gehwegplatten", 4: asphalt. This follows Predicting Water
    Supply and Evapotranspiration of Street Trees Using Hydro-Pedo-Transfer Functions (HPTFs), Gerd Wessolek and Björn Kluge 2021.
    Table 6. This accounts for the fact, that even though if the surface is sealed, some water might be still able to
    infiltrate.
    :param rooting_depth: [m] soil depth of the pit. Needed to calculated the PWP and field capacity.
    :param start_water_content [-]. Fraction of the field capacity that should be filled at simulation start.
    :param cistern_volume  [mm] if not == 0, this water will get used for irrigation as soon as water stress level 2 is reached.
    :param cistern_catchment, [m²] area that the green infrastructure can catch rainfall from. This could be e.g.
    a green roof
    :param loss_factor_in [-] fraction of water lost during the collection of rainfall. E.g. 0.5 for extensive green roof,
    0.8 for normal roof. Source: Grüter, Planung, Ausführung, Betrieb und Wartung, Teil 1
    Regenwassernutzung nach DIN 1989
    :param loss_factor_out [-] fraction of water lost during the irrigtation process.
    :param irrigation_rate: [l] amount of water that gets irrigated, once water stress level 2 is reached.

    :return: np.arrays of actual evaporation [mm], soil water content [mm/m], percolation [mm], water stress level, cistern
    filling and irrigation water,
    Water stress is classified following the advice of the Berliner Senat: Irrigation recommended from PWP + 0.3 * nFK
    https://www.berlin.de/pflanzenschutzamt/stadtgruen/beratung/bewaesserungsempfehlung-fuer-stadtbaeume/
    '''

    #here we work with volumes, because tree pit and crown have different areas
    #print("Calculating soil water content and actual evapotransiration")
    et_a = np.zeros_like(et_pot)
    swc = np.zeros_like(et_pot)
    percolation = np.zeros_like(et_pot)
    water_stress = np.zeros_like(et_pot) #0: no stress, 1 & 2: water stress, 3: extreme stress (PWP)
    cistern = np.zeros_like(et_pot)
    irrigation = np.zeros_like(et_pot)

    doy = pd.to_datetime(datetime_array).dayofyear.to_numpy()
    beta_table = {1: (0.9, 0.95), 2: (0.8, 0.85), 3: (0.55, 0.6), 4: (0.2, 0.25)}

    pwp = pwp * rooting_depth
    fc = fc * rooting_depth
    afc = fc - pwp #available field capacity

    et_pot[0] = 0
    water_stress[0] = 0
    cistern[0] = rain[0] * cistern_catchment * (1-loss_factor_in)

    #because tree pit and crown have different areas, we need to compute a factor here
    sealed_area = crown_area * surface_sealing
    infiltration_area = crown_area * (1- surface_sealing)

    #start_water_content *= rooting_depth Bug that Sophia found
    swc[0] = start_water_content * fc

    water_stress_counter = 0 #we irrigate after 1 week water stress level 2
    water_stress_days = np.zeros_like(et_pot)

    infiltration_area_t = (beta_table[sealing_class][0] * sealed_area) + infiltration_area
    area_factor_t = crown_area / infiltration_area_t


    for i in range(1, len(et_a)):

        #set the seasonally depending infiltration area
        if doy[i] == 100:
            infiltration_area_t = (beta_table[sealing_class][0] * sealed_area) + infiltration_area
            area_factor_t = crown_area / infiltration_area_t

        if doy[i] == 274:
            infiltration_area_t = (beta_table[sealing_class][1] * sealed_area) + infiltration_area
            area_factor_t = crown_area / infiltration_area_t

        #fill the cistern
        cistern_input = rain[i] * cistern_catchment * (1-loss_factor_in)
        cistern[i] = cistern[i-1] + cistern_input

        if cistern[i] > cistern_volume:
            cistern[i] = cistern_volume

        c_p = get_pheno_multiplier(doy[i])

        if i == 41617:
            print("Stop")

        #okay first we calculate et_a with the soil water content from the timestep before, then we update the soil water content
        #here its a chicken egg situation, I take the one from before.

        #if there are leaves, then there is (evapo)transpiration
        if c_p > 0:
            if swc[i-1] - pwp <= 0:
                #No water available for Et
                et_a[i] = 0

            elif swc[i-1] < (pwp + k_s * afc):
                # water stress situation
                et_ax = (c_p * k_c * et_pot[i] * k_ea) / area_factor_t

                if et_ax > swc[i-1] - pwp:
                    et_a[i] = (swc[i-1] - pwp) / area_factor_t
                else:
                    et_a[i] = et_ax

            else:
                #this is the "no-water-stress" situation
                et_ax = (c_p * k_c * et_pot[i]) / area_factor_t

                #just to make sure that we don't go over the PWP
                if et_ax > swc[i-1] - pwp:
                    et_a[i] =(swc[i-1] - pwp) / area_factor_t
                else:
                    et_a[i] = et_ax

        else:
            et_a[i] = 0

        #update field capacity
        delta_swc = infiltration[i] - et_a[i]

        if delta_swc + swc[i-1] > fc:
            swc[i] = fc
            percolation[i] = delta_swc + swc[i-1] - fc

        else:
            swc[i] = np.max([pwp, delta_swc + swc[i-1]])

        water_stress[i] = check_water_stress(swc[i], pwp, afc)


        if water_stress[i] == 2:
            water_stress_counter += 1
        else:
            water_stress_counter = 0


        #irrigation
        #irrigate if we have since one week water stress level 2
        if irrigation_rate > 0:
            if ((water_stress[i] == 2) and (water_stress_counter >= 24 * 7)):

                if cistern[i] >= irrigation_rate:
                    swc[i] = swc[i] + (irrigation_rate * (1 - loss_factor_out)) / infiltration_area_t
                    water_stress[i] = check_water_stress(swc[i], pwp, afc)
                    irrigation[i] = irrigation_rate * (1 - loss_factor_out)
                    cistern[i] = cistern[i] - irrigation_rate

                    if water_stress[i] == 2:
                        water_stress_counter += 1
                    else:
                        water_stress_counter = 0


                else:
                    swc[i] = swc[i] + (cistern[i] * (1 - loss_factor_out)) / infiltration_area_t
                    water_stress[i] = check_water_stress(swc[i], pwp, afc)
                    irrigation[i] = cistern[i] * (1 - loss_factor_out)
                    cistern[i] = 0

                    if water_stress[i] == 2:
                        water_stress_counter += 1
                    else:
                        water_stress_counter = 0

                if water_stress_counter >= (24 * 7):
                    water_stress_days[i] = 1

        else:
            if water_stress_counter >= (24 * 7):
                water_stress_days[i] = 1

        #substract constant soil evaporation of 0.3mm like in Tams et. al "Urban Tree Drought Stress"
        swc[i] = np.max([pwp, swc[i]-(0.3/24)])


    #The soil water content is now scaled to the rooting depth, so it needs to be divided again by it
    swc = np.round((swc / rooting_depth), 1)


    return et_a, swc, percolation, water_stress_days, cistern, irrigation


def urban_tree(et_pot, rain, datetime_array,
               l_max=10, c_i=0.5, c_r=0.2, c_rt=10, c_inf=20.0,
               k_c=0.6, k_ea=0.5, k_s=0.3, fc=220, pwp=40,
               crown_area=4, surface_sealing=0, sealing_class=2, rooting_depth=0.8, start_water_content=0.5,
               cistern_volume=0, cistern_catchment=0, loss_factor_in=0.5, loss_factor_out=0.2,
               irrigation_rate=200, sky_view_factor=1):

    '''
    Just a wrapper for the functions above. See description of the paramters in the individual functions
    :param et_pot: np.array of pot evaporation [mm]
    :param rain: np.array of rain [mm]
    :param datetime_array: np.array, dates of et_pot in rain in datetime format
    :param l_max: maximum interception storage [mm]
    :param c_i: interception coefficient
    :param c_r: runoff coefficient
    :param c_rt: runoff threshold
    :param c_inf: infiltration coefficient
    :param k_c: crop coefficient to calculate actual evapotranspiration
    :param k_ea: water stress coefficient
    :param k_s: water stress factor
    :param fc: field capacity [mm/m]
    :param pwp: permanent wilting point [mm/m]
    :param crown_area: in m²
    :param surface_sealing: fraction of the crown area that is sealed and available for infiltration
    :param rooting_depth: in m
    :param start_water_content: fraction of water content and simulation start
    :param loss_factor_in [-] fraction of water lost during the collection of rainfall. E.g. 0.5 for extensive green roof,
    0.8 for normal roof. Source: Grüter, Planung, Ausführung, Betrieb und Wartung, Teil 1
    Regenwassernutzung nach DIN 1989
    :param loss_factor_out [-] fraction of water lost during the irrigtation process.
    :param irrigation_rate: [mm] amount of water that gets irrigated, once water stress level 2 is reached.
    :param sky_view_factor [-]: Adjust potential to urban environment (shading, advection etc.) according to
    Predicting Water Supply and Evapotranspiration of Street Trees Using Hydro-Pedo-Transfer Functions (HPTFs),
    Gerd Wessolek and Björn Kluge 2021. The value should range between 0.5 and 1.4


    :return: np.arrays of actual evaporation [mm], soil water content [mm/m], percolation [mm],water stress level, cistern
    filling and irrigation water,
    :return:
    '''


    doy = pd.to_datetime(datetime_array).dayofyear.to_numpy()

    interception, eff_rain, interception_loss = calc_interception(rain, et_pot * sky_view_factor, doy,
                                                                  l_max=l_max, c_i=c_i)
    surface_runoff = calc_surface_runoff(eff_rain, c_r=c_r, c_rt=c_rt)
    infiltration = calc_infiltration(eff_rain, surface_runoff, c_inf=c_inf)
    et_a, swc, percolation, water_stress_days, cistern, irrigation = calc_et_actual_and_swc(et_pot * sky_view_factor,
                                                                  infiltration, datetime_array, rain,
                                                                  k_c=k_c, k_ea=k_ea, k_s=k_s, fc=fc, pwp=pwp,
                                                                  crown_area=crown_area, surface_sealing=surface_sealing,
                                                                  sealing_class=sealing_class,
                                                                  rooting_depth=rooting_depth,
                                                                  start_water_content=start_water_content,
                                                                  cistern_volume=cistern_volume,
                                                                  cistern_catchment=cistern_catchment,
                                                                  loss_factor_in=loss_factor_in,
                                                                  loss_factor_out=loss_factor_out,
                                                                  irrigation_rate=irrigation_rate)

    df = pd.DataFrame({"date": datetime_array, "rain_mm": rain,
                       "interception_storage_mm": interception, "interception_loss_mm": interception_loss,
                       "eff_rain_mm": eff_rain,
                       "surface_runoff_mm": surface_runoff, "infiltration_mm": infiltration,
                       "et_a_mm": et_a, "et_pot_mm": et_pot, "soil_water_content_mm/m": swc, "percolation_mm": percolation,
                       "water_stress": water_stress_days, "cistern_volume_mm": cistern, "irrigation_mm": irrigation})

    return df



def make_figure(plot_df):
    #plot rainfall and soil moisture content
    fig, (ax1, ax3) = plt.subplots(2, 1, sharex=True, figsize=(30, 12))
    ax1.set_ylabel('mm/d', color='green')
    ax1.plot(plot_df.date, plot_df.et_a, color='green', label='$ET_{a}$')
    ax1.tick_params(axis='y', labelcolor='green')

    # Plot rainfall as positive bars
    ax2 = ax1.twinx()
    ax2.set_xlabel('Day')
    ax2.set_ylabel('mm/d', color='blue')
    bars = ax2.bar(plot_df.date, plot_df.rain, color='blue', alpha=0.6, label='Niederschlag')
    #ax2.plot(plot_df.date, plot_df.soil_water_content, color='red', label='Soil Moisture')
    ax2.set_ylim((0,60))
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.invert_yaxis()  # Invert y-axis so bars drop from the top
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # Change the line width just in the legend
    legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=20)
    for legline in legend.get_lines():
        legline.set_linewidth(3)

    ax3.set_ylabel('mm')
    ax3.plot(plot_df.date, plot_df.soil_water_content, label='Bodenwassergehalt', color="black")
    ax3.tick_params(axis='y')

    ax4 = ax3.twinx()
    ax4.set_xlabel('Day')
    ax4.set_ylabel('mm/d', color='blue')
    bars = ax4.bar(plot_df.date, plot_df.rain, color='blue', alpha=0.6, label='Niederschlag')
    ax4.set_ylim((0,60))
    ax4.tick_params(axis='y', labelcolor='blue')
    ax4.invert_yaxis()  # Invert y-axis so bars drop from the top
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax4.get_legend_handles_labels()
    legend = ax3.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=20)
    for legline in legend.get_lines():
        legline.set_linewidth(3)

    #Add background color based on water stress
    for i in range(len(plot_df) - 1):
        start = plot_df.date.iloc[i]
        end = plot_df.date.iloc[i + 1]
        cat = plot_df.water_stress.iloc[i]

        if cat == 0:
            color = 'green'
        # elif 1 <= cat < 12:
        #     color = 'yellow'
        elif cat >= 12:
            color = 'red'

        ax3.axvspan(start, end, facecolor=color, alpha=0.3)


    fig.tight_layout()
    plt.grid(True)
    plt.show()



